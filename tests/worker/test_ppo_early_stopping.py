"""Unit tests for PPO KL-based early stopping."""

import multiprocessing
import os
import socket
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch.distributed as dist

pytest.importorskip("trl", reason="trl not installed (needs --extra training)")

from transformers import TrainerControl, TrainerState, TrainingArguments

from worker.content.artifacts import fail_spool, open_spool
from worker.executors.base_executor import ExecutionError
from worker.executors.ppo_executor import (
    PPOExecutor,
    _CheckpointPublishStop,
    _EarlyStopPPOTrainer,
    _EarlyStopSignal,
)
from worker.executors.utils.checkpoint_spool import CheckpointSpool


def _make_trainer(
    target_kl: float | None, callbacks: list[Any] | None = None
) -> _EarlyStopPPOTrainer:
    """Build a bare ``_EarlyStopPPOTrainer`` without running ``__init__``.

    ``_EarlyStopPPOTrainer.log`` defers to ``super().log`` (TRL/HF Trainer); the
    tests patch that with a no-op so they only exercise our override.
    """
    trainer = _EarlyStopPPOTrainer.__new__(_EarlyStopPPOTrainer)
    trainer.target_kl = target_kl
    trainer.callback_handler = cast(Any, SimpleNamespace(callbacks=callbacks or []))
    return trainer


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_kl_above_target_raises(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    with pytest.raises(_EarlyStopSignal):
        trainer.log({"objective/kl": 0.2})


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_kl_below_target_does_not_raise(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    trainer.log({"objective/kl": 0.05})


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_kl_equal_to_target_does_not_raise(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    trainer.log({"objective/kl": 0.1})


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_missing_kl_key_is_ignored(_super_log) -> None:
    trainer = _make_trainer(target_kl=0.1)
    trainer.log({"loss": 1.0})


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_threshold_unset_is_no_op(_super_log) -> None:
    """When ``target_kl is None`` the override is a pure pass-through."""
    trainer = _make_trainer(target_kl=None)
    trainer.log({"objective/kl": 9.99})


def _spool(tmp_path: Path, failed: bool) -> CheckpointSpool:
    """A spool whose save has run, after a publication failed when ``failed``."""
    spool = tmp_path / "spool"
    spool.mkdir()
    open_spool(spool)
    if failed:
        fail_spool(spool, "store unreachable")
    callback = CheckpointSpool(spool)
    callback.on_save(
        TrainingArguments(output_dir=(tmp_path / "out").as_posix(), report_to=[]),
        TrainerState(global_step=1),
        TrainerControl(),
    )
    return callback


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_a_failure_after_the_last_save_waits_for_the_next_save(
    _super_log: Any, tmp_path: Path
) -> None:
    callback = _spool(tmp_path, False)
    fail_spool(tmp_path / "spool", "store unreachable")
    trainer = _make_trainer(target_kl=None, callbacks=[callback])

    trainer.log({"objective/kl": 0.01})


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_a_failed_checkpoint_publication_stops_training(
    _super_log: Any, tmp_path: Path
) -> None:
    trainer = _make_trainer(target_kl=None, callbacks=[_spool(tmp_path, True)])
    with pytest.raises(_CheckpointPublishStop):
        trainer.log({"objective/kl": 0.01})


@patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True)
def test_training_goes_on_while_checkpoints_publish(
    _super_log: Any, tmp_path: Path
) -> None:
    trainer = _make_trainer(target_kl=None, callbacks=[_spool(tmp_path, False)])
    trainer.log({"objective/kl": 0.01})


# ---------------------------------------------------------------------------
# _install_kl_early_stopping activation rules
# ---------------------------------------------------------------------------


def _install(training_config: dict[str, Any]) -> _EarlyStopPPOTrainer:
    """Run the installer against a bare trainer and return it."""
    executor = PPOExecutor.__new__(PPOExecutor)
    trainer = _make_trainer(target_kl=None)
    executor._install_kl_early_stopping(trainer, training_config)
    return trainer


def test_install_no_op_when_flag_missing() -> None:
    trainer = _install({"target_kl": 0.1})
    assert trainer.target_kl is None


@pytest.mark.parametrize("flag", [False, "false", "False", 0, "no", "off"])
def test_install_no_op_when_flag_disabled(flag: Any) -> None:
    trainer = _install({"early_stopping": flag, "target_kl": 0.1})
    assert trainer.target_kl is None


@pytest.mark.parametrize("flag", [True, "true", "True", 1, "yes", "on"])
def test_install_arms_when_flag_enabled(flag: Any) -> None:
    trainer = _install({"early_stopping": flag, "target_kl": 0.1})
    assert trainer.target_kl == pytest.approx(0.1)


@pytest.mark.parametrize("bad_target", [None, 0, -0.5])
def test_install_rejects_enabled_without_positive_target(bad_target: Any) -> None:
    cfg: dict[str, Any] = {"early_stopping": True}
    if bad_target is not None:
        cfg["target_kl"] = bad_target
    with pytest.raises(ExecutionError):
        _install(cfg)


def _unwinds(rank: int, port: int, root: str, out: Any) -> None:
    """Train two save-and-log steps on one rank, the failed marker landing between
    the ranks' first logs, and report the step this rank unwound at."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=10)
    )
    base = Path(root)
    callback = CheckpointSpool(base / "spool", poll_sec=0.05)
    trainer = _make_trainer(target_kl=None, callbacks=[callback])
    args = TrainingArguments(output_dir=(base / "out").as_posix(), report_to=[])
    unwound: int | None = None
    try:
        with patch("trl.experimental.ppo.ppo_trainer.PPOTrainer.log", autospec=True):
            for step in (1, 2):
                state = TrainerState(global_step=step)
                state.is_world_process_zero = rank == 0
                callback.on_save(args, state, TrainerControl())
                for turn in (0, 1):
                    dist.barrier()
                    if step == 1 and rank == 0 and turn == 1:
                        fail_spool(base / "spool", "store unreachable")
                    dist.barrier()
                    if rank == turn and unwound is None:
                        try:
                            trainer.log({"objective/kl": 0.01})
                        except _CheckpointPublishStop:
                            unwound = step
        out.put((rank, unwound))
    finally:
        dist.destroy_process_group()


def test_every_rank_unwinds_at_the_same_save(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
    open_spool(spool)
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ranks = [
        context.Process(target=_unwinds, args=(rank, port, tmp_path.as_posix(), out))
        for rank in (0, 1)
    ]
    for process in ranks:
        process.start()
    results = dict(out.get(timeout=120) for _ in ranks)
    for process in ranks:
        process.join(30)

    assert results == {0: 2, 1: 2}
