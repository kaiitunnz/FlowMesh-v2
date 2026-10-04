"""Distributed launcher helpers used by the training executors.

Exposes :func:`run_torchrun`, which invokes ``torch.distributed.run.main``
directly — the same entry point the ``torchrun`` console script calls. Worker ranks
are spawned by torch's elastic agent and inherit ``CUDA_VISIBLE_DEVICES``; this
module just shapes the argv and scopes the launcher env.

:func:`deepspeed_available` reports whether the DeepSpeed package can be
imported in the current environment, as it is absent from the CPU worker image and
unusable without a CUDA toolchain.

:func:`launcher_task_file` writes the task the ranks read for the span of a launch.

:func:`launch_ranks` runs a launch and returns rank 0's result or the first rank
failure, which each rank entry hands back through :func:`run_rank`.
"""

import importlib.util
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.run import main as _torchrun_main

from shared.utils.manifest import scratch_dir

from ..base_executor import ExecutionError
from .collective import loopback_collective_env

logger = logging.getLogger(__name__)

_RESULT_FILE = "distributed_result.json"
_FAILURE_FILE = "distributed_failure.json"

# .../src/worker/executors/utils/distributed.py → parents[3] = .../src
_SRC_DIR = Path(__file__).resolve().parents[3]


@contextmanager
def launcher_task_file(out_dir: Path, task: BaseModel) -> Iterator[Path]:
    """The task a distributed launch's ranks read, for the span of the launch.

    The task carries its credentials, so the file is readable by the worker alone and
    is removed once the launch exits.
    """
    launcher_dir = scratch_dir(out_dir) / "launcher"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    path = launcher_dir / "task_spec.json"
    # A file a crashed launch left behind is replaced; anything planted at the path in
    # its place, such as a symlink, is refused rather than written through.
    if not path.is_symlink():
        path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(task.model_dump_json(by_alias=True))
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def _scoped_env(updates: dict[str, str]) -> Iterator[None]:
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _launch_env(launcher_env_flag: str) -> dict[str, str]:
    pythonpath = _SRC_DIR.as_posix()
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    return {
        **loopback_collective_env(),
        "PYTHONPATH": pythonpath,
        launcher_env_flag: "1",
    }


def run_torchrun(
    *,
    nproc_per_node: int,
    module: str,
    module_args: list[str],
    launcher_env_flag: str,
) -> None:
    """Run ``torchrun --nproc_per_node N -m <module> <args>`` in-process.

    The launcher prepends the source root to ``PYTHONPATH`` so spawned ranks
    can import ``worker.executors.*``, and sets ``launcher_env_flag`` to
    ``"1"`` so the entry module can detect it is running inside the launched
    ranks and skip a second spawn. Both env mutations are scoped to the
    launch call — the caller's environment is restored on return and on
    exception.

    ``--tee 3`` (stdout+stderr bitmask) is passed so the rank streams reach
    the parent's console and per-rank log files are written under the
    elastic agent's log dir; without it the elastic agent swallows rank
    output and a rank-side crash surfaces as an opaque ``ChildFailedError``
    with ``error_file: <N/A>``.
    """
    with _scoped_env(_launch_env(launcher_env_flag)):
        _torchrun_main(
            [
                "--nproc_per_node",
                str(nproc_per_node),
                "--tee",
                "3",
                "-m",
                module,
                *module_args,
            ]
        )


def deepspeed_available() -> bool:
    """Return whether ``deepspeed`` is importable here.

    DeepSpeed is a ``training-gpu`` extra and is absent from the CPU worker image. Any
    exception raised while resolving the spec is treated as "not available";
    DeepSpeed's package init eagerly probes CUDA op builders and raises
    ``MissingCUDAException`` on a CUDA-less host, which is indistinguishable
    from "not usable here".
    """
    try:
        return importlib.util.find_spec("deepspeed") is not None
    except Exception:
        return False


def launch_ranks[ResultT: BaseModel](
    *,
    nproc_per_node: int,
    module: str,
    out_dir: Path,
    task: BaseModel,
    launcher_env_flag: str,
    result_type: type[ResultT],
) -> ResultT:
    """Run ``module`` as torchrun ranks over ``task`` and return rank 0's result.

    A launch that fails raises the first rank failure with its own message and
    retryability, and a launch whose rank 0 handed back no result fails. A rank that
    died before recording a failure, as one the OOM killer or a watchdog signal ends,
    fails the launch retryably, unless rank 0 already handed back its result: rank 0
    finishes only after every rank has left the collectives, so a rank dying after
    that, as in a teardown abort, ends a finished training.
    """
    ipc = scratch_dir(out_dir)
    ipc.mkdir(parents=True, exist_ok=True)
    result_path, failure_path = ipc / _RESULT_FILE, ipc / _FAILURE_FILE
    result_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    try:
        with launcher_task_file(out_dir, task) as task_file:
            run_torchrun(
                nproc_per_node=nproc_per_node,
                module=module,
                module_args=[task_file.as_posix(), out_dir.as_posix()],
                launcher_env_flag=launcher_env_flag,
            )
    except BaseException as exc:
        if (failure := _read_rank_failure(failure_path)) is not None:
            raise failure from exc
        if isinstance(exc, ChildFailedError) and result_path.exists():
            logger.warning(
                "A rank of %s failed after rank 0 handed back its result: %s",
                module,
                exc,
            )
            return result_type.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        logger.exception("Distributed launch of %s failed", module)
        raise ExecutionError(
            f"distributed training failed: {exc}",
            retryable=isinstance(exc, ChildFailedError),
        ) from exc
    if not result_path.exists():
        raise ExecutionError("distributed training returned no result from rank 0")
    return result_type.model_validate_json(result_path.read_text(encoding="utf-8"))


def run_rank(out_dir: Path, run: Callable[[], BaseModel]) -> None:
    """Run one torchrun rank and hand its outcome back to the launching executor.

    Rank 0 hands back its result; a failing rank records its failure unless another
    rank already recorded one, so the launch reports the first.
    """
    ipc = scratch_dir(out_dir)
    try:
        result = run()
    except BaseException as exc:
        _publish_once(
            ipc / _FAILURE_FILE,
            json.dumps(
                {
                    "message": (
                        str(exc) or type(exc).__name__
                        if isinstance(exc, Exception)
                        else f"{type(exc).__name__}: {exc}"
                    ),
                    "retryable": isinstance(exc, ExecutionError) and exc.retryable,
                }
            ),
        )
        raise
    if os.environ.get("RANK", "0") == "0":
        _publish(
            ipc / _RESULT_FILE,
            (
                result.model_copy(update={"spawned_torchrun": True}).model_dump_json(
                    indent=2
                )
                if "spawned_torchrun" in type(result).model_fields
                else result.model_dump_json(indent=2)
            ),
        )


def _publish(path: Path, text: str) -> None:
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(staged, path)


def _publish_once(path: Path, text: str) -> None:
    # The launcher reads the record only after every rank exits, so an exclusive
    # create is enough to keep the first writer's record whole.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def _read_rank_failure(path: Path) -> ExecutionError | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("message"), str):
        return None
    return ExecutionError(record["message"], retryable=record.get("retryable") is True)
