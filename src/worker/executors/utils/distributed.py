"""Distributed launcher helpers used by the training executors.

Exposes :func:`run_torchrun`, which invokes ``torch.distributed.run.main``
directly — the same entry point the ``torchrun`` console script calls. Worker ranks
are spawned by torch's elastic agent and inherit ``CUDA_VISIBLE_DEVICES``; this
module just shapes the argv and scopes the launcher env.

:func:`deepspeed_available` reports whether the DeepSpeed package can be
imported in the current environment, as it is absent from the CPU worker image and
unusable without a CUDA toolchain.

:func:`launcher_task_file` writes the task the ranks read for the span of a launch.
"""

import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel
from torch.distributed.run import main as _torchrun_main

from shared.utils.manifest import scratch_dir

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
    return {"PYTHONPATH": pythonpath, launcher_env_flag: "1"}


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
