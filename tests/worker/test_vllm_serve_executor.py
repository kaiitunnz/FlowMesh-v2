"""Tests for VLLMServeExecutor."""

import collections
import io
import logging
import shutil
import socketserver
import stat
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs.serve import ServeSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import (
    make_serve_executor,
    make_worker_config,
    make_worker_hardware,
    make_worker_task_message,
)
from worker import hw
from worker.executors import vllm_serve_executor as mod
from worker.executors.base_executor import ExecutionError, TaskCancelledError
from worker.executors.vllm_serve_executor import (
    ServeResult,
    VLLMServeExecutor,
    _drain_to_log,
)
from worker.resident import LocalEngine

_SOCKET = Path("/nonexistent/engine.sock")

_LOOPBACK = {
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "VLLM_HOST_IP": "127.0.0.1",
}


class TestVLLMServeExecutorInit:
    def test_supported_task_types(self) -> None:
        assert TaskType.SERVE in VLLMServeExecutor.supported_task_types

    def test_only_serve_task_type(self) -> None:
        assert VLLMServeExecutor.supported_task_types == frozenset({TaskType.SERVE})

    def test_is_available_false_without_vllm(self) -> None:
        cfg = make_worker_config()
        with patch.dict("sys.modules", {"vllm": None}):
            result = VLLMServeExecutor.is_available(cfg)
        assert result is False

    def test_is_available_true_with_vllm(self) -> None:
        cfg = make_worker_config()
        fake_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            result = VLLMServeExecutor.is_available(cfg)
        assert result is True


class TestServeSpecStrict:
    def test_minimal_spec(self) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE)
        assert spec.model is None
        assert spec.model_name is None
        assert spec.ttlSeconds is None
        assert spec.readinessTimeoutSeconds is None
        assert spec.port is None

    def test_spec_with_all_fields(self) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="meta-llama/Llama-3-8B"),
                vllm={"tensor_parallel_size": 2},
            ),
            ttlSeconds=7200.0,
            readinessTimeoutSeconds=300.0,
            port=8001,
        )
        assert spec.model_name == "meta-llama/Llama-3-8B"
        assert spec.model is not None
        assert spec.model.vllm == {"tensor_parallel_size": 2}
        assert spec.ttlSeconds == 7200.0
        assert spec.readinessTimeoutSeconds == 300.0
        assert spec.port == 8001

    def test_parses_inference_style_model_block(self) -> None:
        """ServeSpecStrict accepts the same model block as InferenceSpecStrict."""
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(
                    type="huggingface",
                    identifier="Qwen/Qwen3-0.6B",
                    revision="main",
                    trust_remote_code=True,
                ),
                vllm={
                    "gpu_memory_utilization": 0.9,
                    "trust_remote_code": True,
                    "max_model_len": 4096,
                },
            ),
        )
        assert spec.model_name == "Qwen/Qwen3-0.6B"
        assert spec.model_revision == "main"
        assert spec.model_trust_remote_code is True
        assert spec.model is not None
        assert spec.model.vllm == {
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "max_model_len": 4096,
        }

    def test_readiness_timeout_accepted(self) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE, readinessTimeoutSeconds=900.0)
        assert spec.readinessTimeoutSeconds == 900.0

    def test_task_type_is_serve(self) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE)
        assert spec.taskType == TaskType.SERVE

    def test_accepts_both_gated_access_modes(self) -> None:
        for mode in ("proxy", "forward"):
            assert (
                ServeSpecStrict(taskType=TaskType.SERVE, accessMode=mode).accessMode
                == mode
            )

    def test_rejects_the_removed_direct_access_mode(self) -> None:
        # ``direct`` named a raw ungated listener, which is no longer a mode at all.
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, accessMode="direct")  # type: ignore[arg-type]

    def test_rejects_removed_api_key_field(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, apiKey="sk-x")  # type: ignore[call-arg]

    def test_ttl_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, ttlSeconds=0.0)
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, ttlSeconds=-1.0)

    def test_readiness_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, readinessTimeoutSeconds=0.0)

    def test_port_must_be_in_range(self) -> None:
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, port=0)
        with pytest.raises(Exception):
            ServeSpecStrict(taskType=TaskType.SERVE, port=65536)


class TestServeExecutorCmdBuilding:
    """Executor maps model.vllm + model_name + revision to vllm api_server flags."""

    def _make_executor(self) -> VLLMServeExecutor:
        return make_serve_executor()

    def _run_capture_cmd(self, spec: ServeSpecStrict, tmp_path: Path) -> list[str]:
        return self._run_capture(spec, tmp_path)[0]

    def _run_capture(
        self,
        spec: ServeSpecStrict,
        tmp_path: Path,
        devices: tuple[str, ...] | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        if devices is not None:
            ex.bind_devices(devices)
        captured: list[tuple[list[str], dict[str, str]]] = []

        def fake_popen(cmd: list[str], env: dict[str, str], **_: object) -> MagicMock:
            captured.append((list(cmd), env))
            m = MagicMock()
            m.stdout = io.StringIO("")
            m.poll.return_value = 0
            m.returncode = 0
            m.pid = 12345
            return m

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch.object(ex, "_poll_health"),
            patch.object(ex, "_wait_for_serve"),
            patch.object(ex, "emit_update"),
            patch.object(ex, "_terminate_process_group"),
        ):
            ex.run(task, tmp_path)

        return captured[0]

    def test_a_bound_launch_sees_only_its_devices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setattr(
            hw,
            "visible_gpus",
            lambda: (
                hw.VisibleGpu(ordinal=0, nvml_index=0, uuid="GPU-a", name="H100"),
                hw.VisibleGpu(ordinal=1, nvml_index=1, uuid="GPU-b", name="H100"),
            ),
        )
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        ex.bind_devices(("GPU-b",))
        envs: list[dict[str, str]] = []

        def fake_popen(cmd: list[str], env: dict[str, str], **_: object) -> MagicMock:
            envs.append(env)
            m = MagicMock()
            m.stdout = io.StringIO("")
            m.poll.return_value = 0
            m.returncode = 0
            m.pid = 12345
            return m

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch.object(ex, "_poll_health"),
            patch.object(ex, "_wait_for_serve"),
            patch.object(ex, "emit_update"),
            patch.object(ex, "_terminate_process_group"),
        ):
            ex.run(task, tmp_path)

        assert envs[0]["CUDA_VISIBLE_DEVICES"] == "1"
        assert envs[0]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_model_name_and_revision_in_cmd(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="Qwen/Qwen3-0.6B", revision="main"),
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-0.6B"
        assert "--revision" in cmd
        assert cmd[cmd.index("--revision") + 1] == "main"

    def test_the_executors_own_options_win_over_the_specs(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="Qwen/Qwen3-0.6B", revision="v2"),
                vllm={
                    "served_model_name": "alias",
                    "revision": "v1",
                    "model": "evil/model",
                },
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        last = {flag: cmd[i + 1] for i, flag in enumerate(cmd) if flag.startswith("--")}
        assert last["--served-model-name"] == "alias"
        assert last["--model"] == "Qwen/Qwen3-0.6B"
        assert "--uds" in last
        assert last["--revision"] == "v2"

    def test_an_unset_engine_option_renders_no_flag(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"max_model_len": None, "dtype": "bfloat16"},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--max-model-len" not in cmd and "None" not in cmd
        assert cmd[cmd.index("--dtype") + 1] == "bfloat16"

    def test_vllm_dict_keys_become_flags(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--tensor-parallel-size" in cmd
        assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
        assert "--gpu-memory-utilization" in cmd
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"

    def test_env_vars_reach_the_engine_environment_not_its_flags(
        self, tmp_path: Path
    ) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={
                    "env_vars": {"VLLM_USE_V1": "1"},
                    "limit_mm_per_prompt": {"image": 2},
                },
            ),
        )
        cmd, env = self._run_capture(spec, tmp_path)
        assert env["VLLM_USE_V1"] == "1"
        assert "--env-vars" not in cmd
        assert cmd[cmd.index("--limit-mm-per-prompt") + 1] == '{"image": 2}'

    def test_a_binding_applies_beside_unrelated_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            hw,
            "visible_gpus",
            lambda: (
                hw.VisibleGpu(ordinal=0, nvml_index=0, uuid="GPU-a", name="H100"),
                hw.VisibleGpu(ordinal=1, nvml_index=1, uuid="GPU-b", name="H100"),
            ),
        )
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"env_vars": {"VLLM_USE_V1": "1"}},
            ),
        )
        _, env = self._run_capture(spec, tmp_path, devices=("GPU-b",))
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert env["VLLM_USE_V1"] == "1"

    def test_a_spec_pinning_its_devices_keeps_its_own(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"env_vars": {"CUDA_VISIBLE_DEVICES": "3"}},
            ),
        )
        assert spec.pins_cuda_devices()
        _, env = self._run_capture(spec, tmp_path)
        assert env["CUDA_VISIBLE_DEVICES"] == "3"

    def test_the_engine_keeps_its_collective_traffic_on_loopback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        _, env = self._run_capture(spec, tmp_path)
        assert {k: env[k] for k in _LOOPBACK} == _LOOPBACK

    def test_a_spec_choosing_its_own_collective_interface_keeps_it(
        self, tmp_path: Path
    ) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"env_vars": {"GLOO_SOCKET_IFNAME": "ib0"}},
            ),
        )
        _, env = self._run_capture(spec, tmp_path)
        assert env["GLOO_SOCKET_IFNAME"] == "ib0"
        assert env["NCCL_SOCKET_IFNAME"] == "lo"

    def test_env_vars_that_are_not_strings_fail_the_task(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"), vllm={"env_vars": {"X": 1}}
            ),
        )
        with pytest.raises(ExecutionError, match="env_vars"):
            self._run_capture(spec, tmp_path)

    def test_trust_remote_code_from_vllm_dict(self, tmp_path: Path) -> None:
        """trust_remote_code: true in model.vllm renders as a bare flag."""
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m"),
                vllm={"trust_remote_code": True},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--trust-remote-code" in cmd
        assert cmd.count("--trust-remote-code") == 1

    def test_trust_remote_code_from_source_not_duplicated(self, tmp_path: Path) -> None:
        """trust_remote_code in source but not model.vllm still renders once."""
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(
                source=ModelSource(identifier="m", trust_remote_code=True),
                vllm={},
            ),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--trust-remote-code" in cmd
        assert cmd.count("--trust-remote-code") == 1

    def test_revision_omitted_when_not_set(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        cmd = self._run_capture_cmd(spec, tmp_path)
        assert "--revision" not in cmd

    def test_missing_model_identifier_raises(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(taskType=TaskType.SERVE)
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        with pytest.raises(ExecutionError, match="model.source.identifier"):
            ex.run(task, tmp_path)


class TestServeLoopbackEndpoint:
    """The engine binds loopback only and emits worker-private endpoint facts.

    The api key is generated internally and surfaced only as a "_"-prefixed field in the
    task update (stripped from public task metadata), never in the result; the emit
    exposes no raw routable host or public listener.
    """

    def _make_executor(self) -> VLLMServeExecutor:
        return make_serve_executor()

    def _run(
        self, spec: ServeSpecStrict, tmp_path: Path
    ) -> tuple[list[str], dict[str, object], ServeResult]:
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        captured: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

        def fake_popen(cmd: list[str], env: dict[str, str], **_: object) -> MagicMock:
            captured.append(list(cmd))
            self.envs.append(env)
            m = MagicMock()
            m.stdout = io.StringIO("")
            m.poll.return_value = 0
            m.returncode = 0
            m.pid = 12345
            return m

        engines = ex._local_engines()
        self.published: list[LocalEngine | None] = []
        self.socket_dirs: list[tuple[Path, int]] = []

        def emit_update(task_id: str, payload: dict[str, Any]) -> None:
            self.published.append(engines.lookup(task_id))
            directory = Path(payload["serve"]["_socket"]).parent
            self.socket_dirs.append((directory, stat.S_IMODE(directory.stat().st_mode)))

        emit = MagicMock(side_effect=emit_update)
        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch.object(ex, "_poll_health"),
            patch.object(ex, "_wait_for_serve"),
            patch.object(ex, "emit_update", emit),
            patch.object(ex, "_terminate_process_group"),
        ):
            result = ex.run(task, tmp_path)

        self.withdrawn = engines.lookup(task.task_id) is None
        serve = emit.call_args.args[1]["serve"]
        return captured[0], serve, result

    def test_generates_api_key_internally(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        cmd, serve, _ = self._run(spec, tmp_path)
        generated = self.envs[0]["VLLM_API_KEY"]
        assert len(generated) == 64
        assert "--api-key" not in cmd
        assert generated not in cmd
        assert generated not in serve.values()

    def test_the_sidecar_resolves_the_key_inside_the_worker(
        self, tmp_path: Path
    ) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        _, serve, _ = self._run(spec, tmp_path)
        assert self.published == [
            LocalEngine(str(serve["_socket"]), self.envs[0]["VLLM_API_KEY"])
        ]
        assert self.withdrawn

    def test_listens_only_on_a_worker_private_socket(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            port=8001,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        cmd, serve, _ = self._run(spec, tmp_path)
        assert cmd[cmd.index("--uds") + 1] == serve["_socket"]
        assert "--host" not in cmd and "--port" not in cmd
        assert serve["model"] == "m"
        # No listener or credential is ever publicly exposed.
        assert set(serve) == {"model", "interface", "_socket"}
        assert serve["interface"] == "chat"

    def test_each_engine_socket_lives_in_its_own_private_directory(
        self, tmp_path: Path
    ) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        self._run(spec, tmp_path)
        ((first, first_mode),) = self.socket_dirs
        self._run(spec, tmp_path)
        ((second, second_mode),) = self.socket_dirs
        assert first != second
        assert first.parent == second.parent == Path(tempfile.gettempdir())
        assert first_mode == second_mode == 0o700
        assert not first.exists() and not second.exists()

    def test_a_socket_path_past_the_unix_limit_fails_the_task_clearly(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / ("d" * 100)
        parent.mkdir()
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = make_serve_executor(engine_parent=parent)
        with (
            patch("subprocess.Popen") as popen,
            pytest.raises(ExecutionError, match="107-byte Unix socket limit"),
        ):
            ex.run(task, tmp_path / "out")
        popen.assert_not_called()
        assert list(parent.iterdir()) == []

    def test_the_socket_directory_is_removed_when_the_engine_fails(
        self, tmp_path: Path
    ) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        task = make_worker_task_message(spec=spec, task_type=TaskType.SERVE)
        ex = self._make_executor()
        sockets: list[Path] = []

        def fake_popen(cmd: list[str], **_: object) -> MagicMock:
            sockets.append(Path(cmd[cmd.index("--uds") + 1]))
            m = MagicMock()
            m.stdout = io.StringIO("")
            m.pid = 12345
            return m

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch.object(ex, "_poll_health", side_effect=ExecutionError("exited")),
            patch.object(ex, "_terminate_process_group"),
            pytest.raises(ExecutionError, match="exited"),
        ):
            ex.run(task, tmp_path)
        assert sockets and not sockets[0].parent.exists()

    def test_result_never_carries_api_key(self, tmp_path: Path) -> None:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
        )
        _, _, result = self._run(spec, tmp_path)
        assert "api_key" not in result.model_dump()
        assert result.port is None


class TestReadinessOverTheSocket:
    def test_the_health_poll_reaches_the_engine_over_its_socket(self) -> None:
        directory = Path(tempfile.mkdtemp())
        path = directory / "engine.sock"

        class _Health(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return None

        server = socketserver.ThreadingUnixStreamServer(path.as_posix(), _Health)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proc = MagicMock()
        proc.poll.return_value = None
        try:
            make_serve_executor()._poll_health(
                proc, path, "tsk-test", 5.0, collections.deque()
            )
        finally:
            server.shutdown()
            server.server_close()
            shutil.rmtree(directory)


class TestDefaultReadinessTimeout:
    def test_default_is_at_least_600s(self) -> None:
        assert mod._DEFAULT_READINESS_TIMEOUT_SEC >= 600.0


class TestVLLMServeExecutorCancelStop:
    def _make_executor(self) -> VLLMServeExecutor:
        cfg = make_worker_config()
        hw = make_worker_hardware()
        return make_serve_executor(cfg, hw)

    def test_cancel_signals_the_running_task(self) -> None:
        ex = self._make_executor()
        with ex._signals.running("tsk-test"):
            assert not ex._signals.cancelled
            ex.cancel("tsk-test")
            assert ex._signals.cancelled

    def test_stop_signals_the_running_task(self) -> None:
        ex = self._make_executor()
        with ex._signals.running("tsk-test"):
            assert not ex._signals.stopped
            ex.stop("tsk-test")
            assert ex._signals.stopped

    def test_a_signal_after_its_task_ended_does_not_reach_the_next(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        ex._proc = mock_proc
        with patch.object(ex, "_terminate_process_group") as mock_term:
            with ex._signals.running("tsk-done"):
                pass
            ex.cancel("tsk-done")
            ex.stop("tsk-done")
            mock_term.assert_not_called()
        with ex._signals.running("tsk-next"):
            assert not ex._signals.cancelled
            assert not ex._signals.stopped

    def test_cancel_terminates_proc(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        ex._proc = mock_proc
        with (
            patch.object(ex, "_terminate_process_group") as mock_term,
            ex._signals.running("tsk-test"),
        ):
            ex.cancel("tsk-test")
            mock_term.assert_called_once_with(mock_proc)

    def test_stop_terminates_proc(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        ex._proc = mock_proc
        with (
            patch.object(ex, "_terminate_process_group") as mock_term,
            ex._signals.running("tsk-test"),
        ):
            ex.stop("tsk-test")
            mock_term.assert_called_once_with(mock_proc)

    def test_cancel_no_proc_is_safe(self) -> None:
        ex = self._make_executor()
        ex._proc = None
        ex.cancel("tsk-test")  # must not raise


class TestWaitForServe:
    def _make_executor(self) -> VLLMServeExecutor:
        cfg = make_worker_config()
        return make_serve_executor(cfg)

    def test_exits_on_cancel(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with ex._signals.running("tsk-test"), pytest.raises(TaskCancelledError):
            ex.cancel("tsk-test")
            ex._wait_for_serve(mock_proc, deadline=time.time() + 60.0)

    def test_exits_on_stop(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with ex._signals.running("tsk-test"):
            ex.stop("tsk-test")
            ex._wait_for_serve(mock_proc, deadline=time.time() + 60.0)

    def test_raises_on_unexpected_proc_exit(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        with pytest.raises(ExecutionError):
            ex._wait_for_serve(mock_proc, deadline=time.time() + 60.0)

    def test_exits_when_ttl_expires(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        original = mod._POLL_INTERVAL_SEC
        mod._POLL_INTERVAL_SEC = 0.01
        try:
            start = time.time()
            ex._wait_for_serve(mock_proc, deadline=time.time() + 0.02)
            elapsed = time.time() - start
        finally:
            mod._POLL_INTERVAL_SEC = original
        assert elapsed < 2.0


class TestPollHealth:
    def _make_executor(self) -> VLLMServeExecutor:
        return make_serve_executor()

    def _empty_tail(self) -> "collections.deque[str]":
        return collections.deque(maxlen=200)

    def test_timeout_error_includes_subprocess_output(self) -> None:
        """Timeout error message contains the last lines captured from vLLM."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        tail: collections.deque[str] = collections.deque(
            ["Loading weights...", "CUDA graph capture OOM"], maxlen=200
        )

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch.object(
                httpx.Client, "get", side_effect=httpx.ConnectError("refused")
            ):
                with pytest.raises(ExecutionError) as exc_info:
                    ex._poll_health(
                        mock_proc, _SOCKET, "tsk-test", timeout_sec=0.01, tail=tail
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig

        msg = str(exc_info.value)
        assert "Loading weights..." in msg
        assert "CUDA graph capture OOM" in msg

    def test_early_exit_error_includes_subprocess_output(self) -> None:
        """Proc-died error message contains the last captured lines."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        tail: collections.deque[str] = collections.deque(
            ["CUDA error: device-side assert triggered"], maxlen=200
        )

        with patch.object(
            httpx.Client, "get", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(ExecutionError) as exc_info:
                ex._poll_health(
                    mock_proc, _SOCKET, "tsk-test", timeout_sec=30.0, tail=tail
                )

        assert "CUDA error: device-side assert triggered" in str(exc_info.value)

    def test_timeout_message_reflects_timeout_sec_argument(self) -> None:
        """Error message reports the actual timeout used, not a hardcoded value."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch.object(
                httpx.Client, "get", side_effect=httpx.ConnectError("refused")
            ):
                with pytest.raises(ExecutionError, match=r"within 3s"):
                    ex._poll_health(
                        mock_proc, _SOCKET, "tsk-x", timeout_sec=3.0, tail=tail
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig

    def test_returns_when_health_200(self) -> None:
        """Returns without raising when /health responds 200."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            ex._poll_health(mock_proc, _SOCKET, "tsk-ok", timeout_sec=30.0, tail=tail)

    def test_cancel_during_poll_raises(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        with patch.object(
            httpx.Client, "get", side_effect=httpx.ConnectError("refused")
        ):
            with ex._signals.running("tsk-cancel"), pytest.raises(TaskCancelledError):
                ex.cancel("tsk-cancel")
                ex._poll_health(
                    mock_proc, _SOCKET, "tsk-cancel", timeout_sec=60.0, tail=tail
                )

    def test_stop_during_poll_returns(self) -> None:
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        with patch.object(
            httpx.Client, "get", side_effect=httpx.ConnectError("refused")
        ):
            with ex._signals.running("tsk-stop"):
                ex.stop("tsk-stop")
                ex._poll_health(
                    mock_proc, _SOCKET, "tsk-stop", timeout_sec=60.0, tail=tail
                )

    def test_empty_tail_no_snippet_in_message(self) -> None:
        """When subprocess produced no output, the error omits the snippet block."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        tail = self._empty_tail()

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch.object(
                httpx.Client, "get", side_effect=httpx.ConnectError("refused")
            ):
                with pytest.raises(ExecutionError) as exc_info:
                    ex._poll_health(
                        mock_proc, _SOCKET, "tsk-empty", timeout_sec=0.01, tail=tail
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig

        assert "last vLLM output" not in str(exc_info.value)


class TestDrainToLog:
    def test_streams_lines_to_logger_and_tail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("Loading model...\nReady.\n")
        eof_event = threading.Event()

        with caplog.at_level(
            logging.INFO, logger="worker.executors.vllm_serve_executor"
        ):
            _drain_to_log(mock_proc, tail, eof_event)

        assert list(tail) == ["Loading model...", "Ready."]
        messages = [r.message for r in caplog.records]
        assert any("Loading model..." in m for m in messages)
        assert any("Ready." in m for m in messages)
        assert eof_event.is_set()

    def test_strips_trailing_newline(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("line with newline\n")

        _drain_to_log(mock_proc, tail, threading.Event())

        assert list(tail) == ["line with newline"]

    def test_respects_deque_maxlen(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=3)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("a\nb\nc\nd\ne\n")

        _drain_to_log(mock_proc, tail, threading.Event())

        assert list(tail) == ["c", "d", "e"]

    def test_empty_output_leaves_tail_empty(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("")

        _drain_to_log(mock_proc, tail, threading.Event())

        assert list(tail) == []

    def test_sets_eof_event_on_completion(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("some line\n")
        eof_event = threading.Event()

        _drain_to_log(mock_proc, tail, eof_event)

        assert eof_event.is_set()

    def test_sets_eof_event_on_empty_output(self) -> None:
        tail: collections.deque[str] = collections.deque(maxlen=200)
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("")
        eof_event = threading.Event()

        _drain_to_log(mock_proc, tail, eof_event)

        assert eof_event.is_set()


class TestPollHealthEofFastFail:
    """Regression: _poll_health must raise promptly on stdout-pipe EOF.

    vLLM is multiprocess. When the APIServer child crashes, the top-level
    Popen may stay alive (proc.poll() returns None), so the old "process
    exited" check missed it and the task hung for the full readiness
    timeout. The drain thread signals EOF via an event; _poll_health detects
    it and fails immediately instead.
    """

    def _make_executor(self) -> VLLMServeExecutor:
        return make_serve_executor()

    def _empty_tail(self) -> "collections.deque[str]":
        return collections.deque(maxlen=200)

    def test_fails_promptly_when_pipe_eof(self) -> None:
        """Raises ExecutionError well before timeout when eof_event is set."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # top-level process still "alive"
        mock_proc.returncode = 1

        tail: collections.deque[str] = collections.deque(
            ["AttributeError: 'NoneType' object has no attribute 'serve'"],
            maxlen=200,
        )
        eof_event = threading.Event()
        eof_event.set()  # simulate pipe EOF arriving before readiness timeout

        start = time.time()
        with patch.object(
            httpx.Client, "get", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(ExecutionError):
                ex._poll_health(
                    mock_proc,
                    _SOCKET,
                    "tsk-eof",
                    timeout_sec=600.0,
                    tail=tail,
                    eof_event=eof_event,
                )
        elapsed = time.time() - start

        assert elapsed < 5.0  # must not wait anywhere near 600s

    def test_error_includes_captured_output_on_eof(self) -> None:
        """ExecutionError raised on EOF carries the tail snippet."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = 1

        tail: collections.deque[str] = collections.deque(
            ["Starting APIServer...", "AttributeError: bad attribute"], maxlen=200
        )
        eof_event = threading.Event()
        eof_event.set()

        with patch.object(
            httpx.Client, "get", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(ExecutionError) as exc_info:
                ex._poll_health(
                    mock_proc,
                    _SOCKET,
                    "tsk-eof-output",
                    timeout_sec=600.0,
                    tail=tail,
                    eof_event=eof_event,
                )

        msg = str(exc_info.value)
        assert "Starting APIServer..." in msg
        assert "AttributeError: bad attribute" in msg

    def test_no_false_trigger_when_eof_not_set(self) -> None:
        """Health poll reaches normal timeout when eof_event is not set."""
        ex = self._make_executor()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        eof_event = threading.Event()  # NOT set

        orig = mod._HEALTH_POLL_INTERVAL_SEC
        mod._HEALTH_POLL_INTERVAL_SEC = 0.001
        try:
            with patch.object(
                httpx.Client, "get", side_effect=httpx.ConnectError("refused")
            ):
                with pytest.raises(ExecutionError, match=r"within 1s"):
                    ex._poll_health(
                        mock_proc,
                        _SOCKET,
                        "tsk-no-eof",
                        timeout_sec=1.0,
                        tail=self._empty_tail(),
                        eof_event=eof_event,
                    )
        finally:
            mod._HEALTH_POLL_INTERVAL_SEC = orig


class TestServeTtlAcrossReruns:
    def _run(
        self, tmp_path: Path, ttl: float, elapsed: float | None
    ) -> tuple[MagicMock, list[float]]:
        spec = ServeSpecStrict(
            taskType=TaskType.SERVE,
            model=ModelConfig(source=ModelSource(identifier="m")),
            ttlSeconds=ttl,
        )
        task = make_worker_task_message(
            spec=spec, task_type=TaskType.SERVE, serve_elapsed_sec=elapsed
        )
        ex = make_serve_executor()
        deadlines: list[float] = []
        proc = MagicMock()
        proc.stdout = io.StringIO("")
        with (
            patch("subprocess.Popen", return_value=proc) as popen,
            patch.object(ex, "_poll_health"),
            patch.object(
                ex, "_wait_for_serve", side_effect=lambda _p, d: deadlines.append(d)
            ),
            patch.object(ex, "emit_update"),
            patch.object(ex, "_terminate_process_group"),
        ):
            ex.run(task, tmp_path)
        return popen, deadlines

    def test_a_re_run_serves_what_remains_of_the_ttl(self, tmp_path: Path) -> None:
        before = time.time()
        _popen, deadlines = self._run(tmp_path, ttl=180.0, elapsed=100.0)
        assert before + 80.0 - 1.0 <= deadlines[0] <= time.time() + 80.0

    def test_an_elapsed_ttl_starts_no_engine(self, tmp_path: Path) -> None:
        popen, deadlines = self._run(tmp_path, ttl=180.0, elapsed=180.0)
        popen.assert_not_called()
        assert deadlines == []
