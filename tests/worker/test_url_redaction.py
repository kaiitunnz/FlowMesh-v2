"""Worker errors, logs, and result fields quote a spec URL without its credentials."""

import io
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import requests

from shared.schemas.result.catalog import InferenceResult
from shared.tasks import TaskType
from shared.tasks.specs import RagSpecStrict
from tests.worker.factories import make_worker_config, make_worker_task_message
from worker.connectors import get_connector_from_spec
from worker.connectors.base_connector import ConnectorError
from worker.connectors.postgresql_connector import PostgreSQLConnector
from worker.executors import rag_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.mixins.data import DataMixin
from worker.executors.utils import artifacts, checkpoints, data_utils
from worker.utils.redaction import redact_urls

_TOKEN = "tok-inline-SECRET"
_URL = f"https://store.example/ckpt.tar?token={_TOKEN}&part=1"


def _forbidden(url: str, *args: Any, **kwargs: Any) -> requests.Response:
    response = requests.Response()
    response.status_code = 403
    response.url = url
    response.raw = io.BytesIO(b"")
    return response


def test_redact_urls_masks_a_known_url_whole_and_by_its_path():
    text = f"GET {_URL} failed; retried /ckpt.tar?token={_TOKEN}&part=1"
    assert _TOKEN not in redact_urls(text, _URL)
    assert "store.example" in redact_urls(text, _URL)


def test_a_forbidden_checkpoint_download_quotes_no_token(tmp_path: Path, caplog):
    with mock.patch.object(checkpoints.requests, "get", _forbidden):
        with caplog.at_level(logging.INFO):
            with pytest.raises(ExecutionError) as raised:
                checkpoints.resolve_checkpoint_load(
                    {"type": "http", "url": _URL, "_logger": logging.getLogger("t")},
                    tmp_path,
                )
    assert "403" in str(raised.value)
    assert _TOKEN not in str(raised.value)
    assert _TOKEN not in caplog.text


def test_a_forbidden_dataset_download_quotes_no_token(tmp_path: Path):
    with mock.patch.object(data_utils.requests, "get", _forbidden):
        with pytest.raises(ExecutionError) as raised:
            data_utils.resolve_jsonl_path(_URL, out_dir=tmp_path)
    assert _TOKEN not in str(raised.value)


def test_a_forbidden_artifact_download_quotes_no_token():
    with mock.patch.object(artifacts.requests, "get", _forbidden):
        with pytest.raises(ExecutionError) as raised:
            artifacts.resolve_artifact(_URL)
    assert _TOKEN not in str(raised.value)


def test_a_forbidden_image_fetch_quotes_no_token():
    with mock.patch("worker.executors.mixins.data.requests.get", _forbidden):
        with pytest.raises(ExecutionError) as raised:
            DataMixin._load_image_from_external_url(_URL)
    assert _TOKEN not in str(raised.value)


def test_an_upload_destination_is_recorded_without_its_userinfo(tmp_path: Path):
    spec = mock.MagicMock()
    spec.output.destination = mock.MagicMock(
        type="http", url=f"https://u:{_TOKEN}@sink.example/results"
    )
    context = checkpoints.build_artifact_context(spec, tmp_path)
    assert context.base_url == "https://sink.example"


_QDRANT = f"https://reader:{_TOKEN}@qdrant.example:6333"


def _rag_task() -> Any:
    return make_worker_task_message(
        task_type=TaskType.RAG,
        spec=RagSpecStrict(
            taskType=TaskType.RAG,
            qdrant={"url": _QDRANT, "collection": "docs"},
            query="hello",
        ),
    )


def test_a_rag_leaf_records_and_logs_its_qdrant_url_without_userinfo(
    tmp_path: Path, caplog
):
    client = mock.MagicMock()
    client.query_points.return_value = mock.MagicMock(points=[])
    with mock.patch.object(rag_executor, "QdrantClient", return_value=client):
        with caplog.at_level(logging.INFO):
            result = rag_executor.RAGExecutor(make_worker_config()).run(
                _rag_task(), tmp_path
            )
    assert result.qdrant is not None
    assert result.qdrant.url == "https://qdrant.example:6333"
    assert _TOKEN not in caplog.text


def test_a_failed_rag_query_quotes_no_qdrant_credential(tmp_path: Path, caplog, capsys):
    client = mock.MagicMock()
    client.query_points.side_effect = RuntimeError(f"cannot reach {_QDRANT}")
    with mock.patch.object(rag_executor, "QdrantClient", return_value=client):
        with pytest.raises(ExecutionError) as raised:
            rag_executor.RAGExecutor(make_worker_config()).run(_rag_task(), tmp_path)
    assert _TOKEN not in str(raised.value)
    assert _TOKEN not in caplog.text
    assert _TOKEN not in capsys.readouterr().out


def test_connector_errors_quote_no_connection_credential():
    with pytest.raises(ConnectorError) as unsupported:
        get_connector_from_spec(f"mysql://root:{_TOKEN}@db.example/app")
    assert _TOKEN not in str(unsupported.value)

    dsn = f"postgresql://app:{_TOKEN}@db.example:5432/app"
    with mock.patch(
        "worker.connectors.postgresql_connector.psycopg.connect",
        side_effect=RuntimeError(f"invalid connection string {dsn!r}"),
    ):
        with pytest.raises(ConnectorError) as refused:
            PostgreSQLConnector(connection_string=dsn).connect()
    assert _TOKEN not in str(refused.value)

    with pytest.raises(ConnectorError) as unparsable:
        get_connector_from_spec(f"s3://key:{_TOKEN}@store.example:notaport/bucket")
    assert _TOKEN not in str(unparsable.value)


def test_a_result_records_its_model_identifier_without_a_credential(tmp_path: Path):
    spec = make_worker_task_message({"taskType": "echo"}).spec
    result = InferenceResult(model=_URL)

    envelope = checkpoints.write_executor_result(
        tmp_path / "results.json", "tsk-1", spec, result
    )

    assert _TOKEN not in envelope
    assert "store.example/ckpt.tar" in envelope
