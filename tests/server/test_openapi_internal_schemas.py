from typing import Any

from fastapi import FastAPI

from server.routers import v1
from server.utils.openapi import FlowMeshAPI, drop_internal_schemas

_ROUTERS = (
    v1.workflows,
    v1.workers,
    v1.nodes,
    v1.tasks,
    v1.results,
    v1.outputs,
    v1.content,
    v1.ssh,
    v1.serve,
    v1.resident,
    v1.network,
    v1.system,
    v1.traces,
    v1.stack,
)
_INTERNAL = {
    "TaskInternalState",
    "TaskRecordRuntimeState",
    "ArtifactUpload",
    "EarlyUploadReceipt",
}


def _document(app: FastAPI) -> dict[str, Any]:
    for module in _ROUTERS:
        app.include_router(module.router, prefix="/api/v1")
    return app.openapi()


def test_the_api_document_names_no_internal_model() -> None:
    plain = _document(FastAPI())
    served = _document(FlowMeshAPI())

    schemas = served["components"]["schemas"]
    assert _INTERNAL <= plain["components"]["schemas"].keys()
    assert not _INTERNAL & schemas.keys()
    assert not any(schema.get("x-internal") for schema in schemas.values())
    assert served["paths"] == plain["paths"]
    assert schemas["TaskInfo"] == plain["components"]["schemas"]["TaskInfo"]


def test_a_schema_an_operation_reaches_stays() -> None:
    ref = "#/components/schemas/"
    document: dict[str, Any] = {
        "paths": {"/a": {"get": {"schema": {"$ref": f"{ref}Shared"}}}},
        "components": {
            "schemas": {
                "Hidden": {
                    "x-internal": True,
                    "properties": {
                        "a": {"$ref": f"{ref}Shared"},
                        "b": {"$ref": f"{ref}Only"},
                    },
                },
                "Only": {"properties": {"c": {"$ref": f"{ref}Deeper"}}},
                "Deeper": {},
                "Shared": {},
                "Orphan": {},
            }
        },
    }

    drop_internal_schemas(document)

    assert document["components"]["schemas"].keys() == {"Shared", "Orphan"}
