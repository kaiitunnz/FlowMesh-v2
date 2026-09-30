from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

# Validation context for a model loaded from durable storage, under which it drops the
# fields it retired.
PERSISTED_LOAD_CONTEXT: Mapping[str, bool] = MappingProxyType({"persisted_load": True})


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", from_attributes=True, populate_by_name=True
    )


class RetiredFieldsModel(BaseModel):
    """A model that loads a stored value carrying a field it retired.

    A submission naming a retired field is rejected like any unknown key; a value
    loaded under ``PERSISTED_LOAD_CONTEXT`` has the field dropped instead.
    """

    retired_fields: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_fields(cls, data: Any, info: ValidationInfo) -> Any:
        context = info.context
        if not (
            cls.retired_fields
            and isinstance(data, dict)
            and isinstance(context, Mapping)
            and context.get("persisted_load")
        ):
            return data
        return {k: v for k, v in data.items() if k not in cls.retired_fields}


class TemplateBaseModel(BaseModel):
    """
    Template-time base model.

    Still forbids unknown keys, but allows placeholder-friendly scalar types
    (e.g. int | "${...}") in *explicit* fields of concrete models.
    """

    model_config = ConfigDict(
        extra="forbid", from_attributes=True, populate_by_name=True
    )
