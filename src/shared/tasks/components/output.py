from typing import ClassVar

from .._base import RetiredFieldsModel, StrictBaseModel, TemplateBaseModel


class OutputSpec(StrictBaseModel, RetiredFieldsModel):
    retired_fields: ClassVar[tuple[str, ...]] = ("destination",)
    artifacts: list[str] | None = None


class OutputSpecTemplate(TemplateBaseModel, RetiredFieldsModel):
    retired_fields: ClassVar[tuple[str, ...]] = ("destination",)
    artifacts: list[str] | None = None
