"""Every Strict task spec declares the credential fields its Template twin does."""

import typing
from typing import Any

import pytest

from shared.tasks.envelope import TaskSpecStrict, TaskSpecTemplate
from shared.tasks.specs import TaskSpecStrictBase, TaskSpecTemplateBase


def _by_task_type[T](union: typing.TypeAliasType, base: type[T]) -> dict[Any, type[T]]:
    members = typing.get_args(typing.get_args(union.__value__)[0])
    found: dict[Any, type[T]] = {}
    for member in members:
        assert issubclass(member, base)
        found[typing.get_args(member.model_fields["taskType"].annotation)] = member
    return found


_STRICT = _by_task_type(TaskSpecStrict, TaskSpecStrictBase)
_TEMPLATE = _by_task_type(TaskSpecTemplate, TaskSpecTemplateBase)


def test_every_task_type_has_a_strict_and_a_template_spec() -> None:
    assert _STRICT.keys() == _TEMPLATE.keys()


@pytest.mark.parametrize("task_type", list(_TEMPLATE), ids=str)
def test_the_twins_declare_the_same_credential_fields(task_type) -> None:
    strict, template = _STRICT[task_type], _TEMPLATE[task_type]

    assert strict.credential_fields == template.credential_fields
