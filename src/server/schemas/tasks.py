from pydantic import BaseModel, Field

from server.task.models import TaskInfo


class TaskPage(BaseModel):
    entries: list[TaskInfo] = Field(description="Tasks, oldest submission first.")
    next_cursor: str | None = Field(
        default=None, description="Pass as `after` for the page of newer tasks."
    )
    prev_cursor: str | None = Field(
        default=None, description="Pass as `before` for the page of older tasks."
    )
