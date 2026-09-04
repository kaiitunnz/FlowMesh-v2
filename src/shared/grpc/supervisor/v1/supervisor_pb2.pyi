from typing import ClassVar as _ClassVar
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import message as _message
from google.protobuf import struct_pb2 as _struct_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class RegisterRequest(_message.Message):
    __slots__ = ("meta",)
    META_FIELD_NUMBER: _ClassVar[int]
    meta: _struct_pb2.Struct

    def __init__(
        self, meta: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("worker_id", "incarnation")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INCARNATION_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    incarnation: int

    def __init__(
        self, worker_id: _Optional[str] = ..., incarnation: _Optional[int] = ...
    ) -> None: ...

class InterruptMessage(_message.Message):
    __slots__ = ("task_id", "reason")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    reason: str

    def __init__(
        self, task_id: _Optional[str] = ..., reason: _Optional[str] = ...
    ) -> None: ...

class StopMessage(_message.Message):
    __slots__ = ("task_id", "reason")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    reason: str

    def __init__(
        self, task_id: _Optional[str] = ..., reason: _Optional[str] = ...
    ) -> None: ...

class TaskMessage(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct

    def __init__(
        self, payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...

class ToolEgressFrame(_message.Message):
    __slots__ = ("session_id", "kind", "payload")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    kind: str
    payload: bytes

    def __init__(
        self,
        session_id: _Optional[str] = ...,
        kind: _Optional[str] = ...,
        payload: _Optional[bytes] = ...,
    ) -> None: ...

class DispatchMessage(_message.Message):
    __slots__ = ("task", "interrupt", "stop", "egress")
    TASK_FIELD_NUMBER: _ClassVar[int]
    INTERRUPT_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    EGRESS_FIELD_NUMBER: _ClassVar[int]
    task: TaskMessage
    interrupt: InterruptMessage
    stop: StopMessage
    egress: ToolEgressFrame

    def __init__(
        self,
        task: _Optional[_Union[TaskMessage, _Mapping]] = ...,
        interrupt: _Optional[_Union[InterruptMessage, _Mapping]] = ...,
        stop: _Optional[_Union[StopMessage, _Mapping]] = ...,
        egress: _Optional[_Union[ToolEgressFrame, _Mapping]] = ...,
    ) -> None: ...

class EventMessage(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct

    def __init__(
        self, payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...

class LogMessage(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: _struct_pb2.Struct

    def __init__(
        self, payload: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...
    ) -> None: ...
