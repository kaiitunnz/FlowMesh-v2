"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 6, 33, 5, "", "supervisor/v1/supervisor.proto"
)
_sym_db = _symbol_database.Default()
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1esupervisor/v1/supervisor.proto\x12\rsupervisor.v1\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1bgoogle/protobuf/empty.proto"8\n\x0fRegisterRequest\x12%\n\x04meta\x18\x01 \x01(\x0b2\x17.google.protobuf.Struct":\n\x10RegisterResponse\x12\x11\n\tworker_id\x18\x01 \x01(\t\x12\x13\n\x0bincarnation\x18\x02 \x01(\x03"H\n\x10InterruptMessage\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x0e\n\x06reason\x18\x02 \x01(\t\x12\x13\n\x0bdispatch_id\x18\x03 \x01(\t"C\n\x0bStopMessage\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x0e\n\x06reason\x18\x02 \x01(\t\x12\x13\n\x0bdispatch_id\x18\x03 \x01(\t"5\n\rRevokeMessage\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x13\n\x0bdispatch_id\x18\x02 \x01(\t"7\n\x0bTaskMessage\x12(\n\x07payload\x18\x01 \x01(\x0b2\x17.google.protobuf.Struct"P\n\x16MediatedOperationFrame\x12\x0c\n\x04kind\x18\x01 \x01(\t\x12(\n\x07payload\x18\x02 \x01(\x0b2\x17.google.protobuf.Struct"\x98\x02\n\x0fDispatchMessage\x12*\n\x04task\x18\x01 \x01(\x0b2\x1a.supervisor.v1.TaskMessageH\x00\x124\n\tinterrupt\x18\x02 \x01(\x0b2\x1f.supervisor.v1.InterruptMessageH\x00\x12*\n\x04stop\x18\x03 \x01(\x0b2\x1a.supervisor.v1.StopMessageH\x00\x12<\n\x0bmediated_op\x18\x05 \x01(\x0b2%.supervisor.v1.MediatedOperationFrameH\x00\x12.\n\x06revoke\x18\x06 \x01(\x0b2\x1c.supervisor.v1.RevokeMessageH\x00B\t\n\x07payload"8\n\x0cEventMessage\x12(\n\x07payload\x18\x01 \x01(\x0b2\x17.google.protobuf.Struct"6\n\nLogMessage\x12(\n\x07payload\x18\x01 \x01(\x0b2\x17.google.protobuf.Struct2\xae\x02\n\nSupervisor\x12Q\n\x0eRegisterWorker\x12\x1e.supervisor.v1.RegisterRequest\x1a\x1f.supervisor.v1.RegisterResponse\x12G\n\x0bStreamTasks\x12\x16.google.protobuf.Empty\x1a\x1e.supervisor.v1.DispatchMessage0\x01\x12C\n\nPushEvents\x12\x1b.supervisor.v1.EventMessage\x1a\x16.google.protobuf.Empty(\x01\x12?\n\x08PushLogs\x12\x19.supervisor.v1.LogMessage\x1a\x16.google.protobuf.Empty(\x01b\x06proto3'
)
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "supervisor.v1.supervisor_pb2", _globals
)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_REGISTERREQUEST"]._serialized_start = 108
    _globals["_REGISTERREQUEST"]._serialized_end = 164
    _globals["_REGISTERRESPONSE"]._serialized_start = 166
    _globals["_REGISTERRESPONSE"]._serialized_end = 224
    _globals["_INTERRUPTMESSAGE"]._serialized_start = 226
    _globals["_INTERRUPTMESSAGE"]._serialized_end = 298
    _globals["_STOPMESSAGE"]._serialized_start = 300
    _globals["_STOPMESSAGE"]._serialized_end = 367
    _globals["_REVOKEMESSAGE"]._serialized_start = 369
    _globals["_REVOKEMESSAGE"]._serialized_end = 422
    _globals["_TASKMESSAGE"]._serialized_start = 424
    _globals["_TASKMESSAGE"]._serialized_end = 479
    _globals["_MEDIATEDOPERATIONFRAME"]._serialized_start = 481
    _globals["_MEDIATEDOPERATIONFRAME"]._serialized_end = 561
    _globals["_DISPATCHMESSAGE"]._serialized_start = 564
    _globals["_DISPATCHMESSAGE"]._serialized_end = 844
    _globals["_EVENTMESSAGE"]._serialized_start = 846
    _globals["_EVENTMESSAGE"]._serialized_end = 902
    _globals["_LOGMESSAGE"]._serialized_start = 904
    _globals["_LOGMESSAGE"]._serialized_end = 958
    _globals["_SUPERVISOR"]._serialized_start = 961
    _globals["_SUPERVISOR"]._serialized_end = 1263
