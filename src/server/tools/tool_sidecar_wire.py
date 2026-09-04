"""Socket/relay framing for the remote external-tool carriage.

The message-body codec (``encode_msg`` / ``decode_msg`` and the ``KIND_*`` values) is
the shared ``shared.tools.wire`` contract the origin and the worker executor both speak;
this module adds the server-side socket framing that carries one body over a
network-plane connection. A relay hop stays byte-transparent, so the frames are
end-to-end.
"""

import asyncio
from typing import Any

from shared.tools.wire import KIND_OPERATION as KIND_OPERATION
from shared.tools.wire import KIND_REJECT as KIND_REJECT
from shared.tools.wire import KIND_RESULT as KIND_RESULT
from shared.tools.wire import (
    decode_msg,
    encode_msg,
)

from ..network import wire as netwire

split_host_port = netwire.split_host_port


async def write_msg(writer: asyncio.StreamWriter, kind: str, **fields: Any) -> None:
    await netwire.write_frame(writer, encode_msg(kind, **fields))


async def read_msg(reader: asyncio.StreamReader) -> dict[str, Any]:
    return decode_msg(await netwire.read_frame(reader))
