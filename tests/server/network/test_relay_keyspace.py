"""The resident and external-tool relay keyspaces are disjoint on one node.

Guards the same-node collision the keyspace parameterization fixes: a node running both
a resident (``rr:*``) and a tool (``xt:*``) attachment must share no stream, session
record, lease, or down-cursor key.
"""

from server.network.reverse_relay import (
    RESIDENT_RELAY_KEYSPACE,
    TOOL_RELAY_KEYSPACE,
)

NODE = "nde-1"
SESSION = "s-1"


def test_keyspaces_share_no_key() -> None:
    rr = RESIDENT_RELAY_KEYSPACE
    xt = TOOL_RELAY_KEYSPACE
    resident_keys = {
        rr.up(NODE),
        rr.down(NODE),
        rr.session(SESSION),
        rr.down_cursor(NODE),
        rr.root_cursor,
        f"{rr.session(NODE)}:lease:down",
    }
    tool_keys = {
        xt.up(NODE),
        xt.down(NODE),
        xt.session(SESSION),
        xt.down_cursor(NODE),
        xt.root_cursor,
        f"{xt.session(NODE)}:lease:down",
    }
    assert resident_keys.isdisjoint(tool_keys)
    assert all(k.startswith("rr:") for k in resident_keys)
    assert all(k.startswith("xt:") for k in tool_keys)
