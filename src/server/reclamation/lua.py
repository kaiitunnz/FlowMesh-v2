"""Lua fragments the reclamation scripts share, prepended to a script's own body.

Each fragment defines local functions only, so a script that includes one runs no
command until its own body calls it.
"""

# ``wrong_kind(kinds)`` returns the first key, in KEYS order, whose type is neither
# absent nor one of the space-separated names ``kinds`` gives for its index; keys
# ``kinds`` leaves out are not checked. ``span(first, last, allowed)`` gives the same
# names to KEYS[first..last].
KINDS = """
local function span(first, last, allowed)
  local kinds = {}
  for i = first, last do
    kinds[i] = allowed
  end
  return kinds
end
local function wrong_kind(kinds)
  for i = 1, #KEYS do
    local allowed = kinds[i]
    if allowed then
      local found = redis.call('TYPE', KEYS[i]).ok
      if found ~= 'none'
          and not string.find(' ' .. allowed .. ' ', ' ' .. found .. ' ', 1, true) then
        return KEYS[i]
      end
    end
  end
  return nil
end
"""

# A holder list is a pins-hash field: the JSON list of ``{n = nonce, l = locator}``
# holds on one digest. ``add_holder`` appends one; ``drop_holder`` removes every hold
# under a nonce, deleting the field once none is left, and returns how many it removed.
HOLDERS = """
local function add_holder(key, digest, nonce, locator)
  local current = redis.call('HGET', key, digest)
  local holders = current and cjson.decode(current) or {}
  table.insert(holders, {n = nonce, l = locator})
  redis.call('HSET', key, digest, cjson.encode(holders))
end
local function drop_holder(key, digest, nonce)
  local current = redis.call('HGET', key, digest)
  if not current then
    return 0
  end
  local kept, dropped = {}, 0
  for _, holder in ipairs(cjson.decode(current)) do
    if holder.n == nonce then
      dropped = dropped + 1
    else
      table.insert(kept, holder)
    end
  end
  if dropped == 0 then
    return 0
  end
  if #kept == 0 then
    redis.call('HDEL', key, digest)
  else
    redis.call('HSET', key, digest, cjson.encode(kept))
  end
  return dropped
end
"""

__all__ = ["HOLDERS", "KINDS"]
