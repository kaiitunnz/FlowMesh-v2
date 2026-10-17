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

# ``retirement_verdict(mode, protected, deleting, history, digest, verified)`` decides
# one identity for a write about to name it, reading retirement before protection,
# since protection stops no delete already issued and an identity protected only as a
# pending upload names no stored bytes. It returns ``R`` while the identity is being
# retired; in ``bind`` mode ``N:<revision>`` when it was retired at a revision other
# than ``verified``; in ``member`` mode ``H`` when it was retired before; else ``P``
# when it is protected and ``A`` otherwise. ``pending`` mode reads protection only.
RETIREMENT = """
local function retirement_verdict(mode, protected, deleting, history, digest, verified)
  if mode ~= 'pending' then
    if redis.call('HEXISTS', deleting, digest) == 1 then
      return 'R'
    end
    local revision = redis.call('HGET', history, digest)
    if revision and mode == 'bind' and revision ~= verified then
      return 'N:' .. revision
    end
    if revision and mode == 'member' then
      return 'H'
    end
  end
  if redis.call('SISMEMBER', protected, digest) == 1 then
    return 'P'
  end
  return 'A'
end
"""

__all__ = ["HOLDERS", "KINDS", "RETIREMENT"]
