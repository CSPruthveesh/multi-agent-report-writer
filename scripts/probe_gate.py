"""Drive the approval gate through the real SSE endpoints, for nothing.

    uv run python -m scripts.probe_gate

Every request to /api/run spends 30-50k tokens, so the endpoint had no cheap mode and
checking a status code once cost a full report. FREE_OVERRIDES swaps every
model-calling node for an offline double — the same topology, no API calls — and
patching it over build() exercises the streaming path end to end.

WHAT IT CHECKS, AND WHY EACH ONE BIT

  the interrupt fires        SqliteSaver is sync-only. astream against one raises
                             NotImplementedError, which is how the API learned it
                             needed the async saver.

  the parked payload reads   _pending_interrupt calls aget_state, not get_state. The
                             async saver refuses sync calls from the event loop, so the
                             graph ran all the way to the gate and then fell over
                             reading its own parked state.

  no rows are replayed       A resumed stream sees a trace holding the whole run. The
                             client sends back how many events it has, and without that
                             the second leg redraws the first under it.

  the gate can fire twice    Approval sits before the Writer, so a revision loop
                             re-enters it. Found here rather than in front of someone.
"""
import asyncio
import io
import json
import sys

# line_buffering: the wrapper replaces stdout, so python -u no longer applies to it —
# without this the whole run sits in a buffer and a probe that hangs prints nothing.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import api.main as M
from src.graph.build import FREE_OVERRIDES
from src.graph.build import build as real_build

# Same topology, offline nodes.
M.build = lambda **kw: real_build(overrides=FREE_OVERRIDES, **kw)


def frames(text):
    for f in text.split("\n\n"):
        if not f.strip():
            continue
        name = f.split("event: ", 1)[1].split("\n", 1)[0]
        data = json.loads(f.split("data: ", 1)[1])
        yield name, data


async def collect(agen):
    return "".join([f async for f in agen])


async def main():
    print("=== POST /api/run  {hitl: true} ===")
    out = await collect(M._stream("a probe topic for the approval gate", False, True))
    thread = seen = None
    for name, d in frames(out):
        if name == "start":
            print(f"  start     hitl={d['hitl']} thread={d['thread_id'][:8]}…")
        elif name == "node":
            print(f"  node      {d['event']['node']:11}{d['event']['action']}")
        elif name == "approval":
            thread, seen = d["thread_id"], d["seen"]
            i = d["interrupt"]
            print(f"  APPROVAL  kind={i['kind']} options={i['options']}")
            print(f"            findings={i['finding_count']} seen={seen}")
            print(f"            outline starts: {i['outline'][:48]!r}")
        else:
            print(f"  {name}: {d}")

    print("\n=== POST /api/resume  {action: approve} ===")
    req = M.ResumeRequest(thread_id=thread, decision={"action": "approve"}, seen=seen)
    out2 = await collect(M._resume(req))
    gates = 0
    for name, d in frames(out2):
        if name == "node":
            print(f"  node      {d['event']['node']:11}{d['event']['action']}")
        elif name == "approval":
            gates += 1
            print("  APPROVAL  parked again — the revision loop re-entered the gate")
        elif name == "done":
            print(f"  done      {len(d['report'].split())} words, "
                  f"{len(d['findings'])} findings, tokens {d['cost']['tokens']}")
        else:
            print(f"  {name}: {str(d)[:70]}")

    print("\n=== no duplicate rows across the two legs ===")
    a = [d["event"]["action"] for n, d in frames(out) if n == "node"]
    b = [d["event"]["action"] for n, d in frames(out2) if n == "node"]
    print(f"  leg 1 sent {len(a)} events, leg 2 sent {len(b)} — none repeated: "
          f"{b[:len(a)] != a}")
    print(f"  second gate encountered: {gates}")


if __name__ == "__main__":
    asyncio.run(main())
