#!/usr/bin/env python3
"""A stand-in for ``router-acp`` that speaks just enough ACP for tests.

``serve --config <path>`` reads newline-delimited JSON-RPC on stdin. Each
prompt is answered with a routing notice and ``turn N: <prompt>``, where N
counts the prompts this process has seen. A prompt containing HANG is never
answered, EMPTY returns only the notice, DIE exits, and QUOTA fails with a
usage-cap error. ``--version`` prints a version line.
"""

import json
import os
import sys

MODEL = os.environ.get("STUB_ROUTER_MODEL", "grok@" + "ab" * 16 + "/grok-4.7")


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def update(session_id, text):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        }
    )


def main():
    if "--version" in sys.argv:
        print("router-acp 0.0.0-stub")
        return 0
    config = sys.argv[sys.argv.index("--config") + 1]
    log = os.environ.get("STUB_ROUTER_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            started = {"config": open(config, encoding="utf-8").read(), "env": dict(os.environ)}
            handle.write(json.dumps(started) + "\n")
    turns = 0
    pinned = ""
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": 1}})
        elif method == "session/new":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": "s-1"}})
        elif method == "session/set_config_option":
            pinned = params.get("value", "")
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif method == "session/prompt":
            turns += 1
            prompt = params["prompt"][0]["text"]
            if log:
                with open(log, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"prompt": prompt, "pinned": pinned}) + "\n")
            if "HANG" in prompt:
                continue
            if "DIE" in prompt:
                return 1
            if "QUOTA" in prompt:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": "usage cap reached"},
                    }
                )
                continue
            if turns == 1:
                update("s-1", f"router-acp · auto → {MODEL}\n> why: first match\n")
            if "EMPTY" not in prompt:
                update("s-1", f"turn {turns}: {prompt}")
            send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "end_turn"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
