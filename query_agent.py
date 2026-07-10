#!/usr/bin/env python3
"""Call the running ChemCrow agent server.

Usage:
    # Plain query (trace collection):
    python3 query_agent.py "What is the MW of tylenol?"

    # Restart first, then query (Overmind optimization iterations):
    python3 query_agent.py --restart "What is the MW of tylenol?"

Overmind shell command for optimization:
    python3 query_agent.py --restart "$INPUT"

Env vars:
    AGENT_URL            default http://localhost:8080
    AGENT_CONTAINER      docker container name to restart (default chemcrow-agent)
    RESTART_WAIT         seconds to wait after restart before querying (default 15)
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8080")
AGENT_CONTAINER = os.getenv("AGENT_CONTAINER", "chemcrow-agent")
RESTART_WAIT = int(os.getenv("RESTART_WAIT", "15"))


def restart_container():
    print(f"Restarting {AGENT_CONTAINER}...", flush=True)
    subprocess.run(
        ["docker", "restart", AGENT_CONTAINER],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for the agent server to be ready
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{AGENT_URL}/health", timeout=3):
                print("Agent ready.", flush=True)
                return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Agent at {AGENT_URL} did not become healthy after restart")


def query(q: str) -> str:
    payload = json.dumps({"query": q}).encode()
    req = urllib.request.Request(
        f"{AGENT_URL}/run",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
        return body.get("output", "")


def main():
    args = sys.argv[1:]
    do_restart = False

    if args and args[0] == "--restart":
        do_restart = True
        args = args[1:]

    if not args:
        print("Usage: query_agent.py [--restart] <question>", file=sys.stderr)
        sys.exit(1)

    q = " ".join(args)

    if do_restart:
        restart_container()

    try:
        print(query(q))
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        print(f"Agent error: {body.get('error', e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
