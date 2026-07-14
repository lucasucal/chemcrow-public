#!/usr/bin/env python3
"""ChemCrow agent entrypoint.

Two modes:
  CLI (one-shot):   python run_agent.py "What is the MW of tylenol?"
  Server (default): python run_agent.py
                    POST /run  {"query": "..."}  -> {"output": "..."}
                    GET  /health                 -> {"status": "ok"}

Overmind and other callers should use the server mode — the agent is
initialised once and reused across all requests, which is much faster
than cold-starting it per query.
"""

import json
import logging
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from overmind import entry_point, init

init(service_name="ChemCrow")

from chemcrow.agents import ChemCrow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PORT = int(os.getenv("AGENT_PORT", "8000"))


@entry_point("ChemCrow")
def run_chemcrow(agent: ChemCrow, query: str) -> str:
    return agent.run(query)


def build_agent() -> ChemCrow:
    log.info("Initialising ChemCrow agent...")
    agent = ChemCrow(streaming=False, local_rxn=True)
    log.info("Agent ready.")
    return agent


class Handler(BaseHTTPRequestHandler):
    agent: ChemCrow = None  # set after init

    def log_message(self, fmt, *args):  # suppress default access log noise
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self._send_json(400, {"error": "invalid JSON"})
            return

        query = (payload.get("query") or "").strip()
        if not query:
            self._send_json(400, {"error": "missing 'query' field"})
            return

        log.info("Query: %s", query)
        try:
            output = run_chemcrow(self.agent, query)
            self._send_json(200, {"output": output})
        except Exception as e:
            log.error("Agent error: %s", traceback.format_exc())
            self._send_json(500, {"error": str(e)})


def serve():
    agent = build_agent()
    Handler.agent = agent
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    log.info("Listening on port %d  (POST /run, GET /health)", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cli(query: str):
    agent = build_agent()
    print(run_chemcrow(agent, query))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli(" ".join(sys.argv[1:]))
    else:
        serve()
