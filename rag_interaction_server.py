from __future__ import annotations

import argparse
import html
import json
import threading
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
INTERACT_PATH = "/interact"
MESSAGES_PATH = "/messages"
VALID_DIRECTIONS = {"left", "right", "none"}
MAX_STORED_MESSAGES = 200

_MESSAGES: deque[dict] = deque(maxlen=MAX_STORED_MESSAGES)
_MESSAGES_LOCK = threading.Lock()


def add_message(message: dict) -> None:
    with _MESSAGES_LOCK:
        _MESSAGES.appendleft(message)


def get_messages() -> list[dict]:
    with _MESSAGES_LOCK:
        return list(_MESSAGES)


def render_messages_html(messages: list[dict]) -> str:
    rows = []
    for message in messages:
        received_at = html.escape(str(message.get("received_at_utc", "")))
        direction = html.escape(str(message.get("direction", "")))
        prompt = html.escape(str(message.get("prompt", "")))
        rows.append(
            "<tr>"
            f"<td>{received_at}</td>"
            f"<td>{direction}</td>"
            f"<td>{prompt}</td>"
            "</tr>"
        )

    table_rows = "\n".join(rows) if rows else (
        '<tr><td colspan="3">No messages received yet.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Interaction Messages</title>
  <style>
    body {{
      font-family: sans-serif;
      margin: 24px;
      background: #f7f7f7;
      color: #111;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    p {{
      margin-top: 0;
      color: #444;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
    }}
    th, td {{
      padding: 10px 12px;
      border: 1px solid #ddd;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f0f0f0;
    }}
    code {{
      background: #eee;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>Received Messages</h1>
  <p>
    POST messages to <code>{INTERACT_PATH}</code>.
    View raw JSON at <code>{MESSAGES_PATH}</code>.
  </p>
  <table>
    <thead>
      <tr>
        <th>Received At (UTC)</th>
        <th>Direction</th>
        <th>Prompt</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>"""


class RagInteractionHandler(BaseHTTPRequestHandler):
    server_version = "RagInteractionServer/0.1"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(HTTPStatus.OK, render_messages_html(get_messages()))
            return

        if self.path == MESSAGES_PATH:
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "count": len(get_messages()),
                    "messages": get_messages(),
                },
            )
            return

        if self.path != "/health":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "Not found"},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "status": "healthy",
                "endpoint": INTERACT_PATH,
            },
        )

    def do_POST(self) -> None:
        if self.path != INTERACT_PATH:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "Not found"},
            )
            return

        content_length_header = self.headers.get("Content-Length", "0").strip()
        try:
            content_length = int(content_length_header)
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid Content-Length header"},
            )
            return

        raw_body = self.rfile.read(max(0, content_length))
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Body must be valid JSON"},
            )
            return

        if not isinstance(payload, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "JSON body must be an object"},
            )
            return

        print(f"[RAG_SERVER][RAW_PAYLOAD] {payload}", flush=True)

        prompt = payload.get("prompt")
        direction = payload.get("direction", "none")

        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Field 'prompt' must be a non-empty string"},
            )
            return

        if not isinstance(direction, str):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Field 'direction' must be a string"},
            )
            return

        direction = direction.strip().lower()
        if direction not in VALID_DIRECTIONS:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "ok": False,
                    "error": f"Field 'direction' must be one of {sorted(VALID_DIRECTIONS)}",
                },
            )
            return

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        clean_prompt = prompt.strip()
        clean_payload = {
            "prompt": clean_prompt,
            "direction": direction,
            "received_at_utc": timestamp_utc,
        }
        add_message(clean_payload)

        print(f"[RAG_SERVER][RECOGNIZED_PHRASE] {clean_prompt}", flush=True)
        print(f"[RAG_SERVER][DIRECTION] {direction}", flush=True)
        print(f"[RAG_SERVER] {clean_payload}", flush=True)

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "message": "Interaction received",
                "data": clean_payload,
            },
        )

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, status: HTTPStatus, payload: str) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple REST server for rag_interaction.py",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind host. Default: {DEFAULT_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port. Default: {DEFAULT_PORT}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RagInteractionHandler)
    print(
        f"[RAG_SERVER] Listening on http://{args.host}:{args.port}{INTERACT_PATH}",
        flush=True,
    )
    print(
        f"[RAG_SERVER] Message viewer: http://{args.host}:{args.port}/",
        flush=True,
    )
    print(
        f"[RAG_SERVER] Messages JSON: http://{args.host}:{args.port}{MESSAGES_PATH}",
        flush=True,
    )
    print(
        f"[RAG_SERVER] Health check: http://{args.host}:{args.port}/health",
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[RAG_SERVER] Stopping...", flush=True)
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
