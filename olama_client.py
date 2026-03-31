"""
Simple Ollama client for sending a prompt to a remote server.

Example:
    python3 olama_client.py "Write a short greeting"
    python3 olama_client.py --model qwen3.5:2b "Explain hand tracking"
"""

from __future__ import annotations

import argparse
import sys

import requests


DEFAULT_BASE_URL = "http://192.168.50.26:11434"
DEFAULT_MODEL = "qwen3.5:2b"
DEFAULT_TIMEOUT_SEC = 120


def send_prompt(
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()

    payload = response.json()
    text = payload.get("response", "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned an unexpected response: {payload}")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to a remote Ollama server.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text to send. If omitted, the script will ask for it.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_BASE_URL,
        help=f"Ollama server URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name on the Ollama server. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Request timeout in seconds. Default: {DEFAULT_TIMEOUT_SEC}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = " ".join(args.prompt).strip()

    if not prompt:
        prompt = input("Prompt: ").strip()

    if not prompt:
        print("Prompt cannot be empty.", file=sys.stderr)
        return 1

    try:
        answer = send_prompt(
            prompt=prompt,
            base_url=args.host,
            model=args.model,
            timeout_sec=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"Failed to reach Ollama server: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
