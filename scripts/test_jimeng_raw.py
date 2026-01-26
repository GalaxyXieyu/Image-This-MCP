#!/usr/bin/env python3
"""
Standalone Jimeng API smoke test (no external deps).

Reads JIMENG_ACCESS_KEY / JIMENG_SECRET_KEY from env, sends a request,
prints response summary, and saves any returned images to disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

HOST = "visual.volcengineapi.com"
REGION = "cn-north-1"
SERVICE = "cv"
VERSION = "2022-08-31"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def resolve_output_dir() -> Path:
    output_dir = os.getenv("IMAGE_OUTPUT_DIR", "").strip()
    if not output_dir:
        output_dir = str(Path.home() / "nanobanana-images")
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def get_signature_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = sign(secret_key.encode(), date_stamp)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    k_signing = sign(k_service, "request")
    return k_signing


def build_authorization(
    access_key: str,
    secret_key: str,
    method: str,
    body: str,
    timestamp: str,
    headers: Dict[str, str],
) -> str:
    sorted_headers = sorted(headers.keys())
    canonical_headers = "\n".join(
        f"{key.lower()}:{headers[key].strip()}" for key in sorted_headers
    ) + "\n"
    signed_headers = ";".join(key.lower() for key in sorted_headers)

    payload_hash = hashlib.sha256(body.encode()).hexdigest()
    canonical_request = "\n".join(
        [
            method,
            "/",
            f"Action=CVProcess&Version={VERSION}",
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    date_stamp = timestamp[:8]
    credential_scope = f"{date_stamp}/{REGION}/{SERVICE}/request"
    hashed_canonical = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = "\n".join(
        ["HMAC-SHA256", timestamp, credential_scope, hashed_canonical]
    )

    signing_key = get_signature_key(secret_key, date_stamp, REGION, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    return (
        f"HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )


def build_request_body(prompt: str, width: int, height: int) -> Dict[str, Any]:
    return {
        "req_key": "jimeng_t2i_v40",
        "req_json": "{}",
        "prompt": prompt,
        "width": width,
        "height": height,
        "scale": 0.5,
        "force_single": True,
    }


def http_post(
    url: str,
    body: str,
    headers: Dict[str, str],
    timeout: int,
) -> Tuple[int, str]:
    data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            payload = response.read().decode("utf-8")
            return status, payload
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        return exc.code, payload


def request_with_retry(
    access_key: str,
    secret_key: str,
    body: Dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_delay: int,
) -> Dict[str, Any]:
    url = f"https://{HOST}/?Action=CVProcess&Version={VERSION}"
    body_str = json.dumps(body, ensure_ascii=False)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        if attempt > 0:
            delay = retry_delay * (2 ** (attempt - 1))
            print(f"INFO: Retry {attempt + 1}/{max_retries} after {delay}s")
            time.sleep(delay)

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        payload_hash = hashlib.sha256(body_str.encode()).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "Host": HOST,
            "X-Date": timestamp,
            "X-Content-Sha256": payload_hash,
        }
        headers["Authorization"] = build_authorization(
            access_key, secret_key, "POST", body_str, timestamp, headers
        )

        status, payload = http_post(url, body_str, headers, timeout)
        try:
            response = json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error = exc
            print(f"ERROR: Failed to decode JSON response (status {status}).")
            if status in RETRYABLE_STATUS and attempt < max_retries - 1:
                continue
            raise

        if status >= 400:
            print(f"ERROR: HTTP {status}")
            if status in RETRYABLE_STATUS and attempt < max_retries - 1:
                continue
            return response

        return response

    raise last_error or RuntimeError("All retry attempts failed")


def summarize_response(response: Dict[str, Any]) -> None:
    code = response.get("code")
    message = response.get("message")
    request_id = response.get("request_id") or response.get("requestId")
    print(f"Response code: {code}")
    if message:
        print(f"Response message: {message}")
    if request_id:
        print(f"Request ID: {request_id}")

    data = response.get("data") or {}
    data_keys = ", ".join(sorted(data.keys()))
    print(f"Data keys: {data_keys or '(none)'}")

    b64_list = data.get("binary_data_base64") or []
    first_len = len(b64_list[0]) if b64_list else 0
    print(f"binary_data_base64 items: {len(b64_list)}, first length: {first_len}")


def save_images(response: Dict[str, Any], output_dir: Path) -> int:
    data = response.get("data") or {}
    b64_list = data.get("binary_data_base64") or []
    if not b64_list:
        print("ERROR: No binary_data_base64 returned.")
        return 1

    saved = 0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for idx, b64 in enumerate(b64_list, start=1):
        if not b64:
            continue
        image_bytes = base64.b64decode(b64)
        filename = f"jimeng_raw_{timestamp}_{idx:02d}.png"
        path = output_dir / filename
        with open(path, "wb") as f:
            f.write(image_bytes)
        print(f"OK: Saved image: {path} ({len(image_bytes)} bytes)")
        saved += 1

    if not saved:
        print("ERROR: No images saved (empty base64 entries).")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Jimeng raw API smoke test")
    parser.add_argument("--prompt", default="A minimal illustration of a rocket on a gradient sky")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("JIMENG_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=5)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    access_key = os.getenv("JIMENG_ACCESS_KEY")
    secret_key = os.getenv("JIMENG_SECRET_KEY")
    if not access_key or not secret_key:
        print("ERROR: Missing JIMENG_ACCESS_KEY or JIMENG_SECRET_KEY.")
        return 1

    output_dir = resolve_output_dir()
    print(f"Output directory: {output_dir}")

    exit_code = 0
    for i in range(args.n):
        if args.n > 1:
            print(f"INFO: Request {i + 1}/{args.n}")

        body = build_request_body(args.prompt, args.width, args.height)
        response = request_with_retry(
            access_key=access_key,
            secret_key=secret_key,
            body=body,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        summarize_response(response)

        if args.inspect_only:
            continue

        exit_code = max(exit_code, save_images(response, output_dir))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
