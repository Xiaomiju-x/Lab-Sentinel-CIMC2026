#!/usr/bin/env python3
"""Resumable SFTP download with known-host enforcement and final SHA verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko


def write_state(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(hostname=host, port=port, username=user, password=password, look_for_keys=False, allow_agent=False, timeout=20, banner_timeout=20, auth_timeout=20)
    return client


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote", required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=120)
    args = parser.parse_args()
    password = os.environ.get("CIMC_SSH_PASSWORD")
    if not password:
        raise RuntimeError("CIMC_SSH_PASSWORD is required")
    target = args.local.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if target.exists() and target.stat().st_size == args.expected_bytes and sha256_file(target) == args.expected_sha256:
        write_state(args.state, {"status": "PASS", "local": str(target), "bytes": args.expected_bytes, "sha256": args.expected_sha256, "retries": 0})
        return 0
    retries = 0
    while retries <= args.max_retries:
        client = None
        try:
            client = connect(args.host, args.port, args.user, password)
            transport = client.get_transport()
            if transport is None:
                raise RuntimeError("SSH_TRANSPORT_MISSING")
            sftp = paramiko.SFTPClient.from_transport(
                transport,
                window_size=64 * 1024 * 1024,
                max_packet_size=1024 * 1024,
            )
            try:
                remote_size = sftp.stat(args.remote).st_size
                if remote_size != args.expected_bytes:
                    raise RuntimeError(f"REMOTE_SIZE:{remote_size}")
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > remote_size:
                    partial.unlink()
                    offset = 0
                write_state(args.state, {"status": "DOWNLOADING", "local": str(target), "partial": str(partial), "downloaded_bytes": offset, "expected_bytes": remote_size, "retries": retries})
                with sftp.open(args.remote, "rb") as remote_handle, partial.open("ab") as local_handle:
                    remote_handle.seek(offset)
                    remote_handle.prefetch(file_size=remote_size, max_concurrent_requests=64)
                    since_state = 0
                    while offset < remote_size:
                        block = remote_handle.read(min(4 * 1024 * 1024, remote_size - offset))
                        if not block:
                            raise RuntimeError("REMOTE_EOF")
                        local_handle.write(block)
                        offset += len(block)
                        since_state += len(block)
                        if since_state >= 16 * 1024 * 1024:
                            local_handle.flush()
                            write_state(args.state, {"status": "DOWNLOADING", "local": str(target), "partial": str(partial), "downloaded_bytes": offset, "expected_bytes": remote_size, "retries": retries})
                            since_state = 0
            finally:
                sftp.close()
            actual_sha = sha256_file(partial)
            if partial.stat().st_size != args.expected_bytes or actual_sha != args.expected_sha256:
                raise RuntimeError(f"LOCAL_IDENTITY:{partial.stat().st_size}:{actual_sha}")
            os.replace(partial, target)
            write_state(args.state, {"status": "PASS", "local": str(target), "bytes": target.stat().st_size, "sha256": actual_sha, "retries": retries})
            return 0
        except Exception as exc:
            retries += 1
            write_state(args.state, {"status": "RETRYING" if retries <= args.max_retries else "FAIL", "error": f"{type(exc).__name__}:{exc}", "local": str(target), "downloaded_bytes": partial.stat().st_size if partial.exists() else 0, "expected_bytes": args.expected_bytes, "retries": retries})
            if retries > args.max_retries:
                return 1
            time.sleep(15)
        finally:
            if client is not None:
                client.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
