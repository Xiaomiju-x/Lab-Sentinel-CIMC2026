#!/usr/bin/env python3
"""Small non-interactive SSH/SFTP helper; credentials are read only from env."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko


def connect(host: str, port: int, user: str) -> paramiko.SSHClient:
    password = os.environ.get("CIMC_SSH_PASSWORD")
    if not password:
        raise RuntimeError("CIMC_SSH_PASSWORD is required")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--command")
    action.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    action.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    args = parser.parse_args()
    client = connect(args.host, args.port, args.user)
    try:
        if args.command is not None:
            stdin, stdout, stderr = client.exec_command(args.command, get_pty=False)
            stdin.close()
            for chunk in iter(lambda: stdout.channel.recv(65536), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            while stdout.channel.recv_stderr_ready():
                sys.stderr.buffer.write(stdout.channel.recv_stderr(65536))
            status = stdout.channel.recv_exit_status()
            remaining = stderr.read()
            if remaining:
                sys.stderr.buffer.write(remaining)
            return status
        sftp = client.open_sftp()
        try:
            if args.put:
                local, remote = args.put
                sftp.put(str(Path(local).resolve()), remote)
            elif args.get:
                remote, local = args.get
                target = Path(local).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(target))
        finally:
            sftp.close()
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
