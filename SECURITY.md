# Security policy

## Supported versions

Security fixes target the latest `v1.x` release and the `main` branch.

## Reporting

Use **GitHub Private Vulnerability Reporting** for credentials, parser/memory
safety issues, model-package or catalog validation bypasses, generation-counter
rollback, voice/touch command injection, watchdog/interlock bypass, or any path
that could produce an unsafe actuator command. Do not place sensitive details
in a public issue.

We aim to acknowledge a private report within seven days and provide an initial
assessment within thirty days. These are goals, not a service-level guarantee.

## Safety disclaimer

Lab-Sentinel is a research and competition prototype. It is not certified to
IEC 61508 or any functional-safety, EMC, metrology or industrial-furnace
standard. Never let the AI path directly control a real furnace. Independent
hardware limits, emergency stop, watchdog, deterministic control and trained
human supervision are required for any physical adaptation.

If you discover a committed secret, revoke it first, then report the affected
revision privately. Removing a string from the latest commit does not remove it
from Git history.

