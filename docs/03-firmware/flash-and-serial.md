# Flash and serial evidence

Use a clean terminal at 115200 8N1 on PB13/PB5. For an acceptance run, capture:

- firmware identity and configuration defines;
- 30 runtime assets / 28 logical-model registration;
- golden/self-test results;
- DWT timing for the model under test;
- sensor presence, fault and freshness;
- heap minimum and critical-stack margin;
- watchdog/reset reason;
- deterministic actuator state on stop/fault.

Disconnect the PC network if demonstrating offline inference. A serial monitor
is allowed as an observer; a PC process must not calculate the answer.

Never publish raw serial logs without scanning for local paths, identity,
addresses and test-only secrets. The compact receipts in `evidence/public/` are
the reviewed public form.

