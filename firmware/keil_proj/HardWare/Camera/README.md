# OV5640 public-source boundary

`ov5640.h`, the team's SCCB/DCI integration contract, and a camera-disabled
public adapter are included. The early
`ov5640.c` contained a register initialization table described as adapted from
a vendor tutorial. No redistribution license for that table was found, so it is
intentionally excluded from the public repository.

The included `ov5640.c` returns an explicit unavailable status and never labels
its fallback as `LIVE_SENSOR`; this keeps the public Keil project link-complete.
To build live camera support, supply a permissively licensed OV5640 QVGA RGB565
implementation of the functions declared in `ov5640.h`, preserve its attribution
and verify the exact board pin map. Do not copy an online register table without
checking its license.
