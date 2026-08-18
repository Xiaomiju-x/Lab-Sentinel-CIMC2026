# Project story

Lab-Sentinel began with a concrete materials question: can a resource-limited
MCU observe a sintering process, judge material quality, explain its evidence
and still keep physical authority in a deterministic safety chain?

Near-infrared phosphor material and its sintering workflow provide the real
validation carrier. They connect formulation, process profile, crucible vision,
XRD phase information and PL spectra to observable quality decisions. The
project then generalizes the **method**, not the chemistry: source-bound data,
time-safe process–structure–property relations, quantized edge inference,
refusal and rollback are mapped to IC-material screening, virtual metrology,
SEM defects and advanced packaging.

The result has three inseparable layers:

1. a GD32H759 physical prototype with sensors, actuators, local HMI and voice;
2. a frozen 30-asset / 28-logical-model board runtime with real board evidence;
3. a separate 170-asset HOST research portfolio whose board deployment is not
   claimed.

This separation is intentional. It lets future contributors improve algorithms
without silently changing the control boundary or rewriting failed evidence.

