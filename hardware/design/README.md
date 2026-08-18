# Hardware design source and fabrication exports

This directory publishes the team's editable EasyEDA Pro design source and the
available manufacturing exports for the low-voltage Lab-Sentinel prototype.
The hardware is licensed under CERN-OHL-S-2.0 unless a file states otherwise.

## Contents

| Artifact | What it is | Reuse boundary |
|---|---|---|
| `lab-sentinel-hardware.epro2` | Editable EasyEDA Pro project, with author-account metadata removed | Open and review in a compatible EasyEDA Pro release; all functional document/link UUIDs are preserved |
| `camera-adapter/` | Camera-adapter drawing PDF and Gerber/drill ZIP | Fabrication reference for this prototype revision |
| `power-and-signal-board/` | Power/signal-board drawing PDF and Gerber/drill ZIP | Fabrication reference for this prototype revision |
| `relay-board/` | Relay-board drawing PDF and Gerber/drill ZIP | Fabrication reference for this prototype revision |

The three ZIP files contain Gerber layer and drill exports plus the team's
flying-probe export data. Exporter-supplied ordering instructions were removed
because they are not team-authored design material and are not required for
fabrication. The cleaned archives pass ZIP CRC inspection, but a fabricator and
an electrical reviewer must still check layer mapping, outline, drill units,
connector orientation, copper clearances and the actual load/voltage domains
before ordering.

## BOM and assembly boundary

[`docs/02-hardware/bom.csv`](../../docs/02-hardware/bom.csv) is a
**system-integration/module BOM**. It identifies the controller, sensors,
interfaces and prototype actuators used by the project. It is not a
reference-designator-level manufacturing BOM.

This release does **not** include:

- a PCBA-ready per-reference BOM with manufacturer part numbers and approved
  alternates;
- a component-placement/CPL (pick-and-place) file;
- a turnkey fabrication/assembly quotation package; or
- certification for industrial, mains-voltage, high-temperature or
  safety-critical service.

Anyone assembling a derivative must open the editable source, reconcile the
schematic and PCB against the physical module revision, export a fresh BOM/CPL
where applicable, and perform an independent electrical/design-rule review.

## Privacy-preserving source release

EasyEDA Pro embeds account metadata in nested `user` JSON objects. The public
project was processed with:

```bash
python tools/sanitize_easyeda_pro.py --in-place
python tools/sanitize_easyeda_pro.py --check
```

The sanitizer replaces only each `user` object with a neutral UUID. It verifies
that all other JSON semantics—including document/link UUIDs—remain unchanged,
that non-JSON members are byte-identical, that no captured author identity token
remains, and that the resulting archive passes ZIP CRC validation.

See [`docs/02-hardware/pcb.md`](../../docs/02-hardware/pcb.md) for fabrication
review steps and
[`docs/02-hardware/hardware-overview.md`](../../docs/02-hardware/hardware-overview.md)
for the prototype's electrical and safety boundary.
