# PCB design release

Editable team-authored hardware files and manufacturing exports are in
`hardware/design/` and licensed CERN-OHL-S-2.0 unless an included file says
otherwise.

The design integrates sensor connectors, storage/time, touch/voice, low-voltage
actuation and protection around the competition GD32H759 platform. It replaces
the early jumper-wire prototype; it is not represented as an independently
certified industrial controller.

Before fabrication:

1. verify the exact core-board connector and voltage domains;
2. audit SDRAM/RGB shared pins against the selected firmware target;
3. verify relay and alarm active polarity on the selected modules;
4. check creepage/current/thermal rules for the **actual low-voltage loads**;
5. add revision, date and non-identifying board mark required by your context;
6. review every third-party footprint and connector datasheet.

The PDFs/zips under component subdirectories are convenient exports; the EDA
source remains the preferred editable artifact.

The published `bom.csv` is a system/module integration list, not a
PCBA-ready reference-designator BOM. This archival release does not include a
validated CPL/pick-and-place file, approved alternates or a per-footprint MPN
table. Open the sanitized EasyEDA Pro source, confirm the manufactured revision
against the photographs, and export/review fabrication data before ordering.
