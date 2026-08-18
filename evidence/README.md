# Public evidence

`public/` contains compact receipts selected from the private archival evidence
tree. Absolute local paths were replaced with repository-relative or
`private-archive://` locators. When that redaction changes JSON bytes, the field
`public_derivative_notice` explains that an embedded content root identifies
the archived original, not the derivative file itself.

Core reading order:

1. `release_gap_audit.v7.json` — 244 dispositions and the 78/92 split;
2. `host_closure.v7.json` — 170 identities, categories and hash uniqueness;
3. `physical_sd_copy.v9.json` — file-level microSD preposition;
4. `forge200_correct32gb_sharedbus_hardware_retest...json` — board rejection;
5. `forge200_new_sdmodule_three_retest_comparison...json` — three repeated CRC
   failures with a new same-model module.

Run `python tools/verify_evidence.py` to check the public ledger invariants.

