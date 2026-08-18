# Public lwIP subset

This archive retains the lwIP runtime under `src/` and the three files in
`contrib/ports/freertos/` that are referenced by the public Keil target. The
selected target has PPP/PPPoE disabled; only `ppp_opts.h`, which supplies the
disabled feature macros used by lwIP core files, remains. The unused PPP
implementation, upstream host-side `makefsdata` utility, and upstream test
suite are intentionally omitted. The unrelated upstream codespell helper
scripts are also omitted; their LGPL terms are not part of the embedded
runtime subset shipped here.

The rest of the upstream `contrib/` tree contains desktop examples, generators
and unrelated tools; it is not part of Lab-Sentinel's firmware build and is not
redistributed here. In particular, an archived MIB compiler carried a private
strong-name key and separately licensed SharpSnmpLib code. That directory is
excluded from the clean public history. This project does not trust or support
artifacts signed with that key. Do not restore it from an older clone.

The upstream lwIP license is retained in `COPYING`.
