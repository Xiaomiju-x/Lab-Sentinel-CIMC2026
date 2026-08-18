# Splits and leakage prevention

Random row splits are insufficient when rows share batch, material family,
paper family or time-adjacent process context.

Project rules:

- split by the strongest dependency unit available (batch/family/document);
- fit scalers, vocabularies and imputers on training only;
- use validation for selection and keep test frozen;
- keep calibration data separate from final test data;
- block post-burn XRD/PL/SEM/EDS from pre-burn prediction inputs;
- hash and audit cross-split identities;
- do not use teacher/API answers as ground-truth labels;
- report the unit and overlap count with every metric.

SinterGraph's pre/post-burn cutoff is an evidence contract even where its
dedicated model candidate was rejected.

