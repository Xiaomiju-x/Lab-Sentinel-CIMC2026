# On-device nano-LM

The nano-LM layer provides bounded diagnostic/explanation tasks under severe
MCU memory limits. It is separate from CI1302 voice recognition.

- One flagship logical model has three deployment-size assets.
- Seven domain experts are swap-loaded under a bounded bank.
- Existing distilled assets use compact Transformer-style C runtimes and W8
  weight paths where stated by their model card.
- Output is advisory and `authority=0`.
- Timeouts, unsupported prompts or invalid packages may refuse rather than
  generating an answer.

Do not describe this as general-purpose conversation, a cloud LLM or eight
simultaneously resident experts. The model card and package identity determine
the permitted task.

