# Offline train/evaluation overlap audit

Use this lightweight, dependency-free command to find **exact prompt overlaps** between
a training split and an evaluation split before reporting model results:

`python tools/audit_dataset_overlap.py data/datasets/instruction_chat.json data/datasets/instruction_chat.json`

The example intentionally uses the same file twice and should report overlap (exit code 1).
Provide two distinct split files in a real audit. JSON arrays of objects and JSONL
(one object per line) are supported. The script compares `instruction`, `prompt`, or
user-role `messages` content after Unicode NFKC, case-folding and whitespace normalization.

The JSON output includes split sizes, duplicate counts, overlap counts and at most
10 matching **row indices**. It deliberately does not print training text.
Exit codes: 0 = no identical prompts, 1 = overlap found, 2 = invalid input.
No network, model weights or GPU are needed; source files are never modified.

**Limitations:** This detects exact normalized prompt overlap, not semantic paraphrases,
answer-only duplication or general train/eval contamination. A clean result does not
prove a leakage-free benchmark.
