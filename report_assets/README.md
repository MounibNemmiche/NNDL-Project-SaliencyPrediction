# Report Asset Provenance

This directory contains only small, report-ready evidence. Raw datasets, full checkpoints, and complete run directories belong in ignored local storage or a release asset.

## Required Final Assets

| Asset | Producer | Required provenance |
|---|---|---|
| `results_table.csv` | Student B, verified by Student A | Run names, model, loss, seed, split hash, checkpoint hash |
| `experiment_manifest.csv` | Student B | Exact CLI, Git SHA, device, versions, duration, sample counts |
| `fusion_weights.csv` | Student B | Fusion checkpoint and ordered side-head names |
| `loss_curve.png` | Student B | Source training CSV and run name |
| `cc_curve.png` | Student B | Source training CSV and run name |
| `qualitative_examples.png` | Student B, verified by Student A | Image IDs and per-image metric source |
| architecture figure | Both students | Source file and model commit |
| `contribution_log.md` | Both students | Factual implementation and review ownership |
| `references_notes.md` | Student A | Citation keys and claims supported by each source |

## Frozen Dataset Split

```text
Validation: data/splits/val_seed42.txt, 500 files, sha256 prefix 28634c48abb6
Final test: data/splits/test_seed42.txt, 4,500 files, sha256 prefix 7ad21b6a799f
```

The final test split must not be used for checkpoint selection or model tuning.

## Acceptance Rule

No metric may be manually typed into a plot or report table. Every value must be traceable to a saved evaluation CSV generated from a named checkpoint and the frozen final-test manifest.
