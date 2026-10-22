# Evaluation

```sh
conversaflow train
conversaflow eval --split eval --json /tmp/conversaflow-eval.json
conversaflow eval --split test --markdown docs/EVALUATION_RESULTS.md
```

Intent rows are JSONL objects containing `text` and `intent`. The trainer uses a deterministic held-out split of the training corpus for calibration; the `eval` and `test` files are separate evaluation inputs. Golden JSON conversations in `data/golden/` exercise dialogue success and containment.

Reports include accuracy, macro F1, confidence, prediction margin, low-confidence rate, per-intent precision/recall/F1, and conversation success. The CLI exports results and exits successfully after evaluation; it does not enforce a model quality threshold. CI requires at least 90% intent accuracy, 90% macro F1, and 100% golden-conversation success on the bundled test fixtures. Review the actual reported values alongside these gates.

These are synthetic, repository-bundled fixtures. High scores do not establish generalization, fairness, production containment, real integration correctness, or a latency SLA. Extend corpora with independently labeled domain examples before deployment. Keep training and evaluation inputs separate and inspect failures rather than tuning directly against the test set.

The Python test suite checks implementation behavior. Coverage measures exercised source and branches; neither coverage nor test pass rate proves the absence of defects. Measured results are recorded in [TEST_RESULTS.md](TEST_RESULTS.md).
