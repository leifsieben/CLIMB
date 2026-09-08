---
license: cc-by-4.0
pretty_name: CLIMB raw evaluation results
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - chemistry
  - molecular-property-prediction
  - benchmark-results
  - reproducibility
  - moleculenet
---

# CLIMB evaluation results

Per-run evaluation outputs for every arm in CLIMB. Every figure and table in the paper is computed
from these files; none is stored pre-aggregated, so a reported number can be recomputed from the
per-molecule predictions.

## Layout

```
<wave>/<run>/moleculenet_cv/          5-fold scaffold CV on BACE, Tox21, QM7, HIV
chemeleon_suite/moleculeace/<run>/    30 activity-cliff targets, 3 evaluation seeds
chemeleon_suite/polaris/<run>/        Polaris tasks, predictions plus scores
cbs_benchmark/<run>/moleculenet_cv/   CBS rare-actives virtual screen, benchmark's own folds
wong_saureus/<run>/, fartdb/<run>/    the two additional external suites
analysis_rigor/                       bootstrap and multiple-testing outputs
```

## Files

| File | Contents |
|---|---|
| `moleculenet_summary.csv` | one row per head seed and fold, all metrics computed for that dataset |
| `suite_summary.json` | dataset means over folds and seeds |
| `test_predictions.csv` | per-molecule out-of-fold predictions; the source of every error bar |
| `polaris_scores.csv` | Polaris scores from the official evaluator against held-out labels |
| `results.csv` | suite-level long table: task, seed, subset, metric, value |
| `verified.json` | written only when every task and seed for that cell completed |

Polaris test labels are held out by the benchmark, so predictions are scored off-box by
[`scripts/chemeleon_suite_score_polaris.py`](https://github.com/leifsieben/CLIMB/blob/v2-redux/scripts/chemeleon_suite_score_polaris.py). A Polaris `results.csv` therefore carries no scores by
design, and `polaris_scores.csv` is the scored artifact.

## Related

- Code: [github.com/leifsieben/CLIMB](https://github.com/leifsieben/CLIMB)
- Encoders: [`lsieben/climb-encoders`](https://huggingface.co/lsieben/climb-encoders)
- Results: [`lsieben/climb-results`](https://huggingface.co/datasets/lsieben/climb-results)
- Pretraining data: [`lsieben/climb-pretrain-data`](https://huggingface.co/datasets/lsieben/climb-pretrain-data)

## Citation

```bibtex
@misc{climb2026,
  title  = {Does Pretraining Teach Chemical Language Models Chemistry?},
  author = {Sieben, Leif and Zimmermann, Yoel},
  year   = {2026},
  note   = {Preprint, arXiv},
  url    = {https://github.com/leifsieben/CLIMB}
}
```

## License

CC-BY-4.0.
