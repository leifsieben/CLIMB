"""Download the 28 Polaris/TDCommons tasks (chemeleon_suite/tasks/polaris_tasks.txt) into standardized
CSVs under chemeleon_suite/data/polaris/<task>.csv with columns: smiles, y, split(train/test), plus a
manifest recording each task's target column, PRIMARY METRIC (Burns' comparison metric), and type.

NO LOGIN REQUIRED — verified 2026-08-13 that these public benchmarks load without `polaris login`.
Run with a polaris-lib >=0.13 env (needs Python >=3.10):  .venv_polaris/bin/python scripts/chemeleon_suite_fetch_polaris.py
No GPU. Idempotent: skips a task whose CSV already exists. Per-task failures are logged and skipped."""
import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
TASKS = (ROOT / "chemeleon_suite" / "tasks" / "polaris_tasks.txt").read_text().split()
OUT = ROOT / "chemeleon_suite" / "data" / "polaris"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import polaris as po
    manifest = {}
    for task in TASKS:
        dst = OUT / (task.replace("/", "__") + ".csv")
        if dst.exists():
            print(f"[polaris] SKIP {task} (exists)"); continue
        try:
            bench = po.load_benchmark(task)
            train, test = bench.get_train_test_split()
            # 0.13 API: Subset.inputs (smiles) + Subset.targets. Polaris HIDES test targets by design
            # (TestAccessError) -> we store test INPUTS only, in test order, and score later via
            # benchmark.evaluate(y_pred) which compares against the held-out labels (verified offline).
            tr = list(zip([str(x) for x in train.inputs], list(train.targets)))
            te = [(str(x), "") for x in test.inputs]   # test labels intentionally absent
            tgt = list(bench.target_cols)[0]
            metric = getattr(bench.main_metric, "label", str(bench.main_metric))
            ttype = str(list(bench.target_types.values())[0]).split(".")[-1].lower()  # REGRESSION -> regression
        except Exception as exc:
            print(f"[polaris] FAIL {task}: {type(exc).__name__}: {exc}", file=sys.stderr); continue
        ttype = "classification" if "class" in ttype else "regression"
        with dst.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["smiles", "y", "split"])
            for s, y in tr:
                w.writerow([s, "" if y is None else y, "train"])
            for s, y in te:
                w.writerow([s, "" if y is None else y, "test"])
        manifest[task] = {"file": dst.name, "target_col": tgt, "primary_metric": metric,
                          "type": ttype, "n_train": len(tr), "n_test": len(te)}
        print(f"[polaris] OK {task}: train={len(tr)} test={len(te)} type={ttype} metric={metric}")
    (OUT / "polaris_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[polaris] wrote {len(manifest)}/{len(TASKS)} tasks -> {OUT}")


if __name__ == "__main__":
    main()
