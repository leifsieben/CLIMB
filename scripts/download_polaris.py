import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
TASKS = (ROOT / "benchmarks" / "tasks" / "polaris_tasks.txt").read_text().split()
OUT = ROOT / "benchmarks" / "data" / "polaris"
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


            tr = list(zip([str(x) for x in train.inputs], list(train.targets)))
            te = [(str(x), "") for x in test.inputs]
            tgt = list(bench.target_cols)[0]
            metric = getattr(bench.main_metric, "label", str(bench.main_metric))
            ttype = str(list(bench.target_types.values())[0]).split(".")[-1].lower()
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
