"""Traceability guard for climb_figures.ipynb.

The paper's figures come from figures_out/, and the notebook is the record of HOW they were made.
That record is only trustworthy if three things are the same thing:

    notebook_cells/  ==  climb_figures.ipynb (on disk)  ==  .notebook_generator_hash

and the on-disk notebook is actually executed (its figure cells carry embedded PNGs) and committed.
This has silently broken before -- a parallel session regenerated the notebook and it began
producing DIFFERENT figures than the ones in figures_out/. This script makes that failure loud.

Exit code 0 = all consistent. Non-zero = something is out of sync; the message says what.
Run from the repo root:  python scripts/verify_notebook_sync.py
"""
import json, hashlib, subprocess, sys
from pathlib import Path

NB = Path("climb_figures.ipynb")
CELLS = Path("notebook_cells")
STAMP = Path(".notebook_generator_hash")

def _hash(cells):
    return hashlib.sha256(json.dumps(
        [("".join(c["source"]), c["cell_type"]) for c in cells]).encode()).hexdigest()

def main():
    problems = []

    nb = json.loads(NB.read_text())
    disk = _hash(nb["cells"])

    # 1) on-disk notebook sources match the stored generator hash
    stored = STAMP.read_text().strip() if STAMP.exists() else None
    if stored != disk:
        problems.append(f"on-disk notebook sources ({disk[:12]}) != .notebook_generator_hash "
                        f"({(stored or 'MISSING')[:12]}). The notebook was edited outside the "
                        f"generator, or the hash is stale. Rebuild from notebook_cells/ or --export.")

    # 2) notebook_cells/ rebuilds to the same sources as the on-disk notebook
    files = sorted(CELLS.glob("*.[mp][dy]"))
    rebuilt = [{"source": f.read_text().splitlines(keepends=True),
                "cell_type": "markdown" if f.suffix == ".md" else "code"} for f in files]
    if _hash(rebuilt) != disk:
        problems.append("notebook_cells/ would rebuild to DIFFERENT sources than the on-disk "
                        ".ipynb. The two have diverged -- run scripts/build_figure_notebook.py "
                        "(or --export the other direction) so they agree.")

    # 3) figure cells are actually executed (carry an embedded PNG)
    figcells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
                and any("image/png" in o.get("data", {}) for o in c.get("outputs", []))]
    if not figcells:
        problems.append("no code cell carries an embedded PNG -- the committed notebook is NOT "
                        "executed, so opening it shows no figures. Execute it before committing.")

    # 4) the notebook, cells and hash are committed (no uncommitted diff)
    dirty = subprocess.run(["git", "diff", "--quiet", "--",
                            str(NB), str(CELLS), str(STAMP)]).returncode != 0
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--",
                             str(NB), str(CELLS), str(STAMP)]).returncode != 0
    if dirty or staged:
        problems.append("notebook / cells / hash have uncommitted changes -- commit them so the "
                        "version on disk is the version in git history.")

    if problems:
        print("NOTEBOOK SYNC: FAIL")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"NOTEBOOK SYNC: OK  ({len(nb['cells'])} cells, {len(figcells)} executed figure cells, "
          f"sources hash {disk[:12]}, everything committed)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
