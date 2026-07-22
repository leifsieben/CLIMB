"""Rebuild climb_figures.ipynb from the per-cell sources in notebook_cells/.

Written as a generator rather than edited in place because the notebook had accreted duplicates
(two A1 cells computing different things), stale captions, and figures nobody uses -- so "which
version is right?" had no answer. One file, one definition per figure, generated.

WHERE THE CELLS LIVE (changed 2026-07-22). The cell sources used to be embedded in this file as
giant triple-quoted literals. That made this script the source of truth, and on 2026-07-22 a
rebuild silently reverted a parallel session's work: the notebook had gained A1.a/A1.b, A2.a/A2.b,
the Table A1 summary, the e2e baseline wiring and the I1/H1 scheme fixes, and a regeneration
threw all of it away because this file had never heard of them. The guard below now prevents the
clobber, but a guard alone still leaves a stale generator that would re-emit the OLD figures the
moment someone passes --force.

So the cells now live in `notebook_cells/NN.{md,py}` -- one real, diffable, editable file per
cell. Editing the notebook and re-exporting, or editing the files and regenerating, both converge
on the same thing, and there is exactly one definition of each cell again.

Figures: A1.a/A1.b, A2.a/A2.b, B1p1, E1, B2, C1J1 (fused), I1, H1, plus the Table A1 summary and
the inventory.

Usage:
    python scripts/build_figure_notebook.py            # refuses if the notebook has newer edits
    python scripts/build_figure_notebook.py --force    # overwrite (fold edits in FIRST)
    python scripts/build_figure_notebook.py --out /tmp/check.ipynb   # write elsewhere, no guard
    python scripts/build_figure_notebook.py --export   # notebook -> notebook_cells/ (round-trip)
"""
import json
from pathlib import Path

CELL_DIR = Path("notebook_cells")

def load_cells():
    """Assemble the notebook's cells from notebook_cells/NN.{md,py}, ordered by filename."""
    files = sorted(CELL_DIR.glob("*.[mp][dy]"))
    if not files:
        raise SystemExit(f"no cell sources in {CELL_DIR}/ -- nothing to build")
    out = []
    for f in files:
        source = f.read_text().splitlines(keepends=True)
        if f.suffix == ".md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": source})
    return out

def export_cells(target):
    """Inverse of load_cells(): notebook -> notebook_cells/. Keeps the two in sync after a
    hand edit in Jupyter, so the next rebuild does not revert it."""
    nb = json.loads(Path(target).read_text())
    CELL_DIR.mkdir(exist_ok=True)
    for old in CELL_DIR.glob("*.[mp][dy]"):
        old.unlink()
    for i, c in enumerate(nb["cells"]):
        ext = "md" if c["cell_type"] == "markdown" else "py"
        (CELL_DIR / f"{i:02d}.{ext}").write_text("".join(c["source"]))
    print(f"exported {len(nb['cells'])} cells -> {CELL_DIR}/")

cells = load_cells()

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.9"}},
      "nbformat": 4, "nbformat_minor": 5}
# ---- never clobber an edited notebook -------------------------------------------------------
# This generator overwrites climb_figures.ipynb wholesale. On 2026-07-22 an automated rebuild did
# exactly that to a notebook someone had edited by hand (it had produced figA1a/figA1b/figA2a/
# figA2b minutes earlier); the rendered PNGs survived, the cells did not. So: always snapshot the
# existing notebook first, and refuse to overwrite one this generator did not itself produce
# unless the caller explicitly says to.
import hashlib, os, shutil, sys, time

if "--export" in sys.argv:
    export_cells("climb_figures.ipynb"); raise SystemExit(0)

_out_i  = sys.argv.index("--out") + 1 if "--out" in sys.argv else None
TARGET  = Path(sys.argv[_out_i]) if _out_i else Path("climb_figures.ipynb")
_guarded = _out_i is None          # only guard the real notebook
STAMP  = Path(".notebook_generator_hash")          # hash of the last notebook WE wrote
new_text = json.dumps(nb, indent=1)

if _guarded and TARGET.exists():
    backups = Path("notebook_backups"); backups.mkdir(exist_ok=True)
    snap = backups / f"climb_figures_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.ipynb"
    shutil.copy2(TARGET, snap)

    # Compare only the SOURCE of each cell: executing the notebook rewrites outputs and
    # execution counts, which are not edits and must not be mistaken for them.
    def _src(path):
        try:
            d = json.loads(Path(path).read_text())
        except Exception:
            return None
        return hashlib.sha256(
            json.dumps([("".join(c["source"]), c["cell_type"]) for c in d["cells"]]).encode()
        ).hexdigest()

    cur = _src(TARGET)
    known = STAMP.read_text().strip() if STAMP.exists() else None
    if known and cur and cur != known and "--force" not in sys.argv:
        print(f"REFUSING to overwrite climb_figures.ipynb: its cells differ from the last version\n"
              f"this generator wrote, so it contains hand edits that would be destroyed.\n"
              f"  a snapshot of it is at: {snap}\n"
              f"  re-run with --force to overwrite anyway, after folding those edits into\n"
              f"  scripts/build_figure_notebook.py so they survive the next rebuild.")
        raise SystemExit(2)

# Carry over existing outputs when the SOURCES are unchanged. A rebuild is not a reason to blank
# every rendered figure: without this, running the generator on an unmodified notebook silently
# turns a fully-executed notebook (10 inline figures) into an empty one, and whoever opens it next
# sees no plots and assumes the analysis is broken.
if TARGET.exists():
    try:
        _prev = json.loads(TARGET.read_text())
        if [("".join(c["source"]), c["cell_type"]) for c in _prev["cells"]] == \
           [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]:
            for _new, _old in zip(nb["cells"], _prev["cells"]):
                if _new["cell_type"] == "code":
                    _new["outputs"] = _old.get("outputs", [])
                    _new["execution_count"] = _old.get("execution_count")
            new_text = json.dumps(nb, indent=1)
            print("sources unchanged -> preserved existing outputs")
    except Exception:
        pass

TARGET.write_text(new_text)
if _guarded:
    Path(".notebook_generator_hash").write_text(hashlib.sha256(
        json.dumps([("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]).encode()).hexdigest())
print(f"wrote climb_figures.ipynb: {len(cells)} cells "
      f"({sum(1 for c in cells if c['cell_type']=='code')} code)")
