"""Headlessly execute the figure notebook's code cells (matplotlib Agg) up to a given
index, so figures render to figures_out/ without a Jupyter kernel. Run from repo root with
.venv_sanity python. Usage: render_figs.py [max_cell_index]  (default 7 = through Fig A1)."""
import matplotlib; matplotlib.use("Agg")
import json, sys, os
from pathlib import Path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nb = json.load(open("climb_figures.ipynb"))
maxcell = int(sys.argv[1]) if len(sys.argv) > 1 else 7
ns = {"Path": Path}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or i > maxcell:
        continue
    try:
        exec(compile(''.join(c["source"]), f"cell{i}", "exec"), ns)
    except Exception:
        import traceback; print(f"CELL {i} ERROR:"); traceback.print_exc(); sys.exit(1)
print(f"[render] done through cell {maxcell}")
