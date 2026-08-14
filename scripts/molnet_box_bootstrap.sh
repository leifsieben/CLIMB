#!/usr/bin/env bash
# Reproducible environment fix for running the CheMeleon MoleculeNet arms (frozen + e2e) in the
# py3.12 `chemeleon` venv (chemprop 2.3.1 + deepchem 2.5.0). deepchem needs TensorFlow to import and
# its RawFeaturizer only tolerates RDKit-unparseable molecules under numpy<1.24 — but py3.12 has no
# numpy<1.24 wheel. So we pin a coherent numpy<2 stack (tf 2.16 is the last that pins numpy<2 AND
# supports py3.12) and patch deepchem's one ragged-array line to fall back to an object array (its
# pre-1.24 behavior), which lets its own valid_inds drop the failed molecules. Idempotent.
set -eu
VENV="${1:-$HOME/venvs/chemeleon}"
"$VENV/bin/pip" install -q "tensorflow-cpu==2.16.2" "numpy>=1.23,<2" "scipy<1.14" "ml-dtypes<0.4"
"$VENV/bin/python" - <<'PY'
import deepchem, pathlib
f = pathlib.Path(deepchem.feat.base_classes.__file__)
lines = f.read_text().splitlines()
if any("CLIMB patch289" in l for l in lines):
    print("deepchem already patched"); raise SystemExit
# patch the base Featurizer.featurize `return np.asarray(features)` (ragged Mol list under numpy>=1.24)
for i, l in enumerate(lines):
    if l.strip() == "return np.asarray(features)":
        # confirm this is the base featurize (nearest preceding def is `def featurize(`)
        for j in range(i, -1, -1):
            if lines[j].lstrip().startswith("def "):
                base = "def featurize(" in lines[j]; break
        if not base:
            continue
        ind = l[:len(l) - len(l.lstrip())]
        lines[i:i+1] = [ind+"try:", ind+"    return np.asarray(features)",
                        ind+"except ValueError:",
                        ind+"    return np.asarray(features, dtype=object)  # CLIMB patch289: ragged Mol list numpy>=1.24"]
        f.write_text("\n".join(lines)+"\n"); print("patched", f); break
else:
    print("PATTERN NOT FOUND")
PY
echo "bootstrap done for $VENV"
