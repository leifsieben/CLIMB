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
if any("CLIMB patch-base" in l for l in lines):
    print("deepchem already patched"); raise SystemExit
# Patch EVERY `return np.asarray(features)` whose preceding ~10 lines contain the empty-array fallback
# (`features.append(np.array([]))` / "Appending empty array") — that is the ragged case strict NumPy
# (>=1.24) rejects. Matching on the fallback context (NOT just "def featurize") targets the RIGHT
# occurrence; an earlier context-only matcher patched a sibling method and left the base one failing.
patched, i = [], 0
while i < len(lines):
    if lines[i].strip() == "return np.asarray(features)":
        ctx = "\n".join(lines[max(0, i-10):i])
        if "features.append(np.array([]))" in ctx or "Appending empty array" in ctx:
            ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            lines[i:i+1] = [ind+"try:", ind+"    return np.asarray(features)", ind+"except ValueError:",
                            ind+"    return np.asarray(features, dtype=object)  # CLIMB patch-base: ragged Mol list numpy>=1.24"]
            patched.append(i+1); i += 4; continue
    i += 1
f.write_text("\n".join(lines)+"\n")
print("patched deepchem base featurize at lines:", patched or "NONE FOUND")
PY
echo "bootstrap done for $VENV"
