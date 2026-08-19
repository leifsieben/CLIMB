"""Construct molecule PAIRS that probe when an embedding can and cannot resolve two molecules.

The question: which chemical differences does a representation destroy? Each mode contributes 100
pairs, and every pair is emitted as SMILES plus a rendered image so the chemistry can be checked
BEFORE any embedding is computed -- a pair set nobody has looked at would produce confident numbers
about the wrong thing.

TWO CLASSES, and they are scored in opposite directions:

  class A -- MUST SEPARATE. The two molecules are genuinely different, so a distance of zero means
    the representation destroyed the difference. This is where the stereo bug lived: until today
    our ECFP mapped L- and D-alanine to identical vectors.

  class B -- MUST NOT SEPARATE. The two SMILES denote the SAME molecule. Any nonzero distance is a
    representation artefact rather than chemistry. Fingerprints pass these by construction; a
    sequence model need not, which makes class B the mirror image of class A rather than a
    formality.

Class B carries its own correctness check: for every pair, canonical isomeric SMILES of A and B
must be EQUAL. A "same molecule" pair that fails that is a construction bug, and the builder
refuses to emit it.
"""
from __future__ import annotations
import csv, glob, json, os, random, sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit import DataStructs
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
OUT = Path("figure_data/embedding_resolution")
OUT.mkdir(parents=True, exist_ok=True)
IMG = OUT / "inspect"; IMG.mkdir(exist_ok=True)
N_PER_MODE = 100
SEED = 0

_GEN = G.GetMorganGenerator(radius=2, fpSize=2048)


def canon(m):
    return Chem.MolToSmiles(m) if m else None


def valid(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return m if m is not None else None


def load_pool():
    """Drug-like molecules from the panels the paper actually evaluates on."""
    pool = []
    for f in sorted(glob.glob("chemeleon_suite/data/moleculeace/*.csv")):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        sc = [c for c in rows[0] if c.lower() in ("smiles", "canonical_smiles")][0]
        pool += [r[sc] for r in rows]
    try:
        import eval_v2
        for ds in ("BACE", "Tox21"):
            s, _ = eval_v2._load_moleculenet_full(ds)
            pool += list(s)
    except Exception as exc:
        print(f"  [pool] MoleculeNet unavailable ({exc}); MoleculeACE only", flush=True)
    seen, out = set(), []
    for s in pool:
        m = valid(s)
        if not m:
            continue
        c = canon(m)
        if c not in seen:
            seen.add(c); out.append(c)
    return out


# ----------------------------------------------------------------- class A: must separate ------
def mode_stereo_flip(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        cs = [a for a in m.GetAtoms() if a.GetChiralTag() in
              (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)]
        if not cs:
            continue
        a = cs[0]
        m2 = Chem.MolFromSmiles(s)
        at = m2.GetAtomWithIdx(a.GetIdx())
        at.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW
                        if a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                        else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        b = canon(m2)
        if b and b != s:
            out.append((s, b, f"inverted stereocentre at atom {a.GetIdx()}"))
    return out


def mode_ez_flip(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        bs = [b for b in m.GetBonds() if b.GetStereo() in
              (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)]
        if not bs:
            continue
        m2 = Chem.MolFromSmiles(s)
        bd = m2.GetBondWithIdx(bs[0].GetIdx())
        bd.SetStereo(Chem.BondStereo.STEREOZ if bs[0].GetStereo() == Chem.BondStereo.STEREOE
                     else Chem.BondStereo.STEREOE)
        b = canon(m2)
        if b and b != s:
            out.append((s, b, "flipped double-bond geometry E<->Z"))
    return out


def _swap_atom(s, frm, to, aromatic_only=True):
    m = Chem.MolFromSmiles(s)
    for a in m.GetAtoms():
        if a.GetSymbol() != frm or a.GetTotalNumHs() < 1:
            continue
        if aromatic_only and not a.GetIsAromatic():
            continue
        em = Chem.RWMol(m)
        em.GetAtomWithIdx(a.GetIdx()).SetAtomicNum(to)
        try:
            m2 = em.GetMol(); Chem.SanitizeMol(m2)
            return canon(m2), a.GetIdx()
        except Exception:
            continue
    return None, None


def mode_c_to_n(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        b, idx = _swap_atom(s, "C", 7)
        if b and b != s:
            out.append((s, b, f"aromatic C->N at atom {idx} (benzene -> pyridine)"))
    return out


def _attach(s, frag_atomic, label):
    m = Chem.MolFromSmiles(s)
    for a in m.GetAtoms():
        if not a.GetIsAromatic() or a.GetTotalNumHs() < 1:
            continue
        em = Chem.RWMol(m)
        idx = em.AddAtom(Chem.Atom(frag_atomic))
        em.AddBond(a.GetIdx(), idx, Chem.BondType.SINGLE)
        try:
            m2 = em.GetMol(); Chem.SanitizeMol(m2)
            return canon(m2), a.GetIdx()
        except Exception:
            continue
    return None, None


def mode_add_methyl(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        b, idx = _attach(s, 6, "methyl")
        if b and b != s:
            out.append((s, b, f"+CH3 on aromatic atom {idx}"))
    return out


def mode_add_fluorine(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        b, idx = _attach(s, 9, "fluorine")
        if b and b != s:
            out.append((s, b, f"+F on aromatic atom {idx}"))
    return out


def mode_isotope(pool, rng, n):
    """TRUE isotope substitution: 12C -> 13C on an existing atom, leaving the graph untouched.

    The first version added an explicit [2H] ATOM, which adds a node to the molecular graph -- so
    every embedding "passed" by noticing the extra atom rather than the isotope, and the mode
    measured nothing. Setting the isotope on an atom that is already there is the honest test:
    aspirin and [13CH3]-aspirin have byte-IDENTICAL ECFP4 vectors, because RDKit's Morgan atom
    invariants do not include the isotope.
    """
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        tgt = next((a for a in m.GetAtoms() if a.GetSymbol() == "C" and a.GetIsotope() == 0), None)
        if tgt is None:
            continue
        m2 = Chem.MolFromSmiles(s)
        m2.GetAtomWithIdx(tgt.GetIdx()).SetIsotope(13)
        try:
            Chem.SanitizeMol(m2); b = canon(m2)
        except Exception:
            continue
        ma, mb = valid(s), valid(b)
        if not b or b == s or ma.GetNumHeavyAtoms() != mb.GetNumHeavyAtoms():
            continue        # the graph must be IDENTICAL -- only the isotope label may differ
        out.append((s, b, f"12C -> 13C at atom {tgt.GetIdx()} (graph unchanged)"))
    return out


# Isolated saturated carbocycles only: every ring atom sp3 and in EXACTLY one ring (R1). Matching
# a bare "C1CCCCC1" also hits rings fused into a steroid or bridged scaffold, and replacing there
# does not shrink a ring -- it detonates the molecule. The first build turned a morphinan into
# "CCN([CH]Cc1ccc(O)c(O)c1)CC1CC1.O.O=C(NC1CCCC1)c1ccccc1" -- three fragments -- and labelled it
# "cyclohexyl -> cyclopentyl". Hence the post-condition below as well as the tighter pattern.
_RING6 = Chem.MolFromSmarts("[CX4;R1]1[CX4;R1][CX4;R1][CX4;R1][CX4;R1][CX4;R1]1")
_RING5 = Chem.MolFromSmarts("[CX4;R1]1[CX4;R1][CX4;R1][CX4;R1][CX4;R1]1")


def _ring_edit_ok(a, b, expect_delta):
    """A ring edit must change heavy-atom count by exactly one and must not fragment."""
    ma, mb = valid(a), valid(b)
    if not ma or not mb:
        return False
    if len(Chem.GetMolFrags(mb)) != len(Chem.GetMolFrags(ma)):
        return False
    return mb.GetNumHeavyAtoms() - ma.GetNumHeavyAtoms() == expect_delta


def mode_ring_size(pool, rng, n):
    """Cyclohexyl <-> cyclopentyl. Bit-ECFP4 maps cyclopentane and cyclohexane to the SAME vector."""
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        if m.HasSubstructMatch(_RING6):
            rep = AllChem.ReplaceSubstructs(m, _RING6, Chem.MolFromSmiles("C1CCCC1"),
                                            replaceAll=False)
            note, delta = "cyclohexyl -> cyclopentyl", -1
        elif m.HasSubstructMatch(_RING5):
            rep = AllChem.ReplaceSubstructs(m, _RING5, Chem.MolFromSmiles("C1CCCCC1"),
                                            replaceAll=False)
            note, delta = "cyclopentyl -> cyclohexyl", +1
        else:
            continue
        try:
            m2 = rep[0]; Chem.SanitizeMol(m2); b = canon(m2)
        except Exception:
            continue
        if b and b != s and _ring_edit_ok(s, b, delta):
            out.append((s, b, note))
    return out


SUBS = ["C", "CC", "Cl", "Br", "F", "O", "OC", "OCC", "N", "NC", "C(=O)O", "C(=O)N",
        "C#N", "C(F)(F)F", "S(=O)(=O)N", "[N+](=O)[O-]"]


def mode_regioisomer(pool, rng, n):
    """ortho / meta / para pairs on a disubstituted benzene.

    Built from a template rather than rewired from dataset molecules: rewiring a ring in place
    routinely produces a molecule that sanitizes but is not the intended regioisomer, and this
    mode is only meaningful if the two members differ ONLY in substitution position.
    """
    out = []
    pats = {"ortho": "c1ccccc1", "meta": "c1cccc c1".replace(" ", ""), "para": "c1ccccc1"}
    combos = [(r1, r2) for r1 in SUBS for r2 in SUBS]
    rng.shuffle(combos)
    for r1, r2 in combos:
        if len(out) >= n:
            break
        a = f"{r1}c1ccccc1{r2}"          # ortho
        b = f"{r1}c1cccc({r2})c1"        # meta
        ma, mb = valid(a), valid(b)
        if not ma or not mb:
            continue
        ca, cb = canon(ma), canon(mb)
        if ca != cb:
            out.append((ca, cb, f"ortho vs meta, substituents {r1} / {r2}"))
    return out


def _desc_vec(s):
    from descriptors_v2 import rdkit_descriptors
    import numpy as np
    v = np.asarray(rdkit_descriptors([s]), dtype=float)[0]
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def mode_matched_mw(pool, rng, n):
    """Different molecules, near-identical molecular weight and low structural similarity."""
    sub = rng.sample(pool, min(4000, len(pool)))
    rows = sorted(((s, Descriptors.MolWt(Chem.MolFromSmiles(s))) for s in sub), key=lambda x: x[1])
    out = []
    for i in range(len(rows) - 1):
        if len(out) >= n:
            break
        (sa, wa), (sb, wb) = rows[i], rows[i + 1]
        if abs(wa - wb) > 0.5:
            continue
        fa = _GEN.GetFingerprint(Chem.MolFromSmiles(sa))
        fb = _GEN.GetFingerprint(Chem.MolFromSmiles(sb))
        t = DataStructs.TanimotoSimilarity(fa, fb)
        if t < 0.30:
            out.append((sa, sb, f"different molecules, MW {wa:.2f} vs {wb:.2f}, Tanimoto {t:.2f}"))
    return out


def mode_matched_descriptors(pool, rng, n):
    """Different molecules whose 217-descriptor vectors are nearest neighbours.

    The direct control for ECFP+desc: if the descriptor block dominates, these collapse.
    """
    import numpy as np
    sub = rng.sample(pool, min(1200, len(pool)))
    X = np.stack([_desc_vec(s) for s in sub])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    out, used = [], set()
    for i, j in order:
        if len(out) >= n:
            break
        if i >= j or i in used or j in used:
            continue
        sa, sb = sub[i], sub[j]
        fa = _GEN.GetFingerprint(Chem.MolFromSmiles(sa))
        fb = _GEN.GetFingerprint(Chem.MolFromSmiles(sb))
        t = DataStructs.TanimotoSimilarity(fa, fb)
        if t >= 0.40:
            continue                       # too structurally alike to be an interesting pair
        used.add(i); used.add(j)
        out.append((sa, sb, f"nearest neighbours in 217-descriptor space, "
                            f"z-distance {float(np.sqrt(D[i, j])):.2f}, Tanimoto {t:.2f}"))
    return out


# ------------------------------------------------------------- class B: must NOT separate ------
def mode_smiles_enumeration(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        rnd = Chem.MolToSmiles(m, canonical=False, doRandom=True)
        if valid(rnd) and rnd != s:
            out.append((s, rnd, "same molecule, canonical vs randomly written SMILES"))
    return out


def mode_kekule(pool, rng, n):
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        if not any(a.GetIsAromatic() for a in m.GetAtoms()):
            continue
        mk = Chem.Mol(m)
        try:
            Chem.Kekulize(mk, clearAromaticFlags=True)
            k = Chem.MolToSmiles(mk, kekuleSmiles=True)
        except Exception:
            continue
        if valid(k) and k != s:
            out.append((s, k, "same molecule, aromatic vs Kekule SMILES"))
    return out


def mode_symmetry_equivalent(pool, rng, n):
    """Substitute a methyl at two TOPOLOGICALLY EQUIVALENT positions -> literally one molecule.

    Equivalence comes from CanonicalRankAtoms(breakTies=False): equal rank means the two positions
    are indistinguishable, so the two products must have identical canonical SMILES. The emitter
    asserts that, which makes this mode self-validating.
    """
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
        by_rank = {}
        for a in m.GetAtoms():
            if a.GetIsAromatic() and a.GetTotalNumHs() >= 1:
                by_rank.setdefault(ranks[a.GetIdx()], []).append(a.GetIdx())
        eq = next((v for v in by_rank.values() if len(v) >= 2), None)
        if not eq:
            continue
        prods = []
        for idx in eq[:2]:
            em = Chem.RWMol(m)
            c = em.AddAtom(Chem.Atom(6))
            em.AddBond(idx, c, Chem.BondType.SINGLE)
            try:
                m2 = em.GetMol(); Chem.SanitizeMol(m2); prods.append(canon(m2))
            except Exception:
                pass
        if len(prods) == 2 and prods[0] == prods[1]:
            out.append((prods[0], prods[1], f"+CH3 at equivalent positions {eq[0]} / {eq[1]}"))
    return out


def mode_salt_form(pool, rng, n):
    """DROPPED, and kept here with the reason.

    A free base and its HCl salt are the same ACTIVE SPECIES but not the same molecular graph, so
    every pair failed class B's own equality check (100/100 rejected on the first build). The check
    was right and the mode was mis-classified: "should be very close" is a graded question, not the
    binary must-not-separate one this harness scores. Re-admit it only with a graded metric.
    """
    out = []
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        if not any(a.GetSymbol() == "N" and a.GetTotalNumHs() >= 1 and not a.GetIsAromatic()
                   for a in m.GetAtoms()):
            continue
        salt = s + ".Cl"
        if valid(salt):
            out.append((s, salt, "free base vs HCl salt (same active species)"))
    return out


MODES = [
    ("A", "stereo_flip",          mode_stereo_flip,          "inverted stereocentre"),
    ("A", "ez_flip",              mode_ez_flip,              "flipped E/Z double bond"),
    ("A", "c_to_n",               mode_c_to_n,               "aromatic C -> N"),
    ("A", "add_methyl",           mode_add_methyl,           "+ one methyl"),
    ("A", "add_fluorine",         mode_add_fluorine,         "+ one fluorine"),
    ("A", "isotope_13c",          mode_isotope,              "12C -> 13C, graph unchanged"),
    ("A", "ring_size",            mode_ring_size,            "cyclopentyl <-> cyclohexyl"),
    ("A", "regioisomer",          mode_regioisomer,          "ortho vs meta"),
    ("A", "matched_mw",           mode_matched_mw,           "different molecule, same MW"),
    ("A", "matched_descriptors",  mode_matched_descriptors,  "different molecule, same 217 descriptors"),
    ("B", "smiles_enumeration",   mode_smiles_enumeration,   "SAME molecule, re-written SMILES"),
    ("B", "kekule",               mode_kekule,               "SAME molecule, Kekule form"),
    ("B", "symmetry_equivalent",  mode_symmetry_equivalent,  "SAME molecule, equivalent positions"),
]


def main() -> int:
    rng = random.Random(SEED)
    pool = load_pool()
    rng.shuffle(pool)
    print(f"pool: {len(pool)} unique molecules\n")

    rows, summary = [], []
    for cls, name, fn, blurb in MODES:
        got = fn(pool, random.Random(SEED), N_PER_MODE)
        bad = 0
        if cls == "B":
            kept = []
            for a, b, note in got:
                ca, cb = canon(valid(a)), canon(valid(b))
                if ca == cb:
                    kept.append((a, b, note))
                else:
                    bad += 1
            got = kept
        for i, (a, b, note) in enumerate(got[:N_PER_MODE]):
            rows.append(dict(mode=name, klass=cls, pair_id=f"{name}_{i:03d}",
                             smiles_a=a, smiles_b=b, note=note))
        flag = "" if len(got) >= N_PER_MODE else f"   <-- ONLY {len(got)}"
        extra = f"  ({bad} rejected: not the same molecule)" if bad else ""
        print(f"  {cls}  {name:22} {min(len(got), N_PER_MODE):3}/{N_PER_MODE}  {blurb}{extra}{flag}")
        summary.append(dict(klass=cls, mode=name, n=min(len(got), N_PER_MODE),
                            rejected=bad, description=blurb))

    with (OUT / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "klass", "pair_id", "smiles_a", "smiles_b", "note"])
        w.writeheader(); w.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT/'pairs.csv'}: {len(rows)} pairs across {len(MODES)} modes")

    # ---- visual inspection ---------------------------------------------------------------
    # B is drawn on A's COORDINATES via their maximum common substructure, and whatever is not in
    # that core is highlighted. Laying the two out independently -- the obvious way, and the first
    # thing I did -- gives two unrelated-looking pictures for a one-atom change, so the reviewer
    # cannot see the edit and the image is worse than no image at all.
    for cls, name, _, blurb in MODES:
        sel = [r for r in rows if r["mode"] == name][:6]
        if not sel:
            continue
        ms, legends, highlights = [], [], []
        for r in sel:
            a, b = Chem.MolFromSmiles(r["smiles_a"]), Chem.MolFromSmiles(r["smiles_b"])
            AllChem.Compute2DCoords(a)
            try:
                mcs = rdFMCS.FindMCS([a, b], timeout=5, matchValences=False,
                                     ringMatchesRingOnly=True, completeRingsOnly=False)
                core = Chem.MolFromSmarts(mcs.smartsString)
                AllChem.GenerateDepictionMatching2DStructure(b, a, refPatt=core, acceptFailure=True)
                ha = [i for i in range(a.GetNumAtoms()) if i not in set(a.GetSubstructMatch(core))]
                hb = [i for i in range(b.GetNumAtoms()) if i not in set(b.GetSubstructMatch(core))]
            except Exception:
                AllChem.Compute2DCoords(b); ha, hb = [], []
            ms += [a, b]
            highlights += [ha, hb]
            legends += [f"{r['pair_id']}  A", f"B  {r['note'][:44]}"]
        img = Draw.MolsToGridImage(ms, molsPerRow=2, subImgSize=(400, 300), legends=legends,
                                   highlightAtomLists=highlights)
        p = IMG / f"{cls}_{name}.png"
        if hasattr(img, "save"):
            img.save(str(p))
        else:
            p.write_bytes(img.data)
        print(f"  inspect -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
