from __future__ import annotations
import csv, glob, json, os, random, sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
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
N_PER_MODE = 1000
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
    for f in sorted(glob.glob("benchmarks/data/moleculeace/*.csv")):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        sc = [c for c in rows[0] if c.lower() in ("smiles", "canonical_smiles")][0]
        pool += [r[sc] for r in rows]
    try:
        import evaluate
        for ds in ("BACE", "Tox21"):
            s, _ = evaluate._load_moleculenet_full(ds)
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
            continue
        out.append((s, b, f"12C -> 13C at atom {tgt.GetIdx()} (graph unchanged)"))
    return out


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


SUBS = ["C", "CC", "CCC", "C(C)C", "C(C)(C)C", "C=C", "C#C",
        "Cl", "Br", "F", "I",
        "O", "OC", "OCC", "OCCC", "OC(F)(F)F", "CO", "CCO",
        "N", "NC", "NCC", "N(C)C", "CN", "CCN",
        "C(=O)O", "C(=O)N", "C(=O)C", "C(=O)CC", "C(=O)OC", "C(=O)OCC", "NC(=O)C",
        "C#N", "CC#N", "C(F)(F)F", "C(F)F", "CF", "CCl", "CBr",
        "S", "SC", "S(=O)(=O)N", "S(=O)(=O)C",
        "CC(=O)O", "OCC(F)(F)F", "c1ccccc1"]


def mode_ch2_homologue(pool, rng, n):

    out, seen = [], set()
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue


        cands = [b for b in m.GetBonds()
                 if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
                 and b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6
                 and not b.GetBeginAtom().GetIsAromatic()
                 and not b.GetEndAtom().GetIsAromatic()]
        if not cands:
            continue


        bond = rng.choice(cands)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rw = Chem.RWMol(m)
        rw.RemoveBond(i, j)
        k = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(i, k, Chem.BondType.SINGLE)
        rw.AddBond(k, j, Chem.BondType.SINGLE)
        try:
            prod = rw.GetMol()
            Chem.SanitizeMol(prod)
        except Exception:
            continue
        a, b = canon(m), canon(prod)
        if not a or not b or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, f"CH2 inserted into acyclic C-C bond {i}-{j}"))
    return out


def mode_halogen_swap(pool, rng, n):

    out, seen = [], set()
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        cls = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 17]
        if not cls:
            continue


        idx = rng.choice(cls)
        rw = Chem.RWMol(m)
        rw.GetAtomWithIdx(idx).SetAtomicNum(9)
        try:
            prod = rw.GetMol()
            Chem.SanitizeMol(prod)
        except Exception:
            continue
        a, b = canon(m), canon(prod)
        if not a or not b or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, f"Cl -> F at atom {idx}"))
    return out


def mode_regioisomer(pool, rng, n):

    out, seen = [], set()
    combos = [(r1, r2) for r1 in SUBS for r2 in SUBS]
    rng.shuffle(combos)
    for r1, r2 in combos:
        if len(out) >= n:
            break
        a = f"{r1}c1ccc({r2})cc1"
        b = f"{r1}c1cccc({r2})c1"
        ma, mb = valid(a), valid(b)
        if not ma or not mb:
            continue
        ca, cb = canon(ma), canon(mb)
        key = tuple(sorted((ca, cb)))
        if ca != cb and key not in seen:
            seen.add(key)
            out.append((ca, cb, f"para vs meta, substituents {r1} / {r2}"))
    return out


def _desc_vec(s):
    from descriptors import rdkit_descriptors
    import numpy as np
    v = np.asarray(rdkit_descriptors([s]), dtype=float)[0]
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def mode_matched_descriptors(pool, rng, n):

    import numpy as np
    sub = rng.sample(pool, min(8000, len(pool)))
    X = np.stack([_desc_vec(s) for s in sub])


    X = np.where(np.isfinite(X), X, np.nan)
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Z = np.clip((X - mu) / sd, -10.0, 10.0)
    Z = np.nan_to_num(Z, nan=0.0)


    sq = (Z ** 2).sum(1)
    cand = []
    for lo in range(0, len(Z), 512):
        blk = Z[lo:lo + 512]
        d = sq[lo:lo + 512, None] + sq[None, :] - 2.0 * (blk @ Z.T)
        for k in range(len(blk)):
            d[k, lo + k] = np.inf
        j = np.argmin(d, axis=1)
        cand += [(float(d[k, j[k]]), lo + k, int(j[k])) for k in range(len(blk))]
    cand.sort()
    out, used = [], set()
    for dist, i, j in cand:
        if len(out) >= n:
            break
        if i in used or j in used:
            continue
        sa, sb = sub[i], sub[j]
        fa = _GEN.GetFingerprint(Chem.MolFromSmiles(sa))
        fb = _GEN.GetFingerprint(Chem.MolFromSmiles(sb))
        t = DataStructs.TanimotoSimilarity(fa, fb)
        if t >= 0.40:
            continue
        used.add(i); used.add(j)
        out.append((sa, sb, f"nearest neighbours in 217-descriptor space, "
                            f"z-distance {float(np.sqrt(max(dist, 0.0))):.2f}, Tanimoto {t:.2f}"))
    return out


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


MODES = [
    ("A", "stereo_flip",          mode_stereo_flip,          "inverted stereocentre"),
    ("A", "ez_flip",              mode_ez_flip,              "flipped E/Z double bond"),
    ("A", "c_to_n",               mode_c_to_n,               "aromatic C -> N"),
    ("A", "add_methyl",           mode_add_methyl,           "+ one methyl"),
    ("A", "add_fluorine",         mode_add_fluorine,         "+ one fluorine"),
    ("A", "isotope_13c",          mode_isotope,              "12C -> 13C, graph unchanged"),
    ("A", "ring_size",            mode_ring_size,            "cyclopentyl <-> cyclohexyl"),
    ("A", "regioisomer",          mode_regioisomer,          "para vs meta"),
    ("A", "ch2_homologue",        mode_ch2_homologue,        "+ one CH2 into a C-C bond"),
    ("A", "halogen_swap",         mode_halogen_swap,         "Cl -> F, skeleton unchanged"),
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


        got = fn(pool, random.Random(SEED), int(N_PER_MODE * 1.30))
        seen_pairs, uniq_got = set(), []
        for a, b, note in got:
            k = tuple(sorted((a, b)))
            if k in seen_pairs:
                continue
            seen_pairs.add(k); uniq_got.append((a, b, note))
        dropped = len(got) - len(uniq_got)
        got = uniq_got


        bad = 0
        kept = []
        for a, b, note in got:
            ca, cb = canon(valid(a)), canon(valid(b))
            same = (ca is not None and ca == cb)
            if same == (cls == "B"):
                kept.append((a, b, note))
            else:
                bad += 1
        got = kept
        for i, (a, b, note) in enumerate(got[:N_PER_MODE]):
            rows.append(dict(mode=name, klass=cls, pair_id=f"{name}_{i:03d}",
                             smiles_a=a, smiles_b=b, note=note))
        flag = "" if len(got) >= N_PER_MODE else f"   <-- ONLY {len(got)}"
        _why = "not the same molecule" if cls == "B" else "SAME molecule after round-trip"
        extra = f"  ({bad} rejected: {_why})" if bad else ""
        extra += f"  ({dropped} duplicate pairs dropped)" if dropped else ""
        print(f"  {cls}  {name:22} {min(len(got), N_PER_MODE):3}/{N_PER_MODE}  {blurb}{extra}{flag}")
        summary.append(dict(klass=cls, mode=name, n=min(len(got), N_PER_MODE),
                            rejected=bad, description=blurb))

    short, dupes = {}, {}
    for cls, name, _, _ in MODES:
        got = [r for r in rows if r["mode"] == name]
        uniq = {tuple(sorted((r["smiles_a"], r["smiles_b"]))) for r in got}
        if len(uniq) != len(got):
            dupes[name] = len(got) - len(uniq)
        if len(got) < N_PER_MODE:
            short[name] = len(got)
    assert not dupes, f"duplicate pairs survived the dedup: {dupes}"
    assert not short, (
        f"modes below N_PER_MODE={N_PER_MODE} after dedup: {short}. Reported n is per-mode, so "
        f"this is legal but must be stated in the caption rather than silently averaged over.")

    with (OUT / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "klass", "pair_id", "smiles_a", "smiles_b", "note"])
        w.writeheader(); w.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT/'pairs.csv'}: {len(rows)} pairs across {len(MODES)} modes")


    can = []
    for r in rows:
        ca, cb = canon(valid(r["smiles_a"])), canon(valid(r["smiles_b"]))
        if not ca or not cb:
            continue
        can.append({**r, "smiles_a": ca, "smiles_b": cb})
    assert len(can) == len(rows), f"canonicalization lost {len(rows) - len(can)} pairs"
    with (OUT / "pairs_canonical.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "klass", "pair_id", "smiles_a", "smiles_b", "note"])
        w.writeheader(); w.writerows(can)
    collapsed = [r for r in can if r["smiles_a"] == r["smiles_b"]]
    stray = [r["pair_id"] for r in collapsed if r["klass"] == "A"]
    assert not stray, (
        f"class-A pairs collapse to one molecule under canonicalization: {stray[:5]}. A class-A "
        f"pair MUST be two different molecules -- a collapsed one is a class-B pair mislabelled, "
        f"and every representation would be scored for failing to respond to nothing.")
    n_b = sum(1 for r in can if r["klass"] == "B")
    assert len(collapsed) == n_b, (
        f"{len(collapsed)} pairs collapse but {n_b} are class B; class B is the SAME molecule "
        f"written two ways, so all of them must collapse and nothing else may.")
    print(f"wrote {OUT/'pairs_canonical.csv'}: {len(can)} pairs "
          f"({len(collapsed)} collapse to identical strings = every class-B pair, as required)")


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
