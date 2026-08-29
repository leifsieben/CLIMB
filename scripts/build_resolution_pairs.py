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


# Monovalent substituents for the regioisomer template. EXPANDED from 16 to 45 (2026-08-25):
# 16 gives 256 ordered combinations, which cannot reach N_PER_MODE=1000 pairs.
#
# The template attaches r1 as a PREFIX and r2 as a SUFFIX, so an asymmetric group attaches through
# its last atom in one slot and its first atom in the other -- (r1, r2) and (r2, r1) are therefore
# usually DIFFERENT molecules, not a redundant ordering. Both slots use the same convention in the
# ortho and the meta string, so each emitted pair is still a genuine ortho/meta pair. Symmetric
# groups (C, F, Cl, ...) do collapse under the swap, which is why the builder deduplicates on the
# canonical pair rather than on the substituent set.
SUBS = ["C", "CC", "CCC", "C(C)C", "C(C)(C)C", "C=C", "C#C",
        "Cl", "Br", "F", "I",
        "O", "OC", "OCC", "OCCC", "OC(F)(F)F", "CO", "CCO",
        "N", "NC", "NCC", "N(C)C", "CN", "CCN",
        "C(=O)O", "C(=O)N", "C(=O)C", "C(=O)CC", "C(=O)OC", "C(=O)OCC", "NC(=O)C",
        "C#N", "CC#N", "C(F)(F)F", "C(F)F", "CF", "CCl", "CBr",
        "S", "SC", "S(=O)(=O)N", "S(=O)(=O)C",
        "CC(=O)O", "OCC(F)(F)F", "c1ccccc1"]


def mode_ch2_homologue(pool, rng, n):
    """One CH2 INSERTED into an acyclic C-C single bond: the classic homologous series.

    Replaces matched_mw (Leif 2026-08-28). matched_mw asked "are these two unrelated molecules the
    A or the B of their pair", which has no answer -- the label is arbitrary, so every arm scored
    ~0.5 whatever it could see, and under a [0.5, 1] axis an empty bar there would have meant
    "ill-posed" while an empty bar everywhere else means "not resolved". One bar, two meanings.
    It survives as a reported harness check rather than a panel.

    Homologation is the natural replacement: DIRECTIONAL by construction (A is always the shorter
    homologue), chemically real, and the edit medicinal chemists actually walk. It is also a
    sharper question than add_methyl -- a methyl BRANCHES the skeleton, where a CH2 insertion
    extends it and leaves every functional group intact, so a representation can respond to one
    and not the other.

    Insertion, not extension at a terminus: a terminal methyl-to-ethyl is a special case of
    add_methyl, and would measure the same thing twice.
    """
    out, seen = [], set()
    for s in pool:
        if len(out) >= n:
            break
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        # Acyclic single bond between two ALIPHATIC carbons. Aromatic or ring bonds cannot take an
        # insertion without changing ring size, which is a different mode on this same plate.
        cands = [b for b in m.GetBonds()
                 if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
                 and b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6
                 and not b.GetBeginAtom().GetIsAromatic()
                 and not b.GetEndAtom().GetIsAromatic()]
        if not cands:
            continue
        # Chosen at random, not cands[0]: the first acyclic C-C bond is almost always the same
        # terminal alkyl position, so a fixed choice would homologate one structural context 1,000
        # times and measure that context rather than homologation.
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


def mode_regioisomer(pool, rng, n):
    """PARA vs META pairs on a disubstituted benzene.

    Built from a template rather than rewired from dataset molecules: rewiring a ring in place
    routinely produces a molecule that sanitizes but is not the intended regioisomer, and this
    mode is only meaningful if the two members differ ONLY in substitution position.

    PARA/META, NOT ORTHO/META (Leif 2026-08-28: "we do just change the structure right, not
    change the SMILES anymore than is needed"). Canonical SMILES for the three isomers of one
    molecule:

        ortho   Cc1ccccc1Br      <- second substituent is TERMINAL, no branch at all
        meta    Cc1cccc(Br)c1    <- branch
        para    Cc1ccc(Br)cc1    <- branch, one ring position over

    Ortho differs from the other two in WHETHER THE STRING HAS A BRANCH, which is a gross
    topological difference available to any tokenizer for free -- and it showed: under the
    separability metric ortho-vs-meta read 1.000 for all seven arms, the untrained random encoder
    included, so the panel measured the template rather than the chemistry. Para vs meta share
    their token inventory and differ only in where the branch sits, which is the actual question:
    is substitution POSITION encoded.

    Direction is fixed (A is always para) so the label means the same thing in every pair, which
    the separability metric requires.
    """
    out, seen = [], set()
    combos = [(r1, r2) for r1 in SUBS for r2 in SUBS]
    rng.shuffle(combos)
    for r1, r2 in combos:
        if len(out) >= n:
            break
        a = f"{r1}c1ccc({r2})cc1"        # para
        b = f"{r1}c1cccc({r2})c1"        # meta
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
    from descriptors_v2 import rdkit_descriptors
    import numpy as np
    v = np.asarray(rdkit_descriptors([s]), dtype=float)[0]
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def mode_matched_mw(pool, rng, n):
    """Different molecules, near-identical molecular weight and low structural similarity."""
    sub = rng.sample(pool, min(20000, len(pool)))
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
    sub = rng.sample(pool, min(8000, len(pool)))
    X = np.stack([_desc_vec(s) for s in sub])
    # Ipc overflows float32 for ~0.6% of drug-like molecules and reaches 1.2e32 when it does not
    # -- 28 orders of magnitude above every other descriptor -- so a plain mean/std leaves that
    # one column carrying the entire distance and the rest at z ~ 0. Standardize on the finite
    # entries and clip, the same convention descriptors_v2.normalize() uses (clip=10).
    X = np.where(np.isfinite(X), X, np.nan)
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Z = np.clip((X - mu) / sd, -10.0, 10.0)
    Z = np.nan_to_num(Z, nan=0.0)
    # Nearest neighbour per row, computed in chunks. The old form materialized an (n, n, 217)
    # difference tensor -- 28 GB at n=8000 -- so the previous n=1200 was a memory ceiling rather
    # than a choice, and the `used` set halved it again to at most 600 pairs.
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
            continue                       # too structurally alike to be an interesting pair
        used.add(i); used.add(j)
        out.append((sa, sb, f"nearest neighbours in 217-descriptor space, "
                            f"z-distance {float(np.sqrt(max(dist, 0.0))):.2f}, Tanimoto {t:.2f}"))
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
    ("A", "regioisomer",          mode_regioisomer,          "para vs meta"),
    ("A", "ch2_homologue",        mode_ch2_homologue,        "+ one CH2 into a C-C bond"),
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
        # Ask for a margin and DEDUPLICATE, rather than taking the first N_PER_MODE and hoping.
        # ez_flip and ring_size both repeat pairs at n=1000 (1 and 31 respectively) because they
        # scan the pool for a matching motif and two different source molecules can canonicalize
        # to the same edited pair. A repeated pair inflates n without adding information and
        # narrows the reported IQR, so it is a silent precision claim.
        got = fn(pool, random.Random(SEED), int(N_PER_MODE * 1.30))
        seen_pairs, uniq_got = set(), []
        for a, b, note in got:
            k = tuple(sorted((a, b)))
            if k in seen_pairs:
                continue
            seen_pairs.add(k); uniq_got.append((a, b, note))
        dropped = len(got) - len(uniq_got)
        got = uniq_got
        # Both classes are checked through a SMILES ROUND-TRIP, not on the mol objects the edit
        # was applied to. Setting a chiral tag or a bond stereo directly bypasses RDKit's stereo
        # perception, so an "inverted stereocentre" on a symmetric ring can be a no-op that only
        # re-parsing reveals: stereo_flip on hexachlorocyclohexane produced a class-A pair whose
        # two members are the same molecule, which is a class-B pair wearing a class-A label and
        # would have been scored as a chemical response the CLMs failed to make.
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

    # The CANONICAL twin, emitted here rather than by hand. Class A is a chemistry question and is
    # measured on canonical strings; class B IS the notation question and is measured as written.
    # Canonicalizing class B collapses it to a no-op by construction -- that is the point of having
    # both files, and it is why the two classes must never be read from the same one.
    # (figures/fig_G.py asserts exactly that, per class, against the `input` column.)
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
