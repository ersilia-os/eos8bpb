#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mechgate.core -- scoring engine for the five measures. SMILES in, 20 columns out.

This module only arranges the interface: the descriptor code and every measure's
algorithm are unchanged, taken verbatim from the modules under mechgate/_vendor/.
The model objects, cut-offs and rulers are read-only.

Column order follows Table 2 of article_v3.docx: 20 columns.
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# ---------------------------------------------------------------- package data
_HERE = os.path.dirname(os.path.abspath(__file__))
_VEND = os.path.join(_HERE, "_vendor")
if _VEND not in sys.path:
    sys.path.insert(0, _VEND)      # modules under _vendor import each other flatly


def resource(*parts: str) -> str:
    """Path to a packaged data file. Resolved relative to the package, never absolute."""
    try:
        from importlib.resources import files
        return str(files(__package__).joinpath(*parts))
    except Exception:
        return os.path.join(_HERE, *parts)


# ---------------------------------------------------------------- descriptors
DESCRIPTORS = ["b_rotR", "n_prim_amine_wide", "dxy_cv", "dtotal_min", "logp_ow", "slogp",
               "glob_min", "rot_rigid_rdkit", "zwit_group", "hwi", "fsp3",
               "fpsa_tpsaSP_asaH", "net_charge_v2", "zwit_charge", "n_prim_amine_entry",
               "mol_size"]

# ---------------------------------------------------------------- the five measures
# Descriptor sets, z-parameters and cut-offs are taken verbatim from the reference
# configuration and must not be edited here.
OMPF_COLS = ["mol_size", "net_charge_v2", "n_prim_amine_entry"]
OMPF_Z = {"mol_size": (291.7670444, 146.9868475, -1.0),
          "net_charge_v2": (-0.377778, 1.466116, +1.0),
          "n_prim_amine_entry": (0.488889, 1.014043, +1.0)}
MULTIPORIN_COLS = ["b_rotR", "dxy_cv", "dtotal_min", "slogp", "zwit_group",
                   "net_charge_v2", "zwit_charge", "n_prim_amine_entry", "mol_size"]
LIPID_COLS = ["b_rotR", "n_prim_amine_entry", "dxy_cv", "dtotal_min", "slogp"]
EFFLUX_COLS = ["slogp", "zwit_charge"]
ACTIVITY_COLS = list(MULTIPORIN_COLS)

CUT_OMPF = 0.95           # percentile >= 0.95 -> high
CUT_MULTIPORIN = 0.98     # percentile >= 0.98 -> high
CUT_LIPID = 0.99          # percentile >= 0.99 -> high (reverse-oriented)
CUT_ACTIVITY = 0.50       # raw score >= 0.50 -> selected (NOT percentile-driven)

_MODEL_KEY = "随机森林(depth=3, n=100, 单线程)"   # key inside the stored model object

#: The 20 delivered columns, in Table 2 order.
COLUMNS = ["set", "InChIKey", "SMILES",
           "ompf_score", "ompf_percentile", "ompf_cut",
           "multiporin_score", "multiporin_percentile", "multiporin_cut",
           "lipid_score", "lipid_percentile", "lipid_cut",
           "aminoglycoside_like", "lipopeptide_like",
           "efflux_score", "efflux_percentile",
           "activity_score", "activity_percentile", "activity_call",
           "missing_descriptors"]

#: Categorical vocabularies used in the output.
HIGH, LOW, MEDIUM = "high", "low", "medium"
NOT_COMPUTABLE = "not_computable"
SELECTED, NOT_SELECTED = "selected", "not_selected"

#: Default per-molecule timeout, in seconds. A molecule that times out is recorded
#: as not computable; it does not interrupt the molecules after it.
DEFAULT_TIMEOUT = 300.0

#: Written into missing_descriptors for a molecule that timed out.
TIMEOUT_MARK = "timeout"

# Self-promoted-uptake flag tiers: the underlying rules are unchanged, only the
# labels are rendered in the vocabulary used by the reference table.
_TIER_LABEL = {"不标": LOW, "疑似": MEDIUM, "高度疑似": HIGH}


def _norm_of(ref_sorted, s):
    """Mid-rank position of s within the sorted reference distribution."""
    N = len(ref_sorted)
    lt = np.searchsorted(ref_sorted, s, side="left")
    le = np.searchsorted(ref_sorted, s, side="right")
    return (lt + 0.5 * (le - lt)) / N


# ---------------------------------------------------------------- timeout isolation
# The only slow step is the descriptor calculation (embedding 20 conformers with
# ETKDGv3 plus MMFF94s optimisation). It runs in a single-worker subprocess; on
# timeout that worker is discarded and a fresh one is started, so the caller never
# hangs.
def _descriptor_worker(smiles: str) -> dict:
    import s37v_swap as V
    return V.work(("_", smiles, set(DESCRIPTORS)))


class _TimedDescriptors:
    """Starts one worker subprocess on demand; rebuilds it after a timeout."""

    def __init__(self):
        self._ex = None

    def _executor(self):
        if self._ex is None:
            import concurrent.futures as cf
            import multiprocessing as mp
            # fork: the child inherits already-imported _vendor modules
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
            self._ex = cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        return self._ex

    def compute(self, smiles: str, timeout: float):
        """Returns the descriptor dict, or None on timeout."""
        import concurrent.futures as cf
        fut = self._executor().submit(_descriptor_worker, smiles)
        try:
            return fut.result(timeout=timeout)
        except cf.TimeoutError:
            self.close(kill=True)
            return None

    def close(self, kill: bool = False):
        ex, self._ex = self._ex, None
        if ex is None:
            return
        try:
            if kill:
                for p in list(getattr(ex, "_processes", {}).values()):
                    try:
                        p.kill()
                    except Exception:
                        pass
            ex.shutdown(wait=not kill, cancel_futures=True)
        except TypeError:                      # cancel_futures needs py>=3.9
            ex.shutdown(wait=not kill)
        except Exception:
            pass


# ---------------------------------------------------------------- engine
class MechGate:
    """Scoring engine for the five measures. Models and rulers load once.

    >>> g = MechGate()
    >>> row = g.row("O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O")
    >>> row["activity_percentile"]
    '0.9996445854'
    """

    def __init__(self, timeout: float | None = DEFAULT_TIMEOUT):
        #: per-molecule timeout in seconds; None means no limit and no subprocess
        self.timeout = timeout
        self._timed = _TimedDescriptors()

        import joblib, __main__
        import s37u_scoredist as SU, s37v_swap as V
        # Needed for unpickling: expose the classes defined in those two modules
        # on __main__, the way the reference pipeline does.
        for m in (SU, V):
            for nm in dir(m):
                o = getattr(m, nm)
                if isinstance(o, type):
                    setattr(__main__, nm, o)
        self._V = V
        mp_ = joblib.load(resource("models", "l2_s1_9col_IL.joblib"))
        lip = joblib.load(resource("models", "oil_5col_IK.joblib"))
        eff = joblib.load(resource("models", "efflux_f3_IJ.joblib"))
        act = joblib.load(resource("models", "pen_9col_IM.joblib"))
        assert list(mp_["构件"]) == MULTIPORIN_COLS, "multi-porin descriptor set mismatch"
        assert list(lip["keep_cols"]) == LIPID_COLS, "lipid descriptor set mismatch"
        assert list(eff["构件"]) == EFFLUX_COLS, "efflux descriptor set mismatch"
        assert list(act["构件"]) == ACTIVITY_COLS, "activity descriptor set mismatch"
        self.M_multiporin = mp_["模型"][_MODEL_KEY]
        self.M_lipid = lip["pipe"]
        self.M_efflux = eff["pipe"]
        self.M_activity = act["model"]
        # Reference distributions on the fixed 73,154-molecule base set.
        # The npz holds five sorted score arrays only -- no molecule identifiers.
        z = np.load(resource("rulers", "ref_73154.npz"))
        self.REF = {"ompf": z["scaleA"], "multiporin": z["l2_s1"], "lipid": z["oil5"],
                    "efflux": z["efflux_f3"], "activity": z["activity"]}
        import s36c_ag_three_tier as AG, s36e_lipopeptide_flag as PM
        self.AG, self.PM = AG, PM
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        self.Chem = Chem

    # ------------------------------------------------------------ lifecycle
    def close(self):
        self._timed.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------ descriptors
    def descriptors(self, smiles: str, timeout: float | None = -1.0):
        """Calls work() from _vendor/s37v_swap.py directly; algorithm unchanged.

        Returns None on timeout. Pass timeout=-1.0 to use the instance's own.
        """
        if timeout == -1.0:
            timeout = self.timeout
        if timeout is None:
            return self._V.work(("_", smiles, set(DESCRIPTORS)))
        return self._timed.compute(smiles, timeout)

    # ------------------------------------------------------------ flags
    def flags(self, smiles: str, d: dict):
        """The two self-promoted-uptake flags. Rules unchanged; labels low/medium/high."""
        m = self.Chem.MolFromSmiles(smiles) or self.PM.to_mol(smiles)
        if m is None:
            return NOT_COMPUTABLE, NOT_COMPUTABLE
        a, b = bool(self.AG.crit_A(m)), bool(self.AG.crit_B(m))
        nc, ne = d.get("net_charge_v2"), d.get("n_prim_amine_entry")
        cc = None if (nc is None or ne is None) else bool(int(nc) >= 3 and int(ne) >= 3)
        nhit = sum(1 for x in (a, b, cc) if x is True)
        ag = _TIER_LABEL["疑似" if nhit >= 2 else "不标"]     # threshold: >= 2 criteria
        l1 = self.PM.crit_L1(m)[0]; l2 = self.PM.crit_L2(m)[0]; l3 = self.PM.crit_L3(m)[0]
        tier = self.PM.TIER_RULE[int(bool(l1)) + int(bool(l2)) + int(bool(l3))]
        return ag, _TIER_LABEL[tier]

    # ------------------------------------------------------------ one row, 20 columns
    def row(self, smiles: str, setname: str = "user_input",
            timeout: float | None = -1.0) -> dict:
        """Returns one molecule's 20 columns. Keys are COLUMNS; every value is a str.

        A molecule whose descriptors time out is recorded as not computable, with
        missing_descriptors set to "timeout". No exception is raised and the caller's
        remaining molecules are unaffected.
        """
        d = self.descriptors(smiles, timeout)
        if d is None:
            return self._blank_row(smiles, setname, TIMEOUT_MARK)
        m = self.Chem.MolFromSmiles(smiles)
        ik = self.Chem.MolToInchiKey(m) if m is not None else ""
        r = {"set": setname, "InChIKey": ik, "SMILES": smiles}
        r["missing_descriptors"] = ";".join(
            [c for c in MULTIPORIN_COLS if d.get(c) is None])

        def have(cols):
            return all(d.get(c) is not None for c in cols)

        # OmpF measure -- fixed formula, equal weights, no fitted coefficients
        if have(OMPF_COLS):
            s = sum(OMPF_Z[c][2] * (float(d[c]) - OMPF_Z[c][0]) / OMPF_Z[c][1]
                    for c in OMPF_COLS)
            q = float(_norm_of(self.REF["ompf"], s))
            r.update({"ompf_score": "%.10f" % s, "ompf_percentile": "%.10f" % q,
                      "ompf_cut": HIGH if q >= CUT_OMPF else LOW})
        else:
            r.update({"ompf_score": "", "ompf_percentile": "", "ompf_cut": NOT_COMPUTABLE})

        # multi-porin measure (forward-oriented)
        if have(MULTIPORIN_COLS):
            x = np.array([[float(d[c]) for c in MULTIPORIN_COLS]], float)
            s = float(np.asarray(self.M_multiporin(x), float)[0])
            q = float(_norm_of(self.REF["multiporin"], s))
            r.update({"multiporin_score": "%.10f" % s,
                      "multiporin_percentile": "%.10f" % q,
                      "multiporin_cut": HIGH if q >= CUT_MULTIPORIN else LOW})
        else:
            r.update({"multiporin_score": "", "multiporin_percentile": "",
                      "multiporin_cut": NOT_COMPUTABLE})

        # lipid route measure (reverse-oriented)
        if have(LIPID_COLS):
            x = np.array([[float(d[c]) for c in LIPID_COLS]], float)
            s = float(self.M_lipid.predict_proba(x)[0, 1])
            q = float(_norm_of(self.REF["lipid"], -s))
            r.update({"lipid_score": "%.10f" % s, "lipid_percentile": "%.10f" % q,
                      "lipid_cut": HIGH if q >= CUT_LIPID else LOW})
        else:
            r.update({"lipid_score": "", "lipid_percentile": "",
                      "lipid_cut": NOT_COMPUTABLE})

        # the two self-promoted-uptake flags (flags only, never scores)
        ag, lp = self.flags(smiles, d)
        r["aminoglycoside_like"] = ag
        r["lipopeptide_like"] = lp

        # efflux measure (reverse-oriented; NO cut column by design)
        if have(EFFLUX_COLS):
            x = np.array([[float(d[c]) for c in EFFLUX_COLS]], float)
            s = float(self.M_efflux.predict_proba(x)[0, 1])
            q = float(_norm_of(self.REF["efflux"], -s))
            r.update({"efflux_score": "%.10f" % s, "efflux_percentile": "%.10f" % q})
        else:
            r.update({"efflux_score": "", "efflux_percentile": ""})

        # activity measure (forward-oriented; the call uses the model's own
        # score threshold of 0.50 and is NOT derived from the percentile)
        if have(ACTIVITY_COLS):
            x = np.array([[float(d[c]) for c in ACTIVITY_COLS]], float)
            s = float(self.M_activity.predict_proba(x)[0, 1])
            q = float(_norm_of(self.REF["activity"], s))
            r["activity_score"] = "%.10f" % s
            r["activity_percentile"] = "%.10f" % q
            r["activity_call"] = SELECTED if s >= CUT_ACTIVITY else NOT_SELECTED
        else:
            r.update({"activity_score": "", "activity_percentile": "",
                      "activity_call": NOT_COMPUTABLE})
        return r

    # ------------------------------------------------------------ whole row unavailable
    def _blank_row(self, smiles: str, setname: str, mark: str) -> dict:
        try:
            m = self.Chem.MolFromSmiles(smiles)
            ik = self.Chem.MolToInchiKey(m) if m is not None else ""
        except Exception:
            ik = ""
        r = {c: "" for c in COLUMNS}
        r.update({"set": setname, "InChIKey": ik, "SMILES": smiles,
                  "ompf_cut": NOT_COMPUTABLE, "multiporin_cut": NOT_COMPUTABLE,
                  "lipid_cut": NOT_COMPUTABLE,
                  "aminoglycoside_like": NOT_COMPUTABLE,
                  "lipopeptide_like": NOT_COMPUTABLE,
                  "activity_call": NOT_COMPUTABLE, "missing_descriptors": mark})
        return r


# ---------------------------------------------------------------- functional API
_SHARED: "MechGate | None" = None


def _shared() -> MechGate:
    global _SHARED
    if _SHARED is None:
        _SHARED = MechGate()
    return _SHARED


def score(smiles: str, setname: str = "user_input",
          timeout: float | None = DEFAULT_TIMEOUT) -> dict:
    """Score one SMILES. Returns a dict of 20 columns (keys are ``COLUMNS``, values str).

    Models and rulers load on the first call and are reused afterwards.
    A molecule whose descriptors time out is recorded as not computable rather
    than raising.

    >>> import mechgate
    >>> r = mechgate.score("O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O")
    >>> len(r)
    20
    """
    return _shared().row(smiles, setname, timeout)


def score_many(smiles_list, setname: str = "user_input",
               timeout: float | None = DEFAULT_TIMEOUT) -> list:
    """Score a sequence of SMILES, returning a list of dicts.

    One molecule timing out does not affect the molecules after it: its row is
    recorded as not computable and scoring continues.
    """
    g = _shared()
    return [g.row(s, setname, timeout) for s in smiles_list]
