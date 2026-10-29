#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s37v_swap.py -- S37-25 实验 B · ★★角色互换

★★训练：正类 = El Zahed 663 ／ 负类 = Ritz 右档 3,503
★★验证：CO-ADD 左档（★训练时完全没见过 —— ★件〇 实测交集 0/0）
★★弱阴性对照：El Zahed『野生型 EC50>50』3,747 ⇒ ★★只打分看分布，★不进训练、★不当标签

★★★不改任何既有秤、★不出对外 verdict、★不裁任何未定项。
★NREP = 5（★承 Lang 定的『探索阶段 NREP=5』）；★其余口径照搬 [S37-24]：
   RepeatedStratifiedKFold(5 折 × NREP 次, seed=42)；★超参不变。
⚠️★★★内部 OOF AUC【不得与 0.6727/0.8756/0.8923 相比】—— ★训练集完全换了，★不是同一把尺。
★切点三套（甲-1/甲-2/乙）★★全部在【新训练集】的 OOF 上定线。

⚠️ 两条禁读：cooper_features.csv / cooper_casecontrol_* —— guard 硬拦。
"""
from __future__ import annotations
import os, tempfile, sys, csv, time, json, hashlib, datetime, argparse, multiprocessing as mp
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 原为开发机的 antibiotic-benchmark/scripts，会抢在本包 _vendor 之前；改为本模块自身所在目录，同族模块照常平铺 import
DC = os.path.join(HOME, "antibiotic-benchmark", "data_curated")
S35 = os.path.join(HOME, "downloads_raw", "section35")
S37 = os.path.join(HOME, "downloads_raw", "section37")
S37_OUT = S37
SCRATCH = os.environ.get("MECHGATE_SCRATCH", os.path.join(tempfile.gettempdir(), "mechgate_scratch"))  # 原为开发机的绝对路径，已去标识化；仅训练/空转代码用到，推理路径不碰
FORBIDDEN = ("cooper_features.csv", "cooper_casecontrol_")
SEED, NREP, NSPLIT, THR = 42, 5, 5, 43.3093
NPROC = 16
LOG, OUT = [], []
OIL12 = ["b_rotR", "n_prim_amine_wide", "dxy_cv", "dtotal_min", "logp_ow", "slogp",
         "glob_min", "rot_rigid_rdkit", "zwit_group", "hwi", "fsp3", "fpsa_tpsaSP_asaH"]
WAT7 = ["min_proj_area", "net_charge_74", "n_prim_amine",
        "net_charge_v2", "zwit_charge", "n_prim_amine_entry", "mol_size"]
ALL19 = OIL12 + WAT7
W3 = ["min_proj_area", "net_charge_74", "n_prim_amine"]
UNION16 = OIL12 + ["net_charge_v2", "zwit_charge", "n_prim_amine_entry", "mol_size"]
SETS = [("甲 水路3", W3), ("乙 油路12", OIL12), ("丙 合15", W3 + OIL12), ("丁 并集16", UNION16)]


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def step(*a):
    s = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(s, flush=True); LOG.append(s)


def guard(p):
    if any(f in os.path.basename(p) for f in FORBIDDEN):
        raise RuntimeError("★★禁读: %s" % p)
    return p


def md5(fp):
    h = hashlib.md5()
    with open(guard(fp), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p, hdr, sep="\t"):
    guard(p)
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        L = [l.rstrip("\r") for l in fh.read().split("\n") if l.strip()]
    i = next(k for k, l in enumerate(L) if l.startswith(hdr))
    return list(csv.DictReader(L[i:], delimiter=sep))


def auc_np(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, int); s = np.asarray(s, float)
    m = np.isfinite(s); y, s = y[m], s[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cut_youden(score, y):
    s = np.asarray(score, float); y = np.asarray(y, int)
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return None
    o = np.argsort(-s, kind="mergesort"); ys = y[o]; ss = s[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]
    J = tp[last] / P - fp[last] / N
    return float(ss[last][int(np.argmax(J))])


def q5(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, min=np.nan, p25=np.nan, med=np.nan, p75=np.nan, max=np.nan)
    return dict(n=len(v), min=v.min(), p25=np.percentile(v, 25), med=np.median(v),
                p75=np.percentile(v, 75), max=v.max())


# ══════ 可 pickle 打分器（★承 [S37-24] 的修法）══════
class EqWeight:
    def __init__(self, mu, sd, sg):
        self.mu, self.sd, self.sg = mu, sd, sg

    def __call__(self, X):
        return (((X - self.mu) / self.sd) * self.sg).sum(1)


class ProbaScorer:
    def __init__(self, model, scaler=None):
        self.model, self.scaler = model, scaler

    def __call__(self, X):
        Z = self.scaler.transform(X) if self.scaler is not None else X
        return self.model.predict_proba(Z)[:, 1]


def fitters():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    def f_eq(Xtr, ytr):
        mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1); sd = np.where(sd == 0, 1.0, sd)
        sg = np.array([1.0 if Xtr[ytr == 1, k].mean() >= Xtr[ytr == 0, k].mean() else -1.0
                       for k in range(Xtr.shape[1])])
        return EqWeight(mu, sd, sg)

    def f_lr(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        return ProbaScorer(m, sc)

    def f_rf(Xtr, ytr):
        return ProbaScorer(RandomForestClassifier(n_estimators=100, max_depth=3,
                                                  random_state=SEED, n_jobs=1).fit(Xtr, ytr))

    def f_xgb(Xtr, ytr):
        return ProbaScorer(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                         random_state=SEED, n_jobs=1,
                                         eval_metric="logloss").fit(Xtr, ytr))
    return [("等权合分(方向统一)", f_eq), ("逻辑回归(L2, C=1)", f_lr),
            ("随机森林(depth=3, n=100, 单线程)", f_rf),
            ("XGBoost(depth=3, n=200, lr=0.1, 单线程)", f_xgb)]


# ══════ 构件现算（★worker，★口径与 [S37-23/24] 逐字相同）══════
_C = {}


def work(job):
    if not _C:
        from _smiles_syntax import normalize_smiles_syntax
        from a2_min_projection_area_v2 import protonate_ph74
        from a2_min_projection_area import min_projection_area
        from penetration_features import (b_rotR, logp_ow, net_charge_pH74,
                                          net_charge_pH74_v2, mol_size)
        from efflux_features import slogp
        from _water_candidates import (conformer_ensemble, dipole_and_globularity,
                                       rot_bond_counts, zwitterion_flags)
        import s31m_water_extra as WX
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        _C.update(norm=normalize_smiles_syntax, ph74=protonate_ph74, mpa=min_projection_area,
                  b_rotR=b_rotR, logp_ow=logp_ow, nc=net_charge_pH74, nc2=net_charge_pH74_v2,
                  msz=mol_size, slogp=slogp, ens=conformer_ensemble, dg=dipole_and_globularity,
                  rot=rot_bond_counts, zw=zwitterion_flags, WX=WX, Chem=Chem,
                  WIDE=Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(N=*)]"),
                  ENTRY=Chem.MolFromSmarts("[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]"))
    key, s, need = job
    d = {"inchikey_full": key}
    C = _C

    def g(f, *a, **kw):
        try:
            return f(*a, **kw)
        except Exception:
            return None
    nrm = g(C["norm"], s) or s
    ph = g(C["ph74"], nrm)
    s3 = ph or nrm
    m = g(C["Chem"].MolFromSmiles, nrm)
    if "min_proj_area" in need:
        d["min_proj_area"] = g(C["mpa"], nrm, 20, 42)
    if "net_charge_74" in need:
        d["net_charge_74"] = g(C["nc"], nrm)
    if "net_charge_v2" in need:
        v = g(C["nc2"], nrm); d["net_charge_v2"] = v[0] if isinstance(v, tuple) else v
    if "n_prim_amine" in need or "n_prim_amine_wide" in need:
        v = None if m is None else len(m.GetSubstructMatches(C["WIDE"]))
        d["n_prim_amine"] = v; d["n_prim_amine_wide"] = v
    if "n_prim_amine_entry" in need:
        d["n_prim_amine_entry"] = None if m is None else len(m.GetSubstructMatches(C["ENTRY"]))
    if "mol_size" in need:
        d["mol_size"] = g(C["msz"], nrm)
    if "b_rotR" in need:
        d["b_rotR"] = g(C["b_rotR"], nrm)
    if "logp_ow" in need:
        d["logp_ow"] = g(C["logp_ow"], nrm, enable=True)
    if "slogp" in need:
        d["slogp"] = g(C["slogp"], nrm)
    if "rot_rigid_rdkit" in need:
        rb = g(C["rot"], nrm)
        d["rot_rigid_rdkit"] = rb.get("rot_rigid_rdkit") if isinstance(rb, dict) else None
    if "zwit_group" in need or "zwit_charge" in need:
        zw = g(C["zw"], nrm)
        if isinstance(zw, tuple):
            zw = zw[0]
        if isinstance(zw, dict):
            d["zwit_group"] = zw.get("zwit_group"); d["zwit_charge"] = zw.get("zwit_charge")
    if {"hwi", "fsp3", "fpsa_tpsaSP_asaH"} & set(need):
        x = g(C["WX"].compute, s3)
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, dict):
            for c in ("hwi", "fsp3", "fpsa_tpsaSP_asaH"):
                d[c] = x.get(c)
    if {"dxy_cv", "dtotal_min", "glob_min"} & set(need):
        ens = g(C["ens"], s3, 20, 42)
        if ens is not None:
            dg = g(C["dg"], ens)
            if isinstance(dg, dict):
                for c in ("dxy_cv", "dtotal_min", "glob_min"):
                    d[c] = dg.get(c)
    return d


def write_feat(path, rows, allc, title, extra):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# %s -- S37-25 实验B 中间产物：%s\n" % (os.path.basename(path), title))
        fh.write("# ★★★**供下轮【直接读，★不得重算】**。\n")
        fh.write("# ★★★★实验B = 角色互换：★训练正类 = El Zahed 663 ／ 训练负类 = Ritz 右档 3,503\n")
        fh.write("#    ★★验证集 = CO-ADD 左档（★训练时完全没见过 —— ★[S37-25 核] 交集 0/0）。\n")
        for e in extra:
            fh.write("# %s\n" % e)
        fh.write("# ★口径逐字（★与 [S37-23/24] 同源实现，★参数一个没改）：\n")
        fh.write("#    前处理 smiles_raw→normalize_smiles_syntax()→smiles_norm→OB pH7.4→smiles_ph74\n")
        fh.write("#    min_proj_area = min_projection_area(nconf=20,seed=42) ★**v1**；net_charge_74 = net_charge_pH74()\n")
        fh.write("#    net_charge_v2 = net_charge_pH74_v2(('sulfonamide','formamidine'))；n_prim_amine = WIDE SMARTS\n")
        fh.write("#    dxy_cv/dtotal_min/glob_min ← conformer_ensemble(ph74,20,42)+dipole_and_globularity（★共用一次系综）\n")
        fh.write("# 行数：%d\n" % len(rows))
        w = csv.DictWriter(fh, fieldnames=["inchikey_full"] + allc, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) in (None, "") else str(r.get(c)))
                        for c in ["inchikey_full"] + allc})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    a = ap.parse_args()
    global NREP, S37_OUT, NPROC
    if a.smoke:
        NREP = 1
        # ★★★冒烟测的产物一律写 scratchpad ⇒ ★不污染真盘（★S37-25 修：原先冒烟测直接写进 section37）
        S37_OUT = os.path.join(SCRATCH, "smoke_s37v"); os.makedirs(S37_OUT, exist_ok=True)
    if a.dryrun:
        NREP = 1; NPROC = 2
        S37_OUT = os.path.join(SCRATCH, "dryrun_s37v"); os.makedirs(S37_OUT, exist_ok=True)

    say("#" * 104)
    say("S37-25 实验 B · ★★角色互换")
    say("★开跑 %s   PID=%d" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.getpid()))
    say("★★训练正类 = El Zahed 663 ／ 训练负类 = Ritz 右档 ／ ★★验证 = CO-ADD 左档（★没见过）")
    say("★★★NREP = %d（★承 Lang 定的『探索阶段 NREP=5』）；★其余口径照搬 [S37-24]。" % NREP)
    say("⚠️★★★内部 OOF AUC【不得与 0.6727/0.8756/0.8923 相比】—— ★训练集完全换了，★不是同一把尺。")
    say("#" * 104)

    # ---------- 件〇 ----------
    say("\n" + "=" * 104)
    say("件〇 · 先核")
    say("=" * 104)
    SETS_J = json.load(open(os.path.join(SCRATCH, "s37v_sets.json")))
    pos_ik = set(SETS_J["pos"]); neg_ik = set(SETS_J["neg"])
    left_ik = set(SETS_J["left"]); weak_ik = set(SETS_J["weak"])
    say("  ★训练正类 El Zahed = %d ｜ 训练负类 Ritz 右档 = %d" % (len(pos_ik), len(neg_ik)))
    say("  ★验证集 CO-ADD 左档唯一 IK = %d" % len(left_ik))
    say("     ⚠️★★任务书写『245（唯一 IK 242）』⇒ ★★实测左档 253 行 / **250 唯一 IK**")
    say("        ★245/242 是 [S37-24] 里乙丙丁【构件齐全后】的训练集正类数 ⇒ ★两者不是一回事，★逐字报出。")
    say("  ★★★验证集 ∩ 训练正类 = %d ／ ∩ 训练负类 = %d ⇒ %s"
        % (len(left_ik & pos_ik), len(left_ik & neg_ik),
           "★★零交集 ⇒ 验证集训练时完全没见过 ✓" if not (left_ik & pos_ik) and not (left_ik & neg_ik)
           else "⚠️★★有交集 ⇒ 须剔"))
    w0 = len(weak_ik)
    weak_ik = weak_ik - pos_ik - neg_ik - left_ik
    say("  ★弱阴性对照 El Zahed『EC50>50』= %d ⇒ ★剔与三集重叠后 = %d（★剔了 %d）"
        % (w0, len(weak_ik), w0 - len(weak_ik)))
    say("     ⚠️★★★**只打分看分布，★不进训练、★不当标签**（★[台账四十五] 已裁定那批不能当反例）")

    if a.dryrun:
        pos_ik = set(sorted(pos_ik)[:12]); neg_ik = set(sorted(neg_ik)[:12])
        left_ik = set(sorted(left_ik)[:5]); weak_ik = set(sorted(weak_ik)[:5])
        step("★★空转：训练集各截 12 个、验证/弱阴性各 5 个（★保证 5 折两类都够）")

    # ---------- 构件 ----------
    say("\n" + "=" * 104)
    say("构件准备（★已落盘的直接读，★缺的现算）")
    say("=" * 104)
    FEAT = {}
    XF = load(os.path.join(S37, "extval663_features_XF.tsv"), "inchikey_full\t")
    step("★★★读到 extval663_features_XF.tsv md5=%s ⇒ ★★直接读，★不重算"
         % md5(os.path.join(S37, "extval663_features_XF.tsv")))
    for r in XF:
        if r["inchikey_full"] in pos_ik:
            FEAT[r["inchikey_full"]] = {c: (None if v == "" else float(v))
                                        for c, v in r.items() if c not in ("inchikey_full", "SMILES")}
    say("     ★正类 663 构件已就位 = %d" % len(FEAT))

    p_bf_in = os.path.join(S37, "swapB_features_BF.tsv")      # ★只读复用：★永远看真盘
    p_bf = os.path.join(S37_OUT, "swapB_features_BF.tsv")     # ★写：随 S37_OUT 走
    if os.path.exists(p_bf_in) and not a.dryrun:
        step("★★★读到已落盘 %s ⇒ ★不重算" % os.path.basename(p_bf_in))
        for r in load(p_bf_in, "inchikey_full\t"):
            FEAT[r["inchikey_full"]] = {c: (None if v == "" else float(v))
                                        for c, v in r.items() if c != "inchikey_full"}
        say("     ★读入 %d 个" % len(FEAT))
    else:
        # ★Ritz：油路 12 从 F2 读，水路 7 现算；F2 没覆盖的全算
        F2 = {r["完整InChIKey"]: r for r in load(os.path.join(S37, "oil12_ritz3436_F2.tsv"), "完整InChIKey\t")}
        RC = load(os.path.join(S35, "ritz_ourfeatures_C.tsv"), "row_i\t")
        rsmi = {r["inchikey"]: r["smiles_raw"] for r in RC if r.get("inchikey")}
        jobs = []
        for k in sorted(neg_ik):
            s = rsmi.get(k)
            if not s:
                continue
            if k in F2:
                FEAT[k] = {c: (None if F2[k].get(c) in (None, "") else float(F2[k][c])) for c in OIL12}
                jobs.append((k, s, WAT7))
            else:
                jobs.append((k, s, ALL19))
        say("     ★Ritz 负类：★油路12 从 F2 读到 %d 个；★需现算 %d 个任务" % (len(FEAT) - len(pos_ik), len(jobs)))
        # ★弱阴性：全算
        EZ = load(os.path.join(HOME, "downloads_raw", "elzahed_matched.tsv"), "Compound Name\t")
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")

        def ik(s):
            try:
                m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None
            except Exception:
                return None
        wsmi = {}
        for r in EZ:
            k = ik(r["SMILES"])
            if k in weak_ik and k not in wsmi:
                wsmi[k] = r["SMILES"]
        for k in sorted(wsmi):
            jobs.append((k, wsmi[k], ALL19))
        say("     ★弱阴性：需现算 %d 个" % len(wsmi))
        step("★开始并行现算（NPROC=%d，★共 %d 个任务）" % (NPROC, len(jobs)))
        t0 = time.time()
        with mp.Pool(NPROC) as pl:
            for i, d in enumerate(pl.imap_unordered(work, jobs, chunksize=8), 1):
                k = d.pop("inchikey_full")
                FEAT.setdefault(k, {}).update({c: v for c, v in d.items()})
                if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                    el = time.time() - t0
                    say("       …%d/%d  %.0f s  预计剩余 %.1f 分钟"
                        % (i, len(jobs), el, (el / i) * (len(jobs) - i) / 60))
        step("★现算完成 %.0f s = %.1f 分钟" % (time.time() - t0, (time.time() - t0) / 60))
        allc = sorted({c for k in (neg_ik | weak_ik) if k in FEAT for c in FEAT[k]})
        write_feat(p_bf, [dict(FEAT[k], inchikey_full=k) for k in sorted(neg_ik | weak_ik) if k in FEAT],
                   allc, "Ritz 右档负类 ＋ 弱阴性对照的构件",
                   ["★★正类 663 的构件另在 extval663_features_XF.tsv（★本表不重复）。",
                    "★★弱阴性那批【只打分看分布，★不进训练、★不当标签】。"])
        step("★★★中间产物 立刻落盘 %s  md5=%s" % (os.path.basename(p_bf), md5(p_bf)))

    # ★左档构件（★从 74k 侧的表取，★不现算）
    P = {r["COADD_ID"]: r for r in load(os.path.join(S35, "coadd_prep_L3e.tsv"), "COADD_ID\t")}
    F3 = {r["COADD_ID"]: r for r in load(os.path.join(S37, "oil12_coadd74k_F3.tsv"), "COADD_ID\t")}
    W2 = {r["COADD_ID"]: r for r in load(os.path.join(S37, "water2_74k_W2.tsv"), "COADD_ID\t")}
    T7 = {r["COADD_ID"]: r for r in load(os.path.join(DC, "features_74k.tsv"), "COADD_ID\t")}
    EN = {r["COADD_ID"]: r for r in load(os.path.join(DC, "features_74k_amine_entry.tsv"), "COADD_ID\t")}
    nL = 0
    for cid, r in P.items():
        k = r.get("inchikey_full")
        if k not in left_ik or k in FEAT:
            continue
        d = {}
        for c in ALL19:
            if c in W3:
                v = r.get(c)
            elif c in OIL12:
                v = F3.get(cid, {}).get(c)
            elif c in ("net_charge_v2", "zwit_charge"):
                v = W2.get(cid, {}).get(c)
            elif c == "n_prim_amine_entry":
                v = EN.get(cid, {}).get("n_prim_amine")
            else:
                v = T7.get(cid, {}).get(c)
            d[c] = None if v in (None, "") else float(v)
        FEAT[k] = d; nL += 1
    say("     ★验证集左档构件从 74k 侧表取到 = %d / %d（★不现算）" % (nL, len(left_ik)))

    # ---------- 训练与验证 ----------
    say("\n" + "=" * 104)
    say("件一 · 训练与验证")
    say("=" * 104)
    for name, cols in SETS:
        tr_ids = [k for k in sorted(pos_ik | neg_ik)
                  if k in FEAT and all(FEAT[k].get(c) not in (None, "") for c in cols)]
        X = np.array([[float(FEAT[k][c]) for c in cols] for k in tr_ids])
        y = np.array([1 if k in pos_ik else 0 for k in tr_ids])
        vd = [k for k in sorted(left_ik) if k in FEAT and all(FEAT[k].get(c) not in (None, "") for c in cols)]
        Xv = np.array([[float(FEAT[k][c]) for c in cols] for k in vd]) if vd else np.zeros((0, len(cols)))
        wk = [k for k in sorted(weak_ik) if k in FEAT and all(FEAT[k].get(c) not in (None, "") for c in cols)]
        Xw = np.array([[float(FEAT[k][c]) for c in cols] for k in wk]) if wk else np.zeros((0, len(cols)))
        say("\n  ── %s ｜ 构件 %d" % (name, len(cols)))
        say("     ★新训练集 n=%d（正类 El Zahed %d ／ 负类 Ritz %d）"
            % (len(y), int(y.sum()), len(y) - int(y.sum())))
        say("     ★验证集 CO-ADD 左档可打分 = %d ／ %d ｜ ★弱阴性可打分 = %d"
            % (len(vd), len(left_ik), len(wk)))
        # ★★空转要走到【每一个落盘点】⇒ 空转时放宽这道闸（★正式跑不变）
        _min_n, _min_c = (10, 5) if a.dryrun else (20, 5)
        if len(y) < _min_n or y.sum() < _min_c or (len(y) - y.sum()) < _min_c:
            say("     ⚠️★n 太小 ⇒ 跳过"); continue
        from sklearn.model_selection import RepeatedStratifiedKFold
        splits = list(RepeatedStratifiedKFold(n_splits=NSPLIT, n_repeats=NREP,
                                              random_state=SEED).split(X, y))
        R, MDL = {}, {}
        for nm, fit in fitters():
            t0 = time.time()
            acc = np.zeros(len(y)); cnt = np.zeros(len(y)); cuts = []
            for tr, te in splits:
                pr = fit(X[tr], y[tr])
                acc[te] += pr(X[te]); cnt[te] += 1
                c = cut_youden(pr(X[tr]), y[tr])
                if c is not None:
                    cuts.append(c)
            full = fit(X, y)
            MDL[nm] = full
            R[nm] = dict(oof=acc / cnt, cut1=float(np.median(cuts)),
                         val=full(Xv) if len(vd) else np.array([]),
                         weak=full(Xw) if len(wk) else np.array([]))
            say("      %s %-38s (%.0fs)" % (name[:1], nm, time.time() - t0))
        # ★★中间产物立刻落盘
        import joblib
        tag = name.split()[0]
        p_m = os.path.join(S37_OUT, "swapB_models_%s_BM.joblib" % tag)
        joblib.dump(dict(套=name, 构件=cols, 模型=MDL, 训练IK=tr_ids, y=y,
                         口径="实验B 角色互换；RepeatedStratifiedKFold(5折×%d次,seed=%d)" % (NREP, SEED),
                         训练正类="El Zahed 663", 训练负类="Ritz 右档 3503",
                         验证集="CO-ADD 左档（★训练时没见过）"), p_m, compress=3)
        say("     ★★★中间产物 立刻落盘 %s md5=%s" % (os.path.basename(p_m), md5(p_m)))
        p_o = os.path.join(S37_OUT, "swapB_oof_%s_BO.tsv" % tag)
        with open(p_o, "w", encoding="utf-8", newline="") as fh:
            fh.write("# swapB_oof_%s_BO.tsv -- S37-25 实验B：★新训练集上的逐分子 OOF。\n" % tag)
            fh.write("# ★★★训练正类 = El Zahed 663 ／ 训练负类 = Ritz 右档 3,503（★y 列：1=正类）。\n")
            fh.write("# ★★切点一律在【本表这些 OOF 分数】上定线，★不得在验证集上重新定线。\n")
            fh.write("# ★口径 RepeatedStratifiedKFold(5折×%d次, seed=%d)；★超参不变。\n" % (NREP, SEED))
            fh.write("# 行数：%d\n" % len(y))
            w = csv.DictWriter(fh, fieldnames=["inchikey_full", "y"] + list(R), delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for i, k in enumerate(tr_ids):
                w.writerow(dict({"inchikey_full": k, "y": int(y[i])},
                                **{nm: "%.10f" % R[nm]["oof"][i] for nm in R}))
        say("     ★★★中间产物 立刻落盘 %s md5=%s 行=%d" % (os.path.basename(p_o), md5(p_o), len(y)))
        for suf, ids2, key in (("BV", vd, "val"), ("BW", wk, "weak")):
            p_x = os.path.join(S37_OUT, "swapB_%s_%s_%s.tsv"
                               % ("valscore" if suf == "BV" else "weakscore", tag, suf))
            with open(p_x, "w", encoding="utf-8", newline="") as fh:
                fh.write("# %s -- S37-25 实验B：%s 的逐分子分数。\n"
                         % (os.path.basename(p_x),
                            "★验证集 CO-ADD 左档（★训练时没见过）" if suf == "BV"
                            else "★★弱阴性对照 El Zahed EC50>50（★只看分布，★不当标签）"))
                fh.write("# ★模型 = 【全新训练集拟合】；★分数与 swapB_oof_*_BO.tsv 同尺度。\n")
                fh.write("# 行数：%d\n" % len(ids2))
                w = csv.DictWriter(fh, fieldnames=["inchikey_full"] + list(R), delimiter="\t", extrasaction="ignore")
                w.writeheader()
                for i, k in enumerate(ids2):
                    w.writerow(dict({"inchikey_full": k},
                                    **{nm: "%.10f" % R[nm][key][i] for nm in R}))
            say("     ★★★中间产物 立刻落盘 %s md5=%s 行=%d" % (os.path.basename(p_x), md5(p_x), len(ids2)))
        step("★★%s 训练完成 ＋ 四个中间产物已落盘" % name)

        # ---------- 报数 ----------
        say("\n     ▎★内部 OOF AUC（★新训练集上）⚠️★★不得与 0.6727/0.8756/0.8923 相比")
        for nm in R:
            say("       %-40s %.4f" % (nm, auc_np(y, R[nm]["oof"])))
        say("\n     ▎★★★必报 · CO-ADD 左档 %d 个里被判『能穿透』的有多少（★切点三套并排）" % len(vd))
        npos = int(y.sum())
        say("       %-40s %-20s %12s %14s %14s %14s"
            % ("模型", "切点套", "阈值", "★验证判对", "判对率", "★训练集灵敏度"))
        for nm in R:
            oof = R[nm]["oof"]; val = R[nm]["val"]
            cut2 = cut_youden(oof, y)
            cut3 = float(np.sort(oof)[::-1][npos - 1])
            for cn, thr in (("甲-1 Youden折内中位", R[nm]["cut1"]),
                            ("甲-2 Youden全体OOF", cut2), ("乙 正类比例", cut3)):
                sens = float(((oof >= thr) & (y == 1)).sum()) / npos
                hit = int((val >= thr).sum()) if len(val) else 0
                rate = hit / len(val) if len(val) else float("nan")
                say("       %-40s %-20s %12.6f %9d/%-4d %13.2f%% %13.2f%%"
                    % (nm[:38], cn, thr, hit, len(val), 100 * rate, 100 * sens))
                OUT.append(dict(套=name, 模型=nm, 切点套=cn, 阈值="%.8f" % thr,
                                内部OOF_AUC="%.4f" % auc_np(y, oof),
                                验证集n=len(val), 验证判对=hit,
                                验证判对率="%.6f" % rate, 训练集灵敏度="%.6f" % sens,
                                掉多少pp="%+.4f" % (100 * (rate - sens)),
                                训练集n=len(y), 训练正类=npos))
        say("\n     ▎★件二 · 弱阴性对照（★只报分布，★不当标签）")
        say("       %-40s %-12s %8s %10s %10s %10s %10s %10s"
            % ("模型", "批", "n", "min", "P25", "中位", "P75", "max"))
        for nm in R:
            for lab, v in (("★训练正类663", R[nm]["oof"][y == 1]),
                           ("训练负类Ritz", R[nm]["oof"][y == 0]),
                           ("★★弱阴性3747", R[nm]["weak"]),
                           ("★★验证左档", R[nm]["val"])):
                s = q5(v)
                say("       %-40s %-12s %8d %10.4f %10.4f %10.4f %10.4f %10.4f"
                    % (nm[:38] if lab.startswith("★训练正") else "", lab, s["n"],
                       s["min"], s["p25"], s["med"], s["p75"], s["max"]))
                OUT.append(dict(套=name, 模型=nm, 切点套="—分布—", 批=lab, 验证集n=s["n"],
                                min="%.6f" % s["min"], P25="%.6f" % s["p25"],
                                中位="%.6f" % s["med"], P75="%.6f" % s["p75"],
                                max="%.6f" % s["max"]))
            pw = q5(R[nm]["weak"])["med"]; pp = q5(R[nm]["oof"][y == 1])["med"]
            pn = q5(R[nm]["oof"][y == 0])["med"]
            sep = "★★分得开" if abs(pw - pp) > abs(pw - pn) else "★★★分不开（★弱阴性更靠近正类）"
            say("       %-40s ★★★必答：★弱阴性中位 %.4f ｜ 正类中位 %.4f ｜ 负类中位 %.4f ⇒ %s"
                % ("", pw, pp, pn, sep))

    if a.dryrun:
        step("★★空转已走完【每一个落盘点】")

    cc = ["套", "模型", "切点套", "批", "阈值", "内部OOF_AUC", "验证集n", "验证判对",
          "验证判对率", "训练集灵敏度", "掉多少pp", "训练集n", "训练正类",
          "min", "P25", "中位", "P75", "max"]
    t = os.path.join(S37_OUT, "swapB_result_BV.tsv")
    with open(t, "w", encoding="utf-8", newline="") as fh:
        fh.write("# swapB_result_BV.tsv -- S37-25 实验B · ★★角色互换：一行 = 一（套,模型,切点套/批）。\n")
        fh.write("# ★★★★训练正类 = El Zahed 663 ／ 训练负类 = Ritz 右档 3,503\n")
        fh.write("# ★★★★验证集 = CO-ADD 左档（★[S37-25 核] 与训练正/负类交集 0/0 ⇒ 训练时完全没见过）。\n")
        fh.write("# ⚠️★★★『内部OOF_AUC』【不得与 0.6727/0.8756/0.8923 相比】—— ★训练集完全换了，★不是同一把尺。\n")
        fh.write("# ★NREP=%d（★Lang 定的探索阶段值）；★切点三套【全部在新训练集的 OOF 上定线】。\n" % NREP)
        fh.write("# ⚠️★★弱阴性那批【只报分布，★不当标签】（★[台账四十五] 已裁定不能当反例）。\n")
        fh.write("# 行数：%d\n" % len(OUT))
        w = csv.DictWriter(fh, fieldnames=cc, delimiter="\t", extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in OUT]
    say("\n★落盘 %s  md5=%s  数据行=%d" % (os.path.relpath(t, HOME), md5(t), len(OUT)))
    mdp = os.path.join(S37_OUT, "swapB_result_BV.md")
    with open(mdp, "w", encoding="utf-8") as fh:
        fh.write("# S37-25 实验 B · 角色互换\n\n★★训练正类=El Zahed 663／负类=Ritz 右档；★验证=CO-ADD 左档。\n\n---\n\n```\n")
        fh.write("\n".join(LOG)); fh.write("\n```\n")
    say("★落盘 %s" % os.path.relpath(mdp, HOME))
    if a.dryrun:
        import shutil
        n = len(os.listdir(S37_OUT))
        step("★★空转通了 —— ★共 %d 个假产物" % n)
        for f in sorted(os.listdir(S37_OUT)):
            say("     ★假产物 %s  %d 字节" % (f, os.path.getsize(os.path.join(S37_OUT, f))))
        shutil.rmtree(S37_OUT, ignore_errors=True)
        step("★★假产物已全部删除")
    say("\n★收工 %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # ★★★冒烟测也【不写 .DONE】（★S37-25 修：原先只排除 --dryrun ⇒ 冒烟测写了个误导性标记）
    _nomark = ("--dryrun" in sys.argv) or ("--smoke" in sys.argv)
    _done = os.path.join(S37, "s37v.DONE") if not _nomark else os.devnull
    try:
        if not _nomark and os.path.exists(_done):
            os.remove(_done)
    except Exception:
        pass
    try:
        main()
        step("★★★★全部完成")
        if not _nomark:
            with open(_done, "w", encoding="utf-8") as fh:
                fh.write("S37-25 DONE %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("★★★写标记 %s" % _done, flush=True)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        step("⚠️★★★★崩溃 —— ★逐字写进日志末尾（★不静默死掉）")
        print(tb, flush=True)
        try:
            with open(os.path.join(S37_OUT, "swapB_result_BV.md"), "w", encoding="utf-8") as fh:
                fh.write("# S37-25 ★★★崩溃，★未跑完\n\n```\n")
                fh.write("\n".join(LOG)); fh.write("\n\n" + tb + "\n```\n")
        except Exception:
            pass
        raise
