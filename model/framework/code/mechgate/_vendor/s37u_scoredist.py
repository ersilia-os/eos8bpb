#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s37u_scoredist.py -- S37-24 件一：外部集 663 的分数分布 vs 74k 左档正类

★★★★★不重训、★不改秤、★不出对外 verdict、★不裁任何未定项。
★★★件〇 未得 Lang 裁定前，★本脚本【不算任何 AUC】、★不自行选负例。

★做法：★与 [S37-23] 逐字同一套 —— 训练只用 74k（★外部集按 InChIKey 硬剔），
   ★用【全 74k 拟合】的模型给外部集打分；★★然后【只报分布】，★不定任何切点。
★★★必答：★外部集的分数分布，★是整体偏低、★还是与左档重合？
   ⇒ ★另报：★外部集分数落在【左档分数分布】的哪个分位区间（★逐个分子求分位后取分布）

⚠️ 两条禁读：cooper_features.csv / cooper_casecontrol_* —— guard 硬拦。
"""
from __future__ import annotations
import os, tempfile, sys, csv, time, json, hashlib, datetime, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 原为开发机的 antibiotic-benchmark/scripts，会抢在本包 _vendor 之前；改为本模块自身所在目录，同族模块照常平铺 import
DC = os.path.join(HOME, "antibiotic-benchmark", "data_curated")
S35 = os.path.join(HOME, "downloads_raw", "section35")
S37 = os.path.join(HOME, "downloads_raw", "section37")
S37_OUT = S37     # ★产物写出目录（★空转时改道 scratchpad，★只读输入始终走 S37）
SCRATCH = os.environ.get("MECHGATE_SCRATCH", os.path.join(tempfile.gettempdir(), "mechgate_scratch"))  # 原为开发机的绝对路径，已去标识化；仅训练/空转代码用到，推理路径不碰
FORBIDDEN = ("cooper_features.csv", "cooper_casecontrol_")
SEED, NREP, NSPLIT, THR = 42, 20, 5, 43.3093
LOG, OUT = [], []
OIL12 = ["b_rotR", "n_prim_amine_wide", "dxy_cv", "dtotal_min", "logp_ow", "slogp",
         "glob_min", "rot_rigid_rdkit", "zwit_group", "hwi", "fsp3", "fpsa_tpsaSP_asaH"]
W3 = ["min_proj_area", "net_charge_74", "n_prim_amine"]
UNION16 = OIL12 + ["net_charge_v2", "zwit_charge", "n_prim_amine_entry", "mol_size"]


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def step(*a):
    """★★进度行：★带时间戳，★每完成一套就写一行（★[S37-24] 新增，★以后每轮照办）。"""
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


def q5(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(n=len(v), min=v.min(), p25=np.percentile(v, 25), med=np.median(v),
                p75=np.percentile(v, 75), max=v.max())


# ══════════════════════════════════════════════════════════════════════════
# ★★★可 pickle 的打分器（★[S37-24] 修 PicklingError）
#   ★根因：★原先四个 fitter 全部返回【闭包 lambda】⇒ ★lambda 不能 pickle
#          ⇒ ★joblib.dump 落盘模型对象时抛 PicklingError
#   ★改法：★★**改存【参数】而非可调用对象** —— ★两个【模块级类】，★实例可 pickle
#   ⚠️★★★**打分公式逐字不变** ⇒ ★★输出数值【完全不变】：
#          · EqWeight   : (((X - mu) / sd) * sg).sum(1)      ★与原 lambda 逐字相同
#          · ProbaScorer: m.predict_proba(sc.transform(X))[:,1]（sc 可为 None）★同上
# ══════════════════════════════════════════════════════════════════════════
class EqWeight:
    """★等权合分(方向统一)：★只存 mu / sd / sg 三个【参数】，★不存闭包。"""
    def __init__(self, mu, sd, sg):
        self.mu, self.sd, self.sg = mu, sd, sg

    def __call__(self, X):
        return (((X - self.mu) / self.sd) * self.sg).sum(1)


class ProbaScorer:
    """★sklearn/xgboost 模型的打分器：★存【模型对象 + 可选 scaler】，★不存闭包。"""
    def __init__(self, model, scaler=None):
        self.model, self.scaler = model, scaler

    def __call__(self, X):
        Z = self.scaler.transform(X) if self.scaler is not None else X
        return self.model.predict_proba(Z)[:, 1]


def eq_weight_fit(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1); sd = np.where(sd == 0, 1.0, sd)
    sg = np.array([1.0 if Xtr[ytr == 1, k].mean() >= Xtr[ytr == 0, k].mean() else -1.0
                   for k in range(Xtr.shape[1])])
    return EqWeight(mu, sd, sg)


def fitters():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    def f_eq(Xtr, ytr):
        return eq_weight_fit(Xtr, ytr)

    def f_lr(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        return ProbaScorer(m, sc)          # ★同一处根因：闭包 → 可 pickle 对象；★公式不变

    def f_rf(Xtr, ytr):
        m = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=SEED, n_jobs=1).fit(Xtr, ytr)
        return ProbaScorer(m)              # ★同上

    def f_xgb(Xtr, ytr):
        m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=SEED,
                          n_jobs=1, eval_metric="logloss").fit(Xtr, ytr)
        return ProbaScorer(m)              # ★同上
    return [("等权合分(方向统一)", f_eq), ("逻辑回归(L2, C=1)", f_lr),
            ("随机森林(depth=3, n=100, 单线程)", f_rf),
            ("XGBoost(depth=3, n=200, lr=0.1, 单线程)", f_xgb)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dryrun", action="store_true",
                    help="★★空转：5 个假分子走完整条流程（★含每一个落盘点），★跑完删假产物")
    a = ap.parse_args()
    global NREP, S37_OUT
    if a.smoke:
        NREP = 1
    if a.dryrun:
        NREP = 1
        # ★★空转：★★**只有【产物写出】改道 scratchpad**；★只读输入仍走真盘（S37）
        S37_OUT = os.path.join(SCRATCH, "dryrun_s37u")
        os.makedirs(S37_OUT, exist_ok=True)
    say("#" * 104)
    say("S37-24 件一 · 外部集 663 的分数分布 vs 74k 左档正类")
    say("★开跑 %s   PID=%d" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.getpid()))
    say("★★★★★不重训、★不改秤、★不出 verdict；★★★件〇 未裁前【不算任何 AUC】、★不选负例。")
    say("#" * 104)

    final = set(json.load(open(os.path.join(SCRATCH, "ez_final.json"))))
    EZ = load(os.path.join(HOME, "downloads_raw", "elzahed_matched.tsv"), "Compound Name\t")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def ik(s):
        try:
            m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None
        except Exception:
            return None
    smi = {}
    for r in EZ:
        k = ik(r["SMILES"])
        if k in final and k not in smi:
            smi[k] = r["SMILES"]
    ext_ids = sorted(smi)
    if a.dryrun:
        ext_ids = ext_ids[:5]
        smi = {k: smi[k] for k in ext_ids}
        step("★★空转：外部集截成 %d 个假分子（★真集是 663）" % len(ext_ids))
    say("\n★外部集 n=%d（★[S37-23] 件〇 定义，★663）" % len(ext_ids))

    # ---------- 现算外部集构件（★与 [S37-23] 逐字同一套）----------
    from _smiles_syntax import normalize_smiles_syntax
    from a2_min_projection_area_v2 import protonate_ph74
    from a2_min_projection_area import min_projection_area
    from penetration_features import (b_rotR, logp_ow, net_charge_pH74,
                                      net_charge_pH74_v2, n_primary_amine, mol_size)
    from efflux_features import slogp
    from _water_candidates import (conformer_ensemble, dipole_and_globularity,
                                   rot_bond_counts, zwitterion_flags)
    import s31m_water_extra as WX
    WIDE = Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(N=*)]")
    ENTRY = Chem.MolFromSmarts("[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]")
    # ★★★优先【直接读】已落盘的构件表，★不重算（★补充条：中间产物一律复用）
    p_feat = os.path.join(S37_OUT, "extval663_features_XF.tsv")
    FEAT = {}
    if os.path.exists(p_feat) and not a.dryrun:
        step("★★★读到已落盘构件表 %s" % os.path.basename(p_feat))
        say("     md5=%s ⇒ ★★**直接读，★不重算**（★省 ~23 分钟现算）" % md5(p_feat))
        for r in load(p_feat, "inchikey_full\t"):
            k = r["inchikey_full"]
            FEAT[k] = {c: (None if v == "" else float(v))
                       for c, v in r.items() if c not in ("inchikey_full", "SMILES")}
        say("     ★读入 %d 个分子 ｜ ★与外部集 663 的交集 = %d"
            % (len(FEAT), len(set(FEAT) & set(ext_ids))))
        if set(ext_ids) <= set(FEAT):
            step("★构件表覆盖完整 ⇒ ★跳过现算")
        else:
            say("     ⚠️★覆盖不全（缺 %d）⇒ ★★仍走现算" % len(set(ext_ids) - set(FEAT)))
            FEAT = {}
    if not FEAT:
        step("★盘上无可复用构件表 ⇒ ★开始现算（★串行）")
    t0 = time.time()
    for j, k in enumerate(ext_ids if not FEAT else [], 1):
        s = smi[k]; d = {}
        try:
            nrm = normalize_smiles_syntax(s) or s
        except Exception:
            nrm = s
        try:
            ph = protonate_ph74(nrm)
        except Exception:
            ph = None
        s3 = ph or nrm

        def g(f, *a2, **kw):
            try:
                return f(*a2, **kw)
            except Exception:
                return None
        m = g(Chem.MolFromSmiles, nrm)
        d["b_rotR"] = g(b_rotR, nrm)
        d["n_prim_amine_wide"] = None if m is None else len(m.GetSubstructMatches(WIDE))
        d["n_prim_amine_entry"] = None if m is None else len(m.GetSubstructMatches(ENTRY))
        d["n_prim_amine"] = d["n_prim_amine_wide"]
        d["logp_ow"] = g(logp_ow, nrm, enable=True)
        d["slogp"] = g(slogp, nrm)
        d["mol_size"] = g(mol_size, nrm)
        d["net_charge_74"] = g(net_charge_pH74, nrm)
        v = g(net_charge_pH74_v2, nrm)
        d["net_charge_v2"] = v[0] if isinstance(v, tuple) else v
        rb = g(rot_bond_counts, nrm)
        d["rot_rigid_rdkit"] = rb.get("rot_rigid_rdkit") if isinstance(rb, dict) else None
        zw = g(zwitterion_flags, nrm)
        if isinstance(zw, tuple):
            zw = zw[0]
        if isinstance(zw, dict):
            d["zwit_group"] = zw.get("zwit_group"); d["zwit_charge"] = zw.get("zwit_charge")
        x = g(WX.compute, s3)
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, dict):
            for c in ("hwi", "fsp3", "fpsa_tpsaSP_asaH"):
                d[c] = x.get(c)
        d["min_proj_area"] = g(min_projection_area, nrm, 20, 42)
        ens = g(conformer_ensemble, s3, 20, 42)
        if ens is not None:
            dg = g(dipole_and_globularity, ens)
            if isinstance(dg, dict):
                for c in ("dxy_cv", "dtotal_min", "glob_min"):
                    d[c] = dg.get(c)
        FEAT[k] = d
        if j % 200 == 0 or j == len(ext_ids):
            say("    …%d/%d  %.0f s" % (j, len(ext_ids), time.time() - t0))
    say("  ✔ 现算 %.0f s" % (time.time() - t0))

    # ══════ ★★★中间产物① · 外部集构件表 —— ★★算完【立刻】落盘，★不攒到最后 ══════
    #   ⚠️★★承 [S37-23] 教训：★那次崩在落盘之前，★41 分钟的结果全丢了
    ALLC = sorted({c for d in FEAT.values() for c in d})
    if os.path.exists(p_feat):
        step("★中间产物① 已在盘上（★本轮是读来的）⇒ ★不重写，★md5=%s" % md5(p_feat))
    else:
      with open(p_feat, "w", encoding="utf-8", newline="") as fh:
          fh.write("# extval663_features_XF.tsv -- S37-24 中间产物①：外部集 663 个的构件表。\n")
          fh.write("# ★★★**本表供下轮【直接读，★不得重算】** —— ★现算一次约 19 分钟。\n")
          fh.write("# ★对象 = [S37-23 件〇] 定义的外部集：El Zahed『野生型 EC50 可测(非>50)』\n")
          fh.write("#    ⇒ 唯一 IK 710 − 左档重叠 17 − 右档冲突 2 = 691 − 与 74k 训练集重叠 28 = **663**\n")
          fh.write("# ★★★**这 663 个【未进任何模型的训练集】**（★build() 里按 InChIKey 硬剔）。\n")
          fh.write("# ★口径逐字（★与 74k 侧同源实现，★参数一个没改）：\n")
          fh.write("#    前处理：smiles_raw → normalize_smiles_syntax() → smiles_norm → OB pH7.4 → smiles_ph74\n")
          fh.write("#    min_proj_area = min_projection_area(nconf=20, seed=42)  ★**v1**（a2_min_projection_area.py:25）\n")
          fh.write("#    net_charge_74 = net_charge_pH74()   ／ net_charge_v2 = net_charge_pH74_v2(('sulfonamide','formamidine'))\n")
          fh.write("#    n_prim_amine  = WIDE SMARTS `[NX3;H2;!$(NC=O);!$(N=*)]`（★与 coadd_prep 表头同口径）\n")
          fh.write("#    n_prim_amine_entry = ENTRY SMARTS `[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]`\n")
          fh.write("#    dxy_cv/dtotal_min/glob_min ← conformer_ensemble(smiles_ph74,20,42)+dipole_and_globularity（★共用一次系综）\n")
          fh.write("#    hwi/fsp3/fpsa_tpsaSP_asaH ← s31m_water_extra.compute(smiles_ph74)\n")
          fh.write("# ⚠️★★★**不含 min_proj_area_v2**（★甲用 v1、★丁那 16 列里没有任何 min_proj_area）。\n")
          fh.write("# 行数：%d\n" % len(ext_ids))
          w = csv.DictWriter(fh, fieldnames=["inchikey_full", "SMILES"] + ALLC,
                             delimiter="\t", extrasaction="ignore")
          w.writeheader()
          for k in ext_ids:
              row = {"inchikey_full": k, "SMILES": smi[k]}
              for c in ALLC:
                  v = FEAT[k].get(c)
                  row[c] = "" if v in (None, "") else str(v)
              w.writerow(row)
    say("\n  ★★★中间产物① 立刻落盘 %s" % os.path.relpath(p_feat, HOME))
    say("     md5=%s  数据行=%d  列=%d" % (md5(p_feat), len(ext_ids), len(ALLC) + 2))

    P = {r["COADD_ID"]: r for r in load(os.path.join(S35, "coadd_prep_L3e.tsv"), "COADD_ID\t")
         if r["入组排除"] == "否"}
    F = {r["COADD_ID"]: r for r in load(os.path.join(S37, "oil12_coadd74k_F3.tsv"), "COADD_ID\t")}
    W = {r["COADD_ID"]: r for r in load(os.path.join(S37, "water2_74k_W2.tsv"), "COADD_ID\t")}
    T = {r["COADD_ID"]: r for r in load(os.path.join(DC, "features_74k.tsv"), "COADD_ID\t")}
    E = {r["COADD_ID"]: r for r in load(os.path.join(DC, "features_74k_amine_entry.tsv"), "COADD_ID\t")}
    ikmap = {c: P[c].get("inchikey_full", "") for c in P}

    def build(cols):
        D = []
        for cid, r in P.items():
            if r.get("inhib_wt") in (None, "") or ikmap[cid] in final:
                continue
            row, ok = {}, True
            for col in cols:
                if col in W3:
                    v = r.get(col)
                elif col in OIL12:
                    v = F.get(cid, {}).get(col)
                elif col in ("net_charge_v2", "zwit_charge"):
                    v = W.get(cid, {}).get(col)
                elif col == "n_prim_amine_entry":
                    v = E.get(cid, {}).get("n_prim_amine")
                else:
                    v = T.get(cid, {}).get(col)
                if v in (None, ""):
                    ok = False; break
                row[col] = float(v)
            if ok:
                D.append((cid, row))
        X = np.array([[d[c] for c in cols] for _, d in D])
        y = (np.array([float(P[c]["inhib_wt"]) for c, _ in D]) >= THR).astype(int)
        ids2 = [c for c, _ in D]
        if a.dryrun:                       # ★★空转：★正负各取若干，★保证两类都在
            pos = [i for i in range(len(y)) if y[i] == 1][:30]
            neg = [i for i in range(len(y)) if y[i] == 0][:170]
            sel = sorted(pos + neg)
            return [ids2[i] for i in sel], X[sel], y[sel]
        return ids2, X, y

    SETS = [("甲 水路3", W3, 0.6727), ("乙 油路12", OIL12, 0.8756),
            ("丙 合15", W3 + OIL12, None), ("丁 并集16", UNION16, 0.8923)]
    if a.smoke:
        SETS = [("甲 水路3", W3, None)]
        say("\n  ⚠️★冒烟测：只跑甲，NREP=1")

    from sklearn.model_selection import RepeatedStratifiedKFold
    say("\n" + "=" * 104)
    say("件一 · 分数分布并排（★★不定任何切点，★不算 AUC）")
    say("=" * 104)
    store = {}
    for name, cols, ref in SETS:
        ids, X, y = build(cols)
        keep = [k for k in ext_ids if all(FEAT[k].get(c) not in (None, "") for c in cols)]
        Xe = np.array([[float(FEAT[k][c]) for c in cols] for k in keep])
        say("\n  ── %s ｜ 构件 %d ｜ 训练 n=%d 正类=%d ｜ 外部可打分 %d"
            % (name, len(cols), len(y), int(y.sum()), len(keep)))
        splits = list(RepeatedStratifiedKFold(n_splits=NSPLIT, n_repeats=NREP,
                                              random_state=SEED).split(X, y))
        R = {}; MDL = {}
        for nm, fit in fitters():
            t0 = time.time()
            acc = np.zeros(len(y)); cnt = np.zeros(len(y))
            for tr, te in splits:
                acc[te] += fit(X[tr], y[tr])(X[te]); cnt[te] += 1
            oof = acc / cnt
            full = fit(X, y)                       # ★★全 74k 拟合（★外部集未进训练）
            MDL[nm] = full
            R[nm] = dict(oof=oof, ext=full(Xe))
            say("      %s %-38s (%.0fs)" % (name[:1], nm, time.time() - t0))
        if ref is not None and not a.dryrun:
            ax = auc_np(y, R["XGBoost(depth=3, n=200, lr=0.1, 单线程)"]["oof"])
            ok = abs(ax - ref) < 0.001
            say("     ▎★★复现核验：XGB OOF AUC = %.4f（[S37-23] 记 %.4f）⇒ %s"
                % (ax, ref, "★一致" if ok else "⚠️★★★对不上"))
            if not ok:
                say("     ⇒ ★★★★**对不上 ⇒ 停下回报。**"); return
        store[name] = (y, R, keep)

        # ══════ ★★★中间产物②③ · ★★每套训完【立刻】落盘，★不攒到最后 ══════
        import joblib
        p_mdl = os.path.join(S37_OUT, "models_%s_XM.joblib" % name.split()[0])
        joblib.dump(dict(套=name, 构件=cols, 模型={nm: MDL[nm] for nm in MDL},
                         训练集COADD_ID=ids, y=y,
                         口径="s35g 现役：RepeatedStratifiedKFold(5折×%d次, seed=%d)" % (NREP, SEED),
                         外部集663未进训练=True), p_mdl, compress=3)
        say("     ★★★中间产物② 立刻落盘 %s  md5=%s"
            % (os.path.relpath(p_mdl, HOME), md5(p_mdl)))
        p_oof = os.path.join(S37_OUT, "oof_%s_XO.tsv" % name.split()[0])
        with open(p_oof, "w", encoding="utf-8", newline="") as fh:
            fh.write("# oof_%s_XO.tsv -- S37-24 中间产物③：★74k 上的【逐分子 OOF 预测值】。\n" % name.split()[0])
            fh.write("# ★★★**本表供下轮【直接读，★不得重训】** —— ★把门指标/取max/任何切点实验都用它。\n")
            fh.write("#    ⚠️★★承 [S37-17] 教训：★那次『逐分子预测值盘上没有』⇒ 同一套模型今天已训过三次。\n")
            fh.write("# ★套 = %s ｜ 构件 %d 个：%s\n" % (name, len(cols), ";".join(cols)))
            fh.write("# ★口径 = s35g 现役：RepeatedStratifiedKFold(5 折 × %d 次, seed=%d)；★超参未改。\n" % (NREP, SEED))
            fh.write("# ★标签 = inhib_wt >= %.4f（★列 y）。\n" % THR)
            fh.write("# ★★★**训练集范围 = 74k 入组后，★已按 InChIKey 硬剔外部集那 663 个 ⇒ 外部集未进训练**。\n")
            fh.write("# ★★切点若要定，★一律在【本表这些 74k OOF 分数】上定线，★不得在外部集上重新定线。\n")
            fh.write("# 行数：%d\n" % len(y))
            cc = ["COADD_ID", "y"] + list(R)
            w = csv.DictWriter(fh, fieldnames=cc, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for i, cid in enumerate(ids):
                w.writerow(dict({"COADD_ID": cid, "y": int(y[i])},
                                **{nm: "%.10f" % R[nm]["oof"][i] for nm in R}))
        say("     ★★★中间产物③ 立刻落盘 %s  md5=%s  数据行=%d"
            % (os.path.relpath(p_oof, HOME), md5(p_oof), len(y)))
        # ★外部集分数也一并落（★下轮判对率/切点实验直接读）
        p_ext = os.path.join(S37_OUT, "extscore_%s_XE.tsv" % name.split()[0])
        with open(p_ext, "w", encoding="utf-8", newline="") as fh:
            fh.write("# extscore_%s_XE.tsv -- S37-24 中间产物③b：★外部集 663 的逐分子分数。\n" % name.split()[0])
            fh.write("# ★★模型 = 【全 74k 拟合】（★外部集未进训练）；★★分数与 oof_*_XO.tsv 同尺度。\n")
            fh.write("# ★套 = %s ｜ 可打分 %d / 663（★构件齐全者）。\n" % (name, len(keep)))
            fh.write("# 行数：%d\n" % len(keep))
            cc = ["inchikey_full"] + list(R)
            w = csv.DictWriter(fh, fieldnames=cc, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for i, k in enumerate(keep):
                w.writerow(dict({"inchikey_full": k},
                                **{nm: "%.10f" % R[nm]["ext"][i] for nm in R}))
        say("     ★★★中间产物③b 立刻落盘 %s  md5=%s  数据行=%d"
            % (os.path.relpath(p_ext, HOME), md5(p_ext), len(keep)))
        step("★★%s 训练完成 ＋ 三个中间产物已落盘" % name)
        say("\n     %-38s %-10s %9s %9s %9s %9s %9s" % ("模型", "批", "min", "P25", "中位", "P75", "max"))
        for nm, d in R.items():
            left = d["oof"][y == 1]
            allo = d["oof"]
            ext = d["ext"]
            for lab, v in (("★74k左档正类", left), ("74k全体", allo), ("★★外部集663", ext)):
                s = q5(v)
                say("     %-38s %-10s %9.4f %9.4f %9.4f %9.4f %9.4f"
                    % (nm[:36] if lab.startswith("★74k左") else "", lab,
                       s["min"], s["p25"], s["med"], s["p75"], s["max"]))
                OUT.append(dict(套=name, 模型=nm, 批=lab, n=s["n"],
                                min="%.6f" % s["min"], P25="%.6f" % s["p25"],
                                中位="%.6f" % s["med"], P75="%.6f" % s["p75"],
                                max="%.6f" % s["max"]))
            # ★★必答：外部集分数落在左档分布的哪个分位
            lp = np.searchsorted(np.sort(left), ext, side="right") / len(left)
            s2 = q5(lp)
            say("     %-38s %-10s %9.4f %9.4f %9.4f %9.4f %9.4f"
                % ("", "→在左档分位", s2["min"], s2["p25"], s2["med"], s2["p75"], s2["max"]))
            below = float((lp < 0.5).mean())
            verdict = ("★★整体偏低 ⇒ 模型的问题" if below > 0.60 else
                       ("★★重合 ⇒ 切点的问题" if below < 0.55 else "★介于两者之间"))
            say("     %-38s %-10s ★★外部集有 %.1f%% 的分子分数低于【左档中位】"
                % ("", "", 100 * below))
            say("     %-38s %-10s    ★左档自身该比例按定义 = 50.0%% ⇒ %s"
                % ("", "", verdict))
            OUT.append(dict(套=name, 模型=nm, 批="→在左档分位", n=s2["n"],
                            min="%.6f" % s2["min"], P25="%.6f" % s2["p25"],
                            中位="%.6f" % s2["med"], P75="%.6f" % s2["p75"],
                            max="%.6f" % s2["max"],
                            低于左档中位比例="%.6f" % below))

    if a.dryrun:
        step("★★空转已走完【每一个落盘点】")
    elif a.smoke:
        say("\n★★冒烟测结束 —— ★不落盘。"); return
    t = os.path.join(S37_OUT, "scoredist_SD.tsv")
    with open(t, "w", encoding="utf-8", newline="") as fh:
        fh.write("# scoredist_SD.tsv -- S37-24 件一：一行 = 一（套, 模型, 批）的分数五数。\n")
        fh.write("# ★★★★★不重训、★不改秤、★不含 verdict；★★★本表【不含任何 AUC】。\n")
        fh.write("#\n")
        fh.write("# ★★★件〇 裁定（Lang 2026-08-23）：★★**四条负例来源都不选，★AUC 本轮不算**。\n")
        fh.write("#    ★理由逐字：★四条各自的性质已列清 —— ★★**没有一条能给出【干净的跨库负例】**\n")
        fh.write("#      ⇒ ★★**算出来的数会被误读成『跨库 AUC』，★而它不是**\n")
        fh.write("#      ⇒ ★★**本轮要的答案由件一给：★不需要负例**\n")
        fh.write("#    ★四条的实测代价（[S37-24 核]）：\n")
        fh.write("#      ① El Zahed『野生型 EC50>50』3,747 唯一 IK ⇒ ★★[台账四十五] 已裁定不能当反例\n")
        fh.write("#         （终点是生长抑制，★分不清『进不去』与『不是抗生素』）\n")
        fh.write("#      ② 74k 的负类 ⇒ ★★**那不是外部的** ⇒ 算出来不是跨库 AUC\n")
        fh.write("#      ③ 外部集内部按 EC50 分档 ⇒ ★★测的是『能不能排出活性高低』，★不是『穿不穿透』\n")
        fh.write("#      ④ 二级那 28 个『必不穿透』⇒ ★见下条单独记录\n")
        fh.write("#\n")
        fh.write("# ★★候选④（二级那 28 个）★单独记一条 —— ★★**不用但不删**：\n")
        fh.write("#    · ★它是盘上★★**唯一非活性反推的负例**（★孔道层面实测硬证据：脂质体溶胀/通透系数）\n")
        fh.write("#    · ⚠️ ★n=28 ／ ★与 74k 训练集重叠 1 ／ ★★★**28/28 已在三级右档当负类用过**\n")
        fh.write("#         ／ ★与外部集 663 交集 **0**（★化学空间不相干）\n")
        fh.write("#    ⇒ ★★**记为『日后若拿到独立的孔道层面负例，★这是唯一可对照的锚』**\n")
        fh.write("# ★『批』三档：★74k左档正类（%d 个）／74k全体（模型 OOF）／★★外部集663。\n" % 0)
        fh.write("# ★『→在左档分位』= 每个外部集分子的分数在【左档正类分数分布】里的分位（0–1），★再报其五数。\n")
        fh.write("# ★口径与 [S37-23] 逐字相同：训练只用 74k（★外部集按 InChIKey 硬剔），★全 74k 拟合后给外部集打分。\n")
        fh.write("# 行数：%d\n" % len(OUT))
        cols = ["套", "模型", "批", "n", "min", "P25", "中位", "P75", "max", "低于左档中位比例"]
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in OUT]
    say("\n★落盘 %s  md5=%s  数据行=%d" % (os.path.relpath(t, HOME), md5(t), len(OUT)))
    mdp = os.path.join(S37_OUT, "scoredist_SD.md")
    with open(mdp, "w", encoding="utf-8") as fh:
        fh.write("# S37-24 件一 · 分数分布\n\n★★★不含任何 AUC（★件〇 未裁）。\n\n---\n\n```\n")
        fh.write("\n".join(LOG)); fh.write("\n```\n")
    say("★落盘 %s" % os.path.relpath(mdp, HOME))
    if a.dryrun:
        import shutil
        n = len(os.listdir(S37_OUT))
        step("★★空转：★★★通了 —— ★每个落盘点都执行过（★共 %d 个假产物）" % n)
        for f in sorted(os.listdir(S37_OUT)):
            say("     ★假产物 %s  %d 字节" % (f, os.path.getsize(os.path.join(S37_OUT, f))))
        shutil.rmtree(S37_OUT, ignore_errors=True)
        step("★★假产物已全部删除 ⇒ ★真盘上不留任何东西")
    say("\n★收工 %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # ★★★[S37-24 新增] ★异常必须写进日志末尾，★★不得静默死掉；★跑完写 DONE 标记
    _dry = "--dryrun" in sys.argv
    _done = os.path.join(S37, "s37u.DONE") if not _dry else os.devnull
    try:
        if os.path.exists(_done):
            os.remove(_done)
    except Exception:
        pass
    try:
        main()
        step("★★★★全部完成")
        if not _dry:
            with open(_done, "w", encoding="utf-8") as fh:
                fh.write("S37-24 DONE %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("★★★写标记 %s ⇒ ★有它就是跑完了" % _done, flush=True)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        step("⚠️★★★★崩溃 —— ★逐字写进日志末尾（★不静默死掉）")
        print(tb, flush=True)
        try:
            mdp = os.path.join(S37_OUT, "scoredist_SD.md")
            with open(mdp, "w", encoding="utf-8") as fh:
                fh.write("# S37-24 件一 · ★★★本轮【崩溃】，★未跑完\n\n```\n")
                fh.write("\n".join(LOG)); fh.write("\n\n" + tb + "\n```\n")
            print("★崩溃现场已写入 %s" % mdp, flush=True)
        except Exception:
            pass
        raise
