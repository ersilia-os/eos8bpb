#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S32-4 乙件 —— #15(HWI / Fsp3 / FPSA) 与 #16(硝基) 在水路标尺(Boi 41)上首测。

★本轮只算只报：不改任何既有实现、不重跑任何下游产物、★不下"该留该丢"的结论。
★不为了找到信号而挑口径 —— 全部读法都报，一个不挑。

============================ 构件定义 ============================
#15-a HWI (Hyper-Wiener Index)
  ★★复用既有实现，★一行未改：
    antibiotic-benchmark/scripts/c7_run_g_feat.py:163  hyper_wiener(mol, addH=..)
    其核心 c7_run_g_feat.py:139 _ww_one = [4] Klein 1995 的 WW = ½Σ_{i<j}(d+d²)
    [4] = Klein, Lukovits, Gutman, JCICS 35, 50-52 (1995), DOI 10.1021/ci00023a007
  默认口径（该模块 FEATURES 表第 1 行逐字）：重原子图、纯拓扑距离、最大片段。
  ★该口径已在 C7 那条线上经 5,000 分子真值验收（未定 13）。
  两个读法都给：
    hwi          = hyper_wiener(m)            重原子图（默认，真值验收的那套）
    hwi_addH     = hyper_wiener(m, addH=True) 加显式氢（该模块自带的对照读法）

#15-b Fsp3
  RDKit 标准实现：rdMolDescriptors.CalcFractionCSP3
  （本仓既有用法见 antibiotic-benchmark/scripts/c7_run_d_opensource.py:81）
  ★无 UNCLEAR：sp3 碳数 / 碳总数，定义唯一。

#15-c FPSA  ⚠️★★UNCLEAR —— 分数极性表面积的定义在文献里不唯一
  本仓★无既有实现。C5-2 提过的"与 clogP 解耦"是【外排侧的来源】，
  ★按任务书要求不作定义依据。
  ⇒ 以下三处一律标出，★每种读法都算，本表不选：
    UNCLEAR-1 分母取哪种表面积？
        本轮限 2D（无构象）⇒ RDKit 唯一的 2D 近似表面积 = LabuteASA。
        ★3D SASA(rdFreeSASA) 需要构象，本轮不做，故【未给该读法】—— 明说缺口。
    UNCLEAR-2 LabuteASA 含不含氢？ includeHs=True(RDKit 默认) / False 两种都给
    UNCLEAR-3 TPSA 计不计 S 和 P？ includeSandP=False(RDKit 默认) / True 两种都给
  ⇒ 2 × 2 = 4 个读法：fpsa_{tpsa 口径}_{分母口径}
  ⚠️ 另记：CPSA 家族(Stanton & Jurs)里的 "FPSA_n" 指的是【带电】分数表面积
     PPSA_n/SASA，与"极性表面积分数"不是同一个量。本表算的是后者，
     ★不声称与任何文献的 FPSA 同口径。

#16 硝基
  SMARTS 逐字（★复用既有，来源 antibiotic-benchmark/scripts/d1_control_saturation.py:23
  「Gurvic 原版硝基 SMARTS（词表第 58 条）」）：
    [$([#6][NX3](=[OX1])=[OX1]),$([#6][NX3+](=[OX1])[O-])]
  ★该 SMARTS 要求硝基氮连在碳上（[#6]），中性五价写法与电荷分离写法两支都收。
  芳香/脂肪的分法（★本项目自定，非文献）：看命中的那个【碳】是否芳香。
  三列都给：nitro_aromatic / nitro_aliphatic / nitro_total（前两者之和 = 后者）。

============================ 口径 ============================
★与 water_candidates_boi41.tsv 那轮完全一致，只换构件：
  输入 SMILES = 该表已落盘的 smiles_ph74 列，★不重跑 OpenBabel
  n = 40（guanosine 无 smiles_ph74，★单列不算，不推断填空）
  标签 = boi2025_fig3_class_final.tsv 的三档（阈值为 Boi 正文 §2.3 原值，未调）

============================ 评估 ============================
  全数据 AUC：_raw（未翻转）与 _oriented（=0.5+|raw-0.5|）两个都报
  CV-AUC：分层 5 折(按 perm_class)，种子 0..19 共 20 次，★符号折内定
  两种正类编码：编码① 正类=good；编码② bad vs 其余，方向统一为"越通透越正"
                ⇒ 正类=非bad(good+int)
  Spearman(构件值, 档次序数 good=3/int=2/bad=1) 与 p，标注 n
  ★分子抽样 bootstrap 95% CI：整行有放回重抽，B/seed 见 argparse 默认值，
    ★口径与 dxy_moments_eval.tsv(s31j) / water_candidates_boi41_stats.tsv(s31e) 同
  ★剔除 class_source=inferred 的敏感性版
    ⚠️ n=40 内 inferred 只剩 5 条（第 6 条是被剔除的 guanosine）⇒ 敏感性版 n=35
"""

import argparse
import csv
import io
import os
import sys
import time

import numpy as np
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors
from sklearn.model_selection import StratifiedKFold

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c7_run_g_feat import hyper_wiener            # noqa: E402  ★既有实现，一行未改

# ★Gurvic 原版硝基 SMARTS（词表第 58 条），逐字取自 d1_control_saturation.py:23
NITRO_SMARTS = "[$([#6][NX3](=[OX1])=[OX1]),$([#6][NX3+](=[OX1])[O-])]"

ORD = {"good": 3, "intermediate": 2, "bad": 1}

# (列名, 说明) —— ★全部读法，一个不挑
QUANTS = [
    ("hwi",                     "#15-a HWI 重原子图(默认,真值验收口径)"),
    ("hwi_addH",                "#15-a HWI 加显式氢(既有对照读法)"),
    ("fsp3",                    "#15-b Fsp3 = RDKit CalcFractionCSP3"),
    ("fpsa_tpsaDef_asaH",       "#15-c FPSA UNCLEAR: TPSA(默认)/LabuteASA(含H)"),
    ("fpsa_tpsaDef_asaNoH",     "#15-c FPSA UNCLEAR: TPSA(默认)/LabuteASA(不含H)"),
    ("fpsa_tpsaSP_asaH",        "#15-c FPSA UNCLEAR: TPSA(计S,P)/LabuteASA(含H)"),
    ("fpsa_tpsaSP_asaNoH",      "#15-c FPSA UNCLEAR: TPSA(计S,P)/LabuteASA(不含H)"),
    ("nitro_aromatic",          "#16 硝基-芳香(命中碳为芳香)"),
    ("nitro_aliphatic",         "#16 硝基-脂肪(命中碳非芳香)"),
    ("nitro_total",             "#16 硝基-合并"),
]


def read_tsv(path):
    """★不用 pandas 的 comment='#'：SMILES 里的三键 `#` 会被截断。只丢行首 '#' 的整行。"""
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines)), delimiter="\t"))


def compute(smiles):
    """一个分子的全部 #15/#16 读法。算不出返回 (None, 原因)。"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, "RDKit 解析失败"
    out = {}
    try:
        out["hwi"] = hyper_wiener(m)
        out["hwi_addH"] = hyper_wiener(m, addH=True)
    except Exception as e:
        return None, "hyper_wiener 抛异常: %s" % e
    out["fsp3"] = float(rdMolDescriptors.CalcFractionCSP3(m))

    tpsa_def = float(rdMolDescriptors.CalcTPSA(m))
    tpsa_sp = float(rdMolDescriptors.CalcTPSA(m, includeSandP=True))
    asa_h = float(rdMolDescriptors.CalcLabuteASA(m, includeHs=True))
    asa_noh = float(rdMolDescriptors.CalcLabuteASA(m, includeHs=False))
    for tn, tv in (("tpsaDef", tpsa_def), ("tpsaSP", tpsa_sp)):
        for an, av in (("asaH", asa_h), ("asaNoH", asa_noh)):
            out["fpsa_%s_%s" % (tn, an)] = (tv / av) if av else None
    out["_tpsa_def"], out["_tpsa_sp"] = tpsa_def, tpsa_sp
    out["_asa_h"], out["_asa_noh"] = asa_h, asa_noh

    patt = Chem.MolFromSmarts(NITRO_SMARTS)
    hits = m.GetSubstructMatches(patt)          # 命中的是那个【碳】原子
    ar = sum(1 for h in hits if m.GetAtomWithIdx(h[0]).GetIsAromatic())
    out["nitro_aromatic"] = float(ar)
    out["nitro_aliphatic"] = float(len(hits) - ar)
    out["nitro_total"] = float(len(hits))
    return out, ""


# ------------------------- 评估（口径与 s31j 一致） -------------------------
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int)
    p, n = x[y == 1], x[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([p, n]))
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n))


def cv_auc(x, y, strat, k_folds, seeds):
    """分层 K 折 × 多种子。★符号方向在训练折内定，绝不用全数据定方向。"""
    vals = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=s)
        fa = []
        for tr, te in skf.split(np.zeros(len(x)), strat):
            a_tr = auc(x[tr], y[tr])
            sign = 1.0 if (not np.isfinite(a_tr) or a_tr >= 0.5) else -1.0
            a_te = auc(sign * x[te], y[te])
            if np.isfinite(a_te):
                fa.append(a_te)
        if fa:
            vals.append(float(np.mean(fa)))
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1))


def boot_ci(fn, n, rng, b_boot, *arrays):
    """★整行有放回重抽的 95% CI —— 与 s31e_water_stats.boot_ci / s31j 同口径。"""
    vals = []
    for _ in range(b_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrays])
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else "%.10g" % v
    return str(v)


def main(a):
    t0 = time.time()
    water = read_tsv(a.water)
    cls_final = {r["name"]: r for r in read_tsv(a.class_final)
                 if r.get("perm_class", "") in ORD}
    long_boi = {r["mol_id"]: r for r in read_tsv(a.long) if r["dataset"] == "boi"}

    rows = [r for r in water if r.get("smiles_ph74", "")]
    miss = [r["name"] for r in water if not r.get("smiles_ph74", "")]
    print("=" * 78)
    print("[读入] %s: %d 行；有 smiles_ph74 的 %d 行" % (os.path.basename(a.water),
                                                        len(water), len(rows)))
    print("[缺失] %d 条无 smiles_ph74，★单列不算、不推断填空：%s"
          % (len(miss), "; ".join(miss)))
    print("[标签] %s: %d 条三档标签" % (os.path.basename(a.class_final), len(cls_final)))

    cls_of, src_of = {}, {}
    for r in rows:
        c = cls_final.get(r["name"])
        if c is None:
            continue
        cls_of[r["name"]] = c["perm_class"]
        src_of[r["name"]] = c.get("class_source", "")
    print("[对齐] 三档标签命中 %d / %d；class_source=inferred %d 条"
          % (len(cls_of), len(rows),
             sum(1 for n in cls_of if src_of[n] == "inferred")))

    # ---------------- 计算 ----------------
    vals, fails = {}, []
    for r in rows:
        out, why = compute(r["smiles_ph74"])
        if out is None:
            fails.append((r["name"], why))
        vals[r["name"]] = out
    if fails:
        print("[失败] %d 条算不出（★单列，不推断填空）：" % len(fails))
        for n, why in fails:
            print("    %-30s %s" % (n, why))
    else:
        print("[计算] %d 个分子全部算出，无失败" % len(rows))

    # ---------------- 落盘构件值 ----------------
    vcols = (["name", "perm_class", "class_source", "smiles_ph74"]
             + [k for k, _ in QUANTS]
             + ["_tpsa_def", "_tpsa_sp", "_asa_h", "_asa_noh", "fail"])
    out1 = os.path.join(a.outdir, a.values_name)
    with open(out1, "w", encoding="utf-8", newline="") as fh:
        fh.write("# %s -- #15(HWI/Fsp3/FPSA) 与 #16(硝基) 在水路标尺(Boi 41)上的构件值\n"
                 "# 生成 %s，脚本 scripts/s31m_water_extra.py\n"
                 "# ★本表只算不判：不含 AUC/阈值/取舍，也不含『该留该丢』的结论。\n"
                 "#\n"
                 "# ========== 只读输入（未修改）==========\n"
                 "#   %s   -> name/perm_class/class_source/smiles_ph74\n"
                 "#   %s   -> 三档标签\n"
                 "#   scripts/c7_run_g_feat.py -> hyper_wiener()（★import，一行未改）\n"
                 "#\n"
                 "# ========== 口径 ==========\n"
                 "#   输入串 = smiles_ph74（该轮 OB AddHydrogens(F,T,7.4) 的输出），"
                 "★不重跑 OpenBabel\n"
                 "#   hwi        = c7_run_g_feat.hyper_wiener(m)  [Klein 1995 ½Σ(d+d²)，"
                 "重原子图、最大片段]\n"
                 "#   hwi_addH   = 同上但加显式氢（该模块自带对照读法）\n"
                 "#   fsp3       = RDKit rdMolDescriptors.CalcFractionCSP3\n"
                 "#   fpsa_*     = ⚠️★UNCLEAR，四个读法全给，本表不选：\n"
                 "#                分子 TPSA{默认 / includeSandP=True}\n"
                 "#                分母 LabuteASA{includeHs=True / False}\n"
                 "#                ★3D SASA 读法需要构象，本轮未做 —— 缺口明说\n"
                 "#   nitro_*    = SMARTS %s\n"
                 "#                （Gurvic 词表第 58 条，逐字取自 d1_control_saturation.py:23）\n"
                 "#                芳香/脂肪按命中碳是否芳香分，★该分法为本项目自定\n"
                 "#   _tpsa_*/_asa_* 为 FPSA 的分子与分母原值，留档备查\n"
                 "#   算不出的分子在 fail 列注明，对应列留空，★不推断填空\n"
                 "# ================================================================\n"
                 % (a.values_name, time.strftime("%Y-%m-%d %H:%M:%S"),
                    a.water, a.class_final, NITRO_SMARTS))
        fh.write("\t".join(vcols) + "\n")
        for r in rows:
            v = vals.get(r["name"]) or {}
            d = dict(name=r["name"], perm_class=cls_of.get(r["name"], ""),
                     class_source=src_of.get(r["name"], ""),
                     smiles_ph74=r["smiles_ph74"],
                     fail=dict(fails).get(r["name"], ""))
            d.update({k: v.get(k) for k in vcols if k not in d})
            fh.write("\t".join(fmt(d.get(c)) for c in vcols) + "\n")
    print("[写出] %s  (%d 行)" % (out1, len(rows)))

    # ---------------- 评估 ----------------
    ecols = ["component", "reading", "subset", "n", "n_good", "n_int", "n_bad",
             "full_auc_enc1_raw", "full_auc_enc1_oriented",
             "ci95_enc1_lo", "ci95_enc1_hi", "ci_width_enc1", "ci_crosses_0.5_enc1",
             "full_auc_enc2_raw", "full_auc_enc2_oriented",
             "ci95_enc2_lo", "ci95_enc2_hi", "ci_width_enc2", "ci_crosses_0.5_enc2",
             "cv_auc_enc1_mean", "cv_auc_enc1_sd",
             "cv_auc_enc2_mean", "cv_auc_enc2_sd",
             "rho_raw", "rho_p", "rho_ci_lo", "rho_ci_hi",
             "n_distinct", "note"]
    ev = []
    seeds = list(range(a.cv_seeds))
    for col, desc in QUANTS:
        for subset in ("全部", "剔除inferred"):
            use, xs = [], []
            for r in rows:
                n = r["name"]
                if n not in cls_of:
                    continue
                if subset != "全部" and src_of[n] == "inferred":
                    continue
                v = (vals.get(n) or {}).get(col)
                if v is None or not np.isfinite(v):
                    continue
                use.append(n); xs.append(float(v))
            if len(use) < 10:
                continue
            x = np.array(xs)
            o = np.array([ORD[cls_of[n]] for n in use], float)
            strat = np.array([cls_of[n] for n in use])
            y1 = (o == 3).astype(int)
            y2 = (o > 1).astype(int)
            nd = len(set(x.tolist()))
            note = []
            if nd < 2:
                note.append("★常数列，AUC/rho 无定义")
            elif nd <= 2:
                note.append("二值列(并列多)")
            if nd < 2:
                ev.append(dict(component=desc.split()[0], reading=col, subset=subset,
                               n=len(use), n_good=int((o == 3).sum()),
                               n_int=int((o == 2).sum()), n_bad=int((o == 1).sum()),
                               n_distinct=nd, note="；".join(note)))
                continue
            a1, a2 = auc(x, y1), auc(x, y2)
            lo1, hi1 = boot_ci(auc, len(x), np.random.default_rng(a.boot_seed),
                               a.boot, x, y1)
            lo2, hi2 = boot_ci(auc, len(x), np.random.default_rng(a.boot_seed),
                               a.boot, x, y2)
            m1, s1 = cv_auc(x, y1, strat, a.k_folds, seeds)
            m2, s2 = cv_auc(x, y2, strat, a.k_folds, seeds)
            rho, p = stats.spearmanr(x, o)
            rlo, rhi = boot_ci(lambda p_, q_: stats.spearmanr(p_, q_)[0],
                               len(x), np.random.default_rng(a.boot_seed), a.boot, x, o)
            ev.append(dict(
                component=desc.split()[0], reading=col, subset=subset, n=len(use),
                n_good=int((o == 3).sum()), n_int=int((o == 2).sum()),
                n_bad=int((o == 1).sum()),
                full_auc_enc1_raw=a1, full_auc_enc1_oriented=0.5 + abs(a1 - 0.5),
                ci95_enc1_lo=lo1, ci95_enc1_hi=hi1, ci_width_enc1=hi1 - lo1,
                **{"ci_crosses_0.5_enc1": "TRUE" if (lo1 <= 0.5 <= hi1) else "FALSE"},
                full_auc_enc2_raw=a2, full_auc_enc2_oriented=0.5 + abs(a2 - 0.5),
                ci95_enc2_lo=lo2, ci95_enc2_hi=hi2, ci_width_enc2=hi2 - lo2,
                **{"ci_crosses_0.5_enc2": "TRUE" if (lo2 <= 0.5 <= hi2) else "FALSE"},
                cv_auc_enc1_mean=m1, cv_auc_enc1_sd=s1,
                cv_auc_enc2_mean=m2, cv_auc_enc2_sd=s2,
                rho_raw=rho, rho_p=p, rho_ci_lo=rlo, rho_ci_hi=rhi,
                n_distinct=nd, note="；".join(note)))

    out2 = os.path.join(a.outdir, a.eval_name)
    with open(out2, "w", encoding="utf-8", newline="") as fh:
        fh.write("# %s -- #15/#16 的评估。构件值与口径见 %s 表头。\n"
                 "# CV-AUC：分层 %d 折 × %d 种子(0..%d)，★符号折内定。\n"
                 "# 编码① 正类=good；编码② bad vs 其余，方向统一为『越通透越正』"
                 "⇒ 正类=非bad。\n"
                 "# full_auc_*_raw 未翻转（<0.5 表示反向）；_oriented = 0.5+|raw-0.5|。\n"
                 "# CI = ★分子抽样 bootstrap，整行有放回重抽 B=%d, seed=%d，"
                 "百分位 2.5/97.5\n"
                 "#      ★口径与 dxy_moments_eval.tsv(s31j) 及 "
                 "water_candidates_boi41_stats.tsv(s31e) 相同。\n"
                 "# ★ci_crosses_0.5 = TRUE 表示该读法在本样本上【没有方向证据】。\n"
                 "# ★所有读法 × 2 编码 × 2 子集全报，一个不挑。\n"
                 % (a.eval_name, a.values_name, a.k_folds, a.cv_seeds,
                    a.cv_seeds - 1, a.boot, a.boot_seed))
        fh.write("\t".join(ecols) + "\n")
        for r in ev:
            fh.write("\t".join(fmt(r.get(c)) for c in ecols) + "\n")
    print("[写出] %s  (%d 行)" % (out2, len(ev)))

    # ---------------- 控制台 ----------------
    print("=" * 78)
    for subset in ("全部", "剔除inferred"):
        sub = [r for r in ev if r["subset"] == subset]
        if not sub:
            continue
        print("--- subset=%s ---" % subset)
        print("  %-24s %4s %9s %-20s %-6s %14s %9s"
              % ("读法", "n", "全AUC①", "CI95①", "跨0.5", "CV-AUC①±sd", "rho"))
        for r in sub:
            if r.get("full_auc_enc1_raw") is None:
                print("  %-24s %4d  ★%s" % (r["reading"], r["n"], r["note"]))
                continue
            print("  %-24s %4d %9.3f [%.3f, %.3f]   %-6s %6.3f±%-7.3f %+9.3f"
                  % (r["reading"], r["n"], r["full_auc_enc1_raw"],
                     r["ci95_enc1_lo"], r["ci95_enc1_hi"], r["ci_crosses_0.5_enc1"],
                     r["cv_auc_enc1_mean"], r["cv_auc_enc1_sd"], r["rho_raw"]))

    # ---------------- 核心一问 ----------------
    print("=" * 78)
    print("★★核心一问：这几个量里，有没有任何一个的 bootstrap CI 不横跨 0.5？")
    hit = [r for r in ev if r.get("ci_crosses_0.5_enc1") == "FALSE"
           or r.get("ci_crosses_0.5_enc2") == "FALSE"]
    if not hit:
        print("  ★★答案：没有。%d 个 (读法 × 子集) 组合的两种编码 CI 【全部横跨 0.5】。"
              % len([r for r in ev if r.get("full_auc_enc1_raw") is not None]))
    else:
        print("  找到 %d 个 CI 不横跨 0.5 的组合：" % len(hit))
        # 与 mol_size 的相关（★与 dxy 那轮同一道尺寸代理检验）
        ms = {}
        for n in cls_of:
            lr = long_boi.get(n)
            if lr and lr.get("mol_size", ""):
                ms[n] = float(lr["mol_size"])
        for r in sorted(hit, key=lambda r: -abs(r["full_auc_enc1_raw"] - 0.5)):
            col = r["reading"]
            pair = [(vals[n][col], ms[n]) for n in cls_of
                    if n in ms and (vals.get(n) or {}).get(col) is not None]
            if len(pair) >= 5:
                rr, pp = stats.spearmanr([p[0] for p in pair], [p[1] for p in pair])
                mstr = "vs mol_size rho=%+.3f (p=%.3g, n=%d)" % (rr, pp, len(pair))
            else:
                mstr = "vs mol_size 无法算(可比分子 <5)"
            print("   %-24s %-12s n=%d  enc1 %.3f CI[%.3f,%.3f]%s | enc2 %.3f "
                  "CI[%.3f,%.3f]%s\n        %s"
                  % (col, r["subset"], r["n"],
                     r["full_auc_enc1_raw"], r["ci95_enc1_lo"], r["ci95_enc1_hi"],
                     "★" if r["ci_crosses_0.5_enc1"] == "FALSE" else "",
                     r["full_auc_enc2_raw"], r["ci95_enc2_lo"], r["ci95_enc2_hi"],
                     "★" if r["ci_crosses_0.5_enc2"] == "FALSE" else "", mstr))
    print("[耗时] %.1f s" % (time.time() - t0))
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        description="S32-4 乙件：#15(HWI/Fsp3/FPSA) 与 #16(硝基) 在水路标尺上首测")
    p.add_argument("--water", default="downloads_raw/boi2025/water_candidates_boi41.tsv",
                   help="构件值来源表，提供 name/smiles_ph74（只读）")
    p.add_argument("--class-final",
                   default="downloads_raw/boi2025/boi2025_fig3_class_final.tsv",
                   help="三档标签表（只读）")
    p.add_argument("--long",
                   default="antibiotic-benchmark/results/section31_redundancy/"
                           "s31_features_long.tsv",
                   help="提供 mol_size 用于尺寸代理检验（只读）")
    p.add_argument("--outdir", default="downloads_raw/boi2025", help="产物目录")
    p.add_argument("--values-name", default="water_extra_15_16.tsv", help="构件值表文件名")
    p.add_argument("--eval-name", default="water_extra_15_16_eval.tsv", help="评估表文件名")
    p.add_argument("--boot", type=int, default=10000, help="bootstrap 重抽次数")
    p.add_argument("--boot-seed", type=int, default=42, help="bootstrap 种子")
    p.add_argument("--k-folds", type=int, default=5, help="分层 CV 折数")
    p.add_argument("--cv-seeds", type=int, default=20, help="重复 CV 的种子数(0..N-1)")
    return p


if __name__ == "__main__":
    sys.exit(main(build_parser().parse_args()))
