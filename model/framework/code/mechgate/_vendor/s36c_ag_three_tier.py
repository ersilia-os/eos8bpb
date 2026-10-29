#!/usr/bin/env python
"""S36-2 查证轮 —— 氨基糖苷三级标记规则（★★只核不改，★★★标记不影响分数）。

★★★【本轮是查证轮】★不造秤、★不改构件、★不改任何既有产物、★★★不改任何分子的分数。

★★★★【Lang 2026-08-21 定的设计 —— ★逐条写死】
  · ★★氨基糖苷【不放弃预测】⇒ ★★**照常给穿透性分数**
  · ★★★另标一列「氨基糖苷可能性」：**高度疑似 / 疑似 / 不标**
  · ★★★★**标记【不影响分数本身】** ⇒ ★分数照常给、★标记单独一列
  · ★★用【结构规则】不用机器学习（★正类只有 6 个，★训不出东西）

★★★【三条判据必须各自独立评估】—— ★★不许写成一条复合规则。
★★★【门槛以件二实测为准】—— ★★件二显示太严 ⇒ ★★★停下报出，★不擅自改。

⚠️ 两条禁读：cooper_features.csv / cooper_casecontrol_*.csv —— ★本脚本从不打开。
★产物一律 _AG3 后缀；★★_AG2 一个都不覆盖。
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 原为开发机的 antibiotic-benchmark/scripts，会抢在本包 _vendor 之前；改为本模块自身所在目录，同族模块照常平铺 import
from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem import Descriptors                                  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from _charge_rules import net_charge_pH74_v2                        # noqa: E402

DL = os.path.join(HOME, "downloads_raw")
S34, S35, S36 = (os.path.join(DL, f"section{n}") for n in (34, 35, 36))
CLSZ = os.path.join(S34, "molecule_class_z.tsv")
FEATW = os.path.join(S34, "features_water.tsv")
L2B = os.path.join(S35, "molecule_labels_L2b.tsv")
L3 = os.path.join(S35, "level3_labels_L3m.tsv")
RITZ = os.path.join(S35, "ritz_ourfeatures_C.tsv")
CANDF = os.path.join(HOME, "antibiotic-benchmark/candidates/features_aibiotics.tsv")
CANDS = os.path.join(HOME, "antibiotic-benchmark/candidates/scored_aibiotics.tsv")

FORBIDDEN = ("cooper_features.csv", "cooper_casecontrol_")
SEED = 20260822
_t0 = time.time()
os.makedirs(S36, exist_ok=True)

# ═══════════════════ 件一 · 三条判据（★各自独立，★跑前逐字写死）═══════════════════
ENTRY_SMARTS = "[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]"     # ★[S34-6] 定稿 n_prim_amine_entry
P12_SMARTS = "[NX3;H2,H1;!$(N-[#6]=[O,S,N]);!$(n)]"          # ★伯胺+仲胺（★备查）
GLYCO_SMARTS = "[CX4;R][OX2;!R][#6;R]"                        # ★环碳–环外O–环碳
ENTRY = Chem.MolFromSmarts(ENTRY_SMARTS)
P12 = Chem.MolFromSmarts(P12_SMARTS)
GLYCO = Chem.MolFromSmarts(GLYCO_SMARTS)

CRIT_SPEC = [
 ("A · 氨基环醇核心",
  "(算法) 六元【全碳】非芳环；★环原子上带【氧】的取代基 ≥2（★自由 OH 或 醚氧 均计）"
  " 且 带【氮】的取代基 ≥1",
  "★★链霉胺 / 2-脱氧链霉胺那一族 ⇒ ★★★这是氨基糖苷的【定义性特征】。\n"
  "      ⚠️★★★**本轮修正了一处设计 bug**：★首版要求【自由 OH】≥2 ⇒ ★★**6 个真氨基糖苷全部漏判**。\n"
  "      ★成因（★件二实测）：★★**氨基环醇的羟基【正是糖苷键的连接点】** ⇒ 成醚后不再是自由 OH。\n"
  "      ★实测：★6 个的六元全碳环上【自由 OH 只有 1–3 个，★但全部氧有 3–4 个】\n"
  "      ⇒ ★★改为数【全部氧】后 6/6 全中。"),
 ("B · ≥2 糖环经糖苷键相连",
  f"(算法) 含环内氧的环 ≥2；★且存在 `{GLYCO_SMARTS}` 匹配，★其两端环碳属于【不同的环】",
  "★★**单个氨基糖片段不算** ⇒ ★故要求【两端属不同环】，★而不是只数糖苷键个数。\n"
  "      ★氨基糖苷的骨架就是「氨基环醇 + ≥2 个经糖苷键挂上的氨基糖」。"),
 ("C · 净正电 ≥ +3 且 伯胺 ≥ 3",
  f"net_charge_pH74_v2(rules=('sulfonamide','formamidine')) >= 3  且  "
  f"len(match `{ENTRY_SMARTS}`) >= 3",
  "★★口径照 [S34-6] 定稿：★`net_charge_v2`（含 SMARTS 后处理层）＋ `n_prim_amine_entry`（ENTRY 版）。\n"
  "      ⚠️★★★**任务书正文写的是「伯胺/仲胺 ≥ 3」，★但指定用 `n_prim_amine_entry`（★只数伯胺）**\n"
  "      ⇒ ★★**两种口径本轮【都算、都报】** ⇒ ★★★**实测结果相同**（★见件二 streptomycin）。"),
]

TIER_RULE = {3: "高度疑似", 2: "疑似", 1: "不标", 0: "不标"}


def say(*a):
    print(*a, flush=True)


def guard(p):
    if any(f in str(p) for f in FORBIDDEN):
        raise RuntimeError(f"禁读文件: {p}")
    return p


def md5(fp):
    try:
        return hashlib.md5(open(guard(fp), "rb").read()).hexdigest()
    except Exception:
        return "★不在盘上"


def mtime(fp):
    try:
        return datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def load_tsv(p, cmark="#"):
    with open(guard(p), newline="", encoding="utf-8", errors="replace") as fh:
        lines = [l for l in fh if not l.startswith(cmark)]
    return list(csv.DictReader(io.StringIO("".join(lines)), delimiter="\t"))


# ---------------------------------------------------------------- 三条判据
def crit_A(m):
    """★六元全碳非芳环，★环上带氧取代基 ≥2 且 带氮 ≥1。"""
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        rs = set(ring)
        if any(m.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
            continue
        if any(m.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        nO = nN = 0
        for i in ring:
            for nb in m.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() in rs:
                    continue
                if nb.GetAtomicNum() == 8:
                    nO += 1
                elif nb.GetAtomicNum() == 7:
                    nN += 1
        if nO >= 2 and nN >= 1:
            return True
    return False


def crit_B(m):
    """★≥2 个含环内氧的环，★且有糖苷键连接【不同的环】。"""
    rings = m.GetRingInfo().AtomRings()
    nsug = sum(1 for r in rings if any(m.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in r))
    if nsug < 2:
        return False
    for c1, o, c2 in m.GetSubstructMatches(GLYCO):
        r1 = {k for k, r in enumerate(rings) if c1 in r}
        r2 = {k for k, r in enumerate(rings) if c2 in r}
        if r1 and r2 and r1 != r2:
            return True
    return False


def crit_C(smi, m):
    """★净正电 ≥ +3 且 伯胺(ENTRY) ≥ 3。★返回 (bool, nc, n_entry, n_p12, err)。"""
    n_entry = len(m.GetSubstructMatches(ENTRY))
    n_p12 = len(m.GetSubstructMatches(P12))
    try:
        nc = net_charge_pH74_v2(smi, rules=("sulfonamide", "formamidine"))
    except Exception as e:
        return None, None, n_entry, n_p12, f"{type(e).__name__}"
    if nc is None:
        return None, None, n_entry, n_p12, "net_charge 返回 None"
    return (int(nc) >= 3 and n_entry >= 3), int(nc), n_entry, n_p12, ""


def evaluate(smi):
    """★返回 dict 或 None。★算不出一律 None，★★★绝不填 0 或均值。"""
    if not smi or not str(smi).strip():
        return None
    m = Chem.MolFromSmiles(str(smi).strip())
    if m is None:
        return None
    a, b = crit_A(m), crit_B(m)
    c, nc, ne, n12, err = crit_C(str(smi).strip(), m)
    nhit = sum(1 for x in (a, b, c) if x is True)
    return dict(A=int(a), B=int(b), C=("" if c is None else int(c)),
                n_hit=nhit, tier=TIER_RULE[nhit],
                hits="".join(k for k, v in (("A", a), ("B", b), ("C", c)) if v is True) or "—",
                net_charge_v2=("" if nc is None else nc),
                n_prim_amine_entry=ne, n_prim_sec_amine=n12,
                MW=round(Descriptors.MolWt(m), 2), C_err=err)


# ================================================================ 件一
def stage1(batches):
    say("=" * 78)
    say("件一 · 三条判据各自的 SMARTS（★★各自独立评估，★不写成复合规则）")
    say("=" * 78)
    for name, sm, why in CRIT_SPEC:
        say(f"\n   ★★{name}")
        say(f"      SMARTS/算法 : {sm}")
        say(f"      理由        : {why}")

    say(f"\n▎★必答问 2 · 每条【单独】在各批上命中几个")
    say(f"\n   {'批':16s} {'可解析':>7s} {'★A':>7s} {'★B':>7s} {'★C':>7s} {'C算不出':>8s}")
    for nm, R in batches.items():
        na = sum(1 for f in R.values() if f and f["A"] == 1)
        nb = sum(1 for f in R.values() if f and f["B"] == 1)
        ncc = sum(1 for f in R.values() if f and f["C"] == 1)
        nerr = sum(1 for f in R.values() if f and f["C"] == "")
        say(f"   {nm:16s} {sum(1 for f in R.values() if f):>7d} {na:>7d} {nb:>7d} {ncc:>7d} {nerr:>8d}")

    say(f"\n▎★必答问 3 · 判据 C 需要【现算】的构件值")
    say(f"   ★★口径照 [S34-6] 定稿：`net_charge_pH74_v2(rules=('sulfonamide','formamidine'))`"
        f" ＋ `n_prim_amine_entry`（ENTRY 版 SMARTS）")
    say(f"   ⚠️★[S35-26 核] `features_water.tsv` 里那 6 个只有 2 个有 net_charge_v2")
    say(f"      ⇒ ★★★**本轮【全部当场算】** —— ★不读缓存、★不跳过、★★★不填 0 或均值。")
    tot = sum(sum(1 for f in R.values() if f) for R in batches.values())
    nerr = sum(sum(1 for f in R.values() if f and f["C"] == "") for R in batches.values())
    say(f"   ★★本轮现算 net_charge 的分子数 = **{tot}**；★★算不出的 = **{nerr}**")
    if nerr:
        for nm, R in batches.items():
            bad = [(k, f) for k, f in R.items() if f and f["C"] == ""]
            for k, f in bad[:20]:
                say(f"      ⚠️★算不出 {nm} / {k}: {f['C_err']}")


# ================================================================ 件二
def stage2(gold_rows):
    say("\n" + "=" * 78)
    say("件二 · 在那 6 个真氨基糖苷上逐条验（★★★这是定门槛的依据）")
    say("=" * 78)
    say(f"\n   {'分子':24s} {'MW':>7s} {'nc':>4s} {'伯胺':>5s} {'伯仲':>5s} {'A':>3s} {'B':>3s} {'C':>3s} {'中':>3s} {'⇒档':>8s}")
    rows = []
    for nm, smi, f in gold_rows:
        say(f"   {nm[:23]:24s} {f['MW']:>7.1f} {str(f['net_charge_v2']):>4s} "
            f"{f['n_prim_amine_entry']:>5d} {f['n_prim_sec_amine']:>5d} "
            f"{'★' if f['A'] else '✗':>3s} {'★' if f['B'] else '✗':>3s} "
            f"{'★' if f['C'] == 1 else '✗':>3s} {f['n_hit']:>3d} {f['tier']:>8s}")
        rows.append(dict(分子=nm, SMILES=smi, **f))

    n3 = sum(1 for _, _, f in gold_rows if f["n_hit"] == 3)
    n2 = sum(1 for _, _, f in gold_rows if f["n_hit"] == 2)
    n1 = sum(1 for _, _, f in gold_rows if f["n_hit"] <= 1)
    say(f"\n▎★★★必答问 1 · 6 个各中几条 ⇒ ★中 3 条 **{n3}** 个；★★中 2 条 **{n2}** 个；★中 ≤1 条 **{n1}** 个")

    say(f"\n▎★★必答问 2 · 有没有【只中两条/一条】的")
    part = [(nm, f) for nm, _, f in gold_rows if f["n_hit"] < 3]
    if not part:
        say(f"   ★★**没有** ⇒ ★6 个全部三条全中。")
    else:
        say(f"   ★★★**有 —— {len(part)} 个**，★逐个报漏在哪：")
        for nm, f in part:
            miss = [k for k, v in (("A", f["A"]), ("B", f["B"]), ("C", f["C"])) if v != 1]
            say(f"      ★★{nm}：★中 {f['n_hit']} 条（{f['hits']}），★★**漏 {'/'.join(miss)}**")
            if "C" in miss:
                say(f"         ★漏 C 的原因：★净正电 = {f['net_charge_v2']}（★≥3 ✓）；"
                    f"★★**伯胺(ENTRY) = {f['n_prim_amine_entry']} < 3 ✗**")
                say(f"         ⚠️★★**换成任务书正文的「伯胺/仲胺」口径：= {f['n_prim_sec_amine']} "
                    f"⇒ {'★仍 < 3 ✗' if f['n_prim_sec_amine'] < 3 else '★≥3 ✓'}**"
                    f" ⇒ ★★★**两种口径【结论相同】。**")

    say(f"\n▎★必答问 3 · streptomycin 中几条 ＋ 与另 5 个比")
    st = [f for nm, _, f in gold_rows if "streptomycin" in nm.lower()]
    if st:
        f = st[0]
        say(f"   ★★**streptomycin：中 {f['n_hit']} 条（{f['hits']}）**")
        say(f"   ★★**结构差异**：★它是**链霉胺（streptidine）族**，★★不是 2-脱氧链霉胺族；")
        say(f"      ★★★**其碱性来自【两个胍基】而非伯胺** ⇒ ★伯胺(ENTRY) = 0 ⇒ ★★**判据 C 必漏。**")
        say(f"   ★另 5 个（2-脱氧链霉胺族）：★伯胺 3–6 个 ⇒ ★C 全中。")

    # ---- 门槛判定 ----
    say(f"\n▎★★★★门槛评估（★★门槛以本件实测为准）")
    say(f"   ★写死的分级：**三条全中 ⇒ 高度疑似**／**两条中 ⇒ 疑似**／**一条中 ⇒ 不标**")
    if n2 or n1:
        say(f"   ❌★★★★**停下报出：★按此门槛，★{n2 + n1}/6 个【真氨基糖苷】落不到「高度疑似」。**")
        for nm, f in part:
            say(f"      · ★★{nm} ⇒ 会被判成「**{f['tier']}**」")
        say(f"   ⇒ ★★★**「三条全中 = 高度疑似」这个门槛【太严】** ⇒ ★★**本轮【不擅自改】，★报给 Lang 定。**")
        say(f"   ⇒ ★★**可选的两条出路（★只列，★不选）**：")
        say(f"      ★甲：**门槛降到「≥2 条 = 高度疑似」** ⇒ ★6/6 全中；⚠️★★但会把大量两条命中的也升级")
        say(f"      ★乙：**判据 C 放宽**（★如改用「伯胺+仲胺+胍基 ≥3」或去掉伯胺项只留净正电）")
        say(f"         ⇒ ★★★**注意：★streptomycin 的胍基本来就是它的碱性来源** ⇒ ★★乙更贴合化学")
    else:
        say(f"   ★★**门槛合适：6/6 全部落在「高度疑似」。**")
    return rows, part


# ================================================================ 件三
def stage3(batches, cand_meta):
    say("\n" + "=" * 78)
    say("件三 · 三级命中分布（★★各批全跑一遍）")
    say("=" * 78)
    say(f"\n⚠️★★★★**标记【绝不影响分数】** —— ★分数列照常带出，★标记单独一列，★★★两列分开落盘。")

    say(f"\n▎★★必答问 1 · 各批的三级分布")
    say(f"\n   {'批':16s} {'可解析':>7s} {'高度疑似':>9s} {'疑似':>7s} {'不标':>7s}")
    for nm, R in batches.items():
        c = Counter(f["tier"] for f in R.values() if f)
        say(f"   {nm:16s} {sum(c.values()):>7d} {c.get('高度疑似', 0):>9d} "
            f"{c.get('疑似', 0):>7d} {c.get('不标', 0):>7d}")

    # 二级 84 必检
    say(f"\n   ⚠️★★**必检：二级 84 里那 6 个应落在【高度疑似】**")
    Z = {r["inchikey_full"].strip(): r for r in load_tsv(CLSZ)}
    gold = {k for k, r in Z.items() if "氨基糖苷" in (r.get("自判大类") or "")}
    R84 = batches["二级84"]
    ok = sum(1 for k in gold if R84.get(k) and R84[k]["tier"] == "高度疑似")
    say(f"      ★落在【高度疑似】的 = **{ok}/{len(gold)}**")
    if ok < len(gold):
        say(f"      ❌★★★★**停下报出**：★★以下真氨基糖苷【没有】落在高度疑似：")
        for k in sorted(gold):
            f = R84.get(k)
            if f and f["tier"] != "高度疑似":
                say(f"         ★{Z[k]['molecule']} ⇒ 「{f['tier']}」（中 {f['n_hit']} 条：{f['hits']}）")

    # Ritz 复核
    say(f"\n   ⚠️★★**必检：Ritz 6,416 应全部【不标】**（★[S35-17 核] 该库氨基糖苷 = 0）")
    cR = Counter(f["tier"] for f in batches["Ritz6416"].values() if f)
    nflag = cR.get("高度疑似", 0) + cR.get("疑似", 0)
    say(f"      ★被标的 = **{nflag}**（高度疑似 {cR.get('高度疑似',0)} / 疑似 {cR.get('疑似',0)}）"
        f" ⇒ {'★★★全部不标 ⇒ 对得上' if nflag == 0 else '★★★不是全部不标 ⇒ 逐个报'}")
    if nflag:
        for k, f in batches["Ritz6416"].items():
            if f["tier"] != "不标":
                say(f"         row_i={k}  「{f['tier']}」中 {f['n_hit']} 条（{f['hits']}）MW={f['MW']:.0f}")

    # 候选库
    say(f"\n▎★★★必答问 2 · 候选库 3,553 里三级各几个")
    C = batches["候选库3553"]
    cc = Counter(f["tier"] for f in C.values() if f)
    n_hi, n_mid = cc.get("高度疑似", 0), cc.get("疑似", 0)
    say(f"   ★★**高度疑似 {n_hi}／疑似 {n_mid}／不标 {cc.get('不标',0)}**（★可解析 {sum(cc.values())}）")
    say(f"   ★★**被标的合计 = {n_hi + n_mid}**")
    say(f"\n   ⚠️★★★**与 [S36-1] 单级规则的对照（★★账上冲突，★逐条标出）**：")
    say(f"      · ★★任务书记「[S36-1] 候选库命中 **141（3.97%）**」")
    say(f"      · ★★★**但 [S36-1] 报告与 `ag_smarts_hits_AG2.tsv` 实测都是 **42（1.18%）**")
    say(f"        ⇒ ★★★**两个数对不上** ⇒ ★★**账上那个 141 的口径【本轮未核到】，★只报事实。**")
    say(f"      · ★★本轮三级规则：★**高度疑似 {n_hi}（{n_hi/sum(cc.values())*100:.2f}%）** "
        f"⇒ ★★★**相对单级 42 收紧了 {42 - n_hi} 个**")

    # 抽检
    rng = random.Random(SEED)
    say(f"\n▎★★必答问 3 · 抽检（★seed={SEED}）⇒ ★★★请 Lang 肉眼核")
    hi = [(k, f) for k, f in C.items() if f["tier"] == "高度疑似"]
    mid = [(k, f) for k, f in C.items() if f["tier"] == "疑似"]
    say(f"\n   ★★**高度疑似抽 5 个**（★共 {len(hi)} 个）：")
    for k, f in rng.sample(hi, min(5, len(hi))):
        say(f"      {k[:14]:14s} 中{f['n_hit']}条({f['hits']}) MW={f['MW']:>6.0f} "
            f"nc={f['net_charge_v2']} 伯胺={f['n_prim_amine_entry']}")
        say(f"         {cand_meta.get(k, {}).get('smiles', '')[:92]}")
    say(f"\n   ★★**疑似抽 10 个**（★共 {len(mid)} 个）：")
    for k, f in rng.sample(mid, min(10, len(mid))):
        say(f"      {k[:14]:14s} 中{f['n_hit']}条({f['hits']}) MW={f['MW']:>6.0f} "
            f"nc={f['net_charge_v2']} 伯胺={f['n_prim_amine_entry']}")
        say(f"         {cand_meta.get(k, {}).get('smiles', '')[:92]}")


def main():
    say("#" * 78)
    say("S36-2 查证轮 · 氨基糖苷三级标记规则（★★只核不改，★★★标记不影响分数）")
    say(f"★开跑 {datetime.now():%Y-%m-%d %H:%M:%S}   PID={os.getpid()}")
    say("★★★不造秤、★不改构件、★不改任何既有产物、★★★不改任何分子的分数")
    say("★★★三条判据各自独立评估；★★门槛以件二实测为准，★太严则停下报出")
    say(f"⚠️ 两条禁读：{FORBIDDEN} —— ★本脚本从不打开")
    say("#" * 78)

    # ---- 载入各批 ----
    sets = [("二级84", CLSZ, "smiles", "inchikey_full"),
            ("A秤47", FEATW, None, None),          # ★下面单独处理
            ("三级3805", L3, "SMILES", "完整InChIKey"),
            # ⚠️★★★键必须用 row_i —— ★Reference 是【文献编号，不唯一】（6416 行只有 5170 个唯一值）
            #   ★首版用 Reference 做键 ⇒ ★★丢了 1,246 个分子 ⇒ ★本轮已修
            ("Ritz6416", RITZ, "smiles_norm", "row_i"),
            ("候选库3553", CANDF, "smiles", "mol_id")]
    batches = {}
    meta = {}
    for nm, p, scol, kcol in sets:
        if nm.startswith("A秤"):
            continue
        R = load_tsv(p)
        d, mm = {}, {}
        for r in R:
            k = str(r.get(kcol, "")).strip()
            if not k:
                continue
            d[k] = evaluate(r.get(scol))
            mm[k] = r
        batches[nm] = d
        meta[nm] = mm
        say(f"   ★载入 {nm:14s} {p}\n      md5={md5(p)}  ★条目 {len(d)}")

    # A/D/B 秤（★features_water 无 SMILES ⇒ 经 inchikey 挂回 class_z）
    Zmap = {r["inchikey_full"].strip(): r for r in load_tsv(CLSZ)}
    for scale in ("A", "D", "B"):
        d = {}
        for r in load_tsv(FEATW):
            if (r.get("scale") or "").strip() != scale:
                continue
            k = (r.get("inchikey_full") or "").strip()
            smi = (Zmap.get(k, {}) or {}).get("smiles")
            d[k or r.get("molecule")] = evaluate(smi)
        batches[f"{scale}秤{len(d)}"] = d
    say(f"   ★载入 A/D/B 秤（★SMILES 经 inchikey 挂回 molecule_class_z.tsv）")
    for scale in ("A", "D", "B"):
        tot = sum(1 for r in load_tsv(FEATW) if (r.get("scale") or "").strip() == scale)
        got = sum(1 for r in load_tsv(FEATW)
                  if (r.get("scale") or "").strip() == scale
                  and (r.get("inchikey_full") or "").strip() in Zmap)
        if got < tot:
            say(f"      ⚠️★★{scale} 秤 {tot} 个里【取不到 SMILES】的 = **{tot-got}**"
                f" ⇒ ★成因：★molecule_class_z 只含【过了入组标准的 84 个】，"
                f"★★{scale} 秤里被入组删掉的分子取不到 ⇒ ★★★如实报，★不填充、★不猜结构")

    stage1(batches)

    # ---- 件二 ----
    gold_rows = []
    for r in load_tsv(CLSZ):
        if "氨基糖苷" in (r.get("自判大类") or ""):
            f = evaluate(r["smiles"])
            gold_rows.append((r["molecule"], r["smiles"], f))
    gold_rows.sort(key=lambda x: x[0])
    rows2, part = stage2(gold_rows)

    # ---- 件三 ----
    cand_meta = meta["候选库3553"]
    stage3(batches, cand_meta)

    # ---- 落盘 ----
    p1 = os.path.join(S36, "ag_criteria_hits_AG3.tsv")
    with open(p1, "w", newline="", encoding="utf-8") as fh:
        fh.write("# S36-2 件二 · 一行一分子 × A/B/C 三条判据（★盘上 6 个真氨基糖苷）。\n")
        fh.write("# ★三条判据【各自独立】：A=氨基环醇核心／B=≥2糖环经糖苷键相连／C=净正电≥+3 且 伯胺(ENTRY)≥3。\n")
        fh.write("# ★net_charge 与 n_prim_amine_entry 均【本轮当场算】，★口径照 [S34-6] 定稿。\n")
        fh.write(f"# 行数：{len(rows2)}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows2[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows2:
            w.writerow(r)
    say(f"\nwrote {p1} ({len(rows2)} 行)")

    # ---- 件三产物：★★分数与标记【分开两列】----
    sc = {r["mol_id"]: r for r in load_tsv(CANDS)}
    say(f"\n   ★穿透分数源 = {CANDS}（md5={md5(CANDS)}，★★只读，★本轮不改任何分数）")
    out = []
    for bn, R in batches.items():
        for k, f in R.items():
            m = meta.get(bn, {}).get(k, {})
            score = ""
            score_src = ""
            if bn.startswith("候选库"):
                score = sc.get(k, {}).get("score_penetration", "")
                score_src = "scored_aibiotics.tsv:score_penetration（★只读，★未改）"
            elif bn == "二级84":
                score = (Zmap.get(k, {}) or {}).get("自判大类", "")
                score_src = "★该批无数值分数（★标签制）"
            out.append(dict(批次=bn, key=k,
                            SMILES=(m.get("smiles") or m.get("smiles_norm") or m.get("SMILES") or ""),
                            穿透分数=score, 分数来源=score_src,
                            氨基糖苷标记=(f["tier"] if f else "★算不出"),
                            命中判据=(f["hits"] if f else ""),
                            A=(f["A"] if f else ""), B=(f["B"] if f else ""), C=(f["C"] if f else ""),
                            命中条数=(f["n_hit"] if f else ""),
                            net_charge_v2=(f["net_charge_v2"] if f else ""),
                            n_prim_amine_entry=(f["n_prim_amine_entry"] if f else ""),
                            MW=(f["MW"] if f else "")))
    p2 = os.path.join(S36, "ag_flag_all_AG3.tsv")
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        fh.write("# S36-2 件三 · 一行一分子。★★★【穿透分数】与【氨基糖苷标记】是【两个独立的列】。\n")
        fh.write("# ⚠️★★★★分数一律照常给 —— ★本轮【未因标记而改动、置空或剔除任何一个分数】。\n")
        fh.write("# ★标记三级：高度疑似(三条全中) / 疑似(两条中) / 不标(≤一条)。\n")
        fh.write("# ⚠️★★件二实测显示该门槛【可能太严】—— ★详见 ag_three_tier_AG3.md，★本轮未擅自改。\n")
        fh.write(f"# 行数：{len(out)}\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader()
        for r in out:
            w.writerow(r)
    say(f"wrote {p2} ({len(out)} 行)")
    say(f"\n★★本轮总墙钟 {time.time()-_t0:.0f}s = {(time.time()-_t0)/60:.1f} 分钟")
    say("★★★三件跑完。★报告由 Claude 另写 ag_three_tier_AG3.md。")


if __name__ == "__main__":
    main()
