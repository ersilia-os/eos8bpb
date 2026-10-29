#!/usr/bin/env python
"""S34-12 —— 定"抗生素样分子"的入组标准 ＋ 排除前后并列重算。

★★★【Lang 2026-08-18 定的入组标准 —— ★先验理由，★在看结果之前定死】逐字：
    「本 benchmark 的评价对象是【抗生素样分子】；糖与寡肽属于营养物/代谢物，
      其跨膜以不同机制为主，故【预先排除】。」
  ⚠️★★★这条是【先验的机制理由】，★不是"因为信号方向相反才删"。
     ⇒ ★★排除后信号变好只作【佐证】，★★★不作理由。
  ⇒ ★★★故本轮【排除前 / 排除后】两套数【并列公布】，★旧的一套照样保留、不覆盖。

★★★【用途决定写法 —— ★错误方向不对称，★跑前写死】
    · 把真抗生素误判成糖/肽（★误删）= 真损失 ⇒ 模型的好分子被扔掉
    · 把糖漏判成抗生素（★漏删）= 它进来被打个低分 ⇒ 代价小得多
  ⇒ ★★★宁可漏删，★★★绝不误删 ⇒ ★标准一律写得【保守】。
  ⇒ ★凡"保守项"在下面逐条标出，★它们只会让删得【更少】。

⚠️★★★【不许用 classify() 的类别名来删】—— [S34-11 实测] 它在这 94 个上就已出错：
    · fosfomycin 是抗生素，被判成"糖·糖醇·糖酸"（★最坏的一类错）
    · N-Acetyl-D-glucosamine → 判成"氨基酸·多胺"；两个半乳糖苷 → 一个判"其他"
  ⇒ ★本模块【完全不看 chem_class】，★一律自己的结构 SMARTS。

★★★【三类不删，逐条写死（★Lang 已指明，★不许自行扩大）】
    · 氨基酸（单体）不是糖也不是肽 ⇒ ★本轮不删（★要不要删由 Lang 另裁）
      ⇒ 规则 P 要求【至少 1 个主链肽键】⇒ 单体天然不命中
    · linezolid 是抗生素（噁唑烷酮类），★落在 neutral 档里 ⇒ ★不许因"中性"而删
      ⇒ 规则 S 要求【≥3 个醇羟基 且 无芳环】⇒ linezolid 两条都不满足
    · fosfomycin 是抗生素 ⇒ ★不许删
      ⇒ 规则 S 要求【可电离基团数 = 0】⇒ 膦酸使其 n_ionizable>0；且醇羟基数 = 0

★★ _v2 / _v3 / _w / _x 一个都不覆盖 ⇒ ★新产物一律 _y 后缀。
⚠️ 两条禁读：cooper_features.csv / cooper_casecontrol_*.csv —— ★本脚本从不打开。
⚠️ ★读 tsv 不用 pandas 的 comment='#'；★long 表是混合行尾。
"""
from __future__ import annotations

import collections
import csv
import math
import os
import random
import sys

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 原为开发机的 antibiotic-benchmark/scripts，会抢在本包 _vendor 之前；改为本模块自身所在目录，同族模块照常平铺 import
DL = os.path.join(HOME, "downloads_raw")
S33 = os.path.join(DL, "section33")
S34 = os.path.join(DL, "section34")
FORBIDDEN = ("cooper_features.csv", "cooper_casecontrol_")

NBOOT, NREP, SEED = 2000, 200, 42      # ★与 [S34-8]/[S34-9] 逐字一致
FEATS = ["min_proj_area_v2_lowest", "net_charge_v2", "n_prim_amine_entry",
         "b_rotR", "glob_lowest"]
MAIN = set(FEATS[:3])
N_MIN_SCALE = 4      # ★秤剩不足 4 个 ⇒ 无法保留
N_NOPOWER = 10       # ★n<10 ⇒ 无统计功效


def load_tsv(p):
    if any(f in p for f in FORBIDDEN):
        raise RuntimeError(f"禁读文件: {p}")
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        lines = [l.rstrip("\r") for l in fh.read().split("\n")
                 if not l.startswith("#") and l.strip()]
    return list(csv.DictReader(lines, delimiter="\t"))


# ══════════════════════════════════════════════════════════════════════════
# 件一 / 件四 · ★★★可复用的门（★输入 SMILES，★输出 是否排除 + 命中规则 + 原子索引）
# ══════════════════════════════════════════════════════════════════════════
BETA_LACTAM = "C1(=O)NCC1"          # ★4 元 β-内酰胺环（★与 classify() 同一条，★但只当"保护罩"用）

RULE_S = [   # ★糖类：★★全部满足才删
    ("S1 含糖环 O（吡喃/呋喃/环醚）", "[CX4;R][OX2;R][CX4;R]", ">=1"),
    ("S2 醇羟基（连 sp3 C）",        "[OX2H][CX4]",            ">=3"),
    ("S3 强档可电离基团总数",         "(承 S34-11 件一 n_acidic+n_basic)", "==0"),
    ("S4 不含 β-内酰胺环",           BETA_LACTAM,              "==0"),
    ("S5 芳香环个数（★保守项）",      "(RingInfo 全芳香原子的环)", "==0"),
    ("S6 非糖环个数（★★Lang 2026-08-18 明令增补）", "(RingInfo 里【不是糖环】的环)", "==0"),
]
# ★★S6 里"糖环"的判据（★跑前写死，★因为它现在【参与删除决策】，不再只是诊断）：
#   ★一个环算【糖环】⇔ 该环里【≥3 个原子】落在 [CX4;R][OX2;R][CX4;R] 的命中原子集合里；
#   ★否则算【非糖环】。⇒ S6 要求非糖环 == 0，★即"整个分子的环全是糖环"。

RULE_P = [   # ★寡肽：★P0 先排除，其余全部满足才删
    ("P0 ★先排除：含 β-内酰胺环 ⇒ 一律不删", BETA_LACTAM, "==0"),
    ("P1 主链肽键 Cα-C(=O)-NH-Cα-C(=O)",
     "[CX4][CX3](=[OX1])[NX3;H1][CX4][CX3](=[OX1])", ">=1"),
    ("P2 游离 α-氨基（N 端）",
     "[$([NX3;H2;!$(NC=[O,S,N]);!$(N[!#6;!#1])][CX4][CX3]=[OX1]),"
     "$([NX4;H3;+][CX4][CX3]=[OX1])]", ">=1"),
    ("P3 游离羧基（C 端）", "[CX4][CX3](=[OX1])[OX2H1,OX1-]", ">=1"),
    ("P4 环个数（★保守项：含任何环一律不删）", "(RingInfo.NumRings)", "==0"),
    ("P5 主链肽键数（★保守项：寡肽上限）", "(同 P1 的命中数)", "<=4"),
]


def _sm(cache={}):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if not cache:
        cache["bl"] = Chem.MolFromSmarts(BETA_LACTAM)
        cache["s1"] = Chem.MolFromSmarts("[CX4;R][OX2;R][CX4;R]")
        cache["s2"] = Chem.MolFromSmarts("[OX2H][CX4]")
        cache["p1"] = Chem.MolFromSmarts("[CX4][CX3](=[OX1])[NX3;H1][CX4][CX3](=[OX1])")
        cache["p2"] = Chem.MolFromSmarts(
            "[$([NX3;H2;!$(NC=[O,S,N]);!$(N[!#6;!#1])][CX4][CX3]=[OX1]),"
            "$([NX4;H3;+][CX4][CX3]=[OX1])]")
        cache["p3"] = Chem.MolFromSmarts("[CX4][CX3](=[OX1])[OX2H1,OX1-]")
        for k, v in cache.items():
            if v is None:
                raise RuntimeError(f"SMARTS 编译失败: {k}")
    return cache


def _diag(smiles):
    """★★只用于【事后诊断报告】，★★★不参与任何删除决策。

    报环的构成：★糖环（环内含 [CX4;R][OX2;R][CX4;R] 的环）与★非糖环各几个。
    ★有非糖环 ⇒ 说明它可能是"糖苷化的非糖骨架"（★如糖苷化聚酮）⇒ 提请 Lang 肉眼核。
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import Descriptors
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return dict(MW="", n_heavy="", n_ring="", n_sugar_ring="", n_other_ring="")
    ri = m.GetRingInfo()
    patt = Chem.MolFromSmarts("[CX4;R][OX2;R][CX4;R]")
    sug_atoms = {i for mm in m.GetSubstructMatches(patt) for i in mm}
    ns = no = 0
    for ring in ri.AtomRings():
        if len(set(ring) & sug_atoms) >= 3:
            ns += 1
        else:
            no += 1
    return dict(MW=f"{Descriptors.MolWt(m):.1f}", n_heavy=m.GetNumHeavyAtoms(),
                n_ring=ri.NumRings(), n_sugar_ring=ns, n_other_ring=no)


def scope_gate(smiles: str, with_S6: bool = True) -> dict:
    """★★★可复用的门。输入 SMILES ⇒ 输出为什么删/为什么不删。

    返回 dict:
      excluded      bool    ★是否排除
      rule          str     ★命中哪条规则（'S'=糖类 / 'P'=寡肽 / '' = 不删）
      why           str     ★★为什么 —— ★逐条列出每个子条件的实测值与是否满足
      atoms         list    ★命中的原子索引（★S 命中糖环+羟基；P 命中主链肽键）
      err           str     ★解析失败时的说明
    ★★不许只输出布尔值 —— why 与 atoms 一定填。
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from s34k_charge_boundary import ionizable_profile
    out = dict(excluded=False, rule="", why="", atoms=[], err="",
               近似命中S_仅差保守项S5=False, 近似命中P_仅差保守项P4P5=False,
               n_sugar_ring="", n_other_ring="")
    if not (smiles or "").strip():
        out["err"] = "★空 SMILES"; out["why"] = "★空 SMILES ⇒ 不删"
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            from _smiles_syntax import normalize_smiles_syntax
            mol = Chem.MolFromSmiles(normalize_smiles_syntax(smiles))
        except Exception:
            mol = None
    if mol is None:
        out["err"] = "★SMILES 解析不了"
        out["why"] = "★SMILES 解析不了 ⇒ ★★不删（★保守：判不了就不删）"
        return out
    c = _sm()
    ri = mol.GetRingInfo()
    n_bl = len(mol.GetSubstructMatches(c["bl"]))
    m_s1 = mol.GetSubstructMatches(c["s1"])
    m_s2 = mol.GetSubstructMatches(c["s2"])
    m_p1 = mol.GetSubstructMatches(c["p1"])
    m_p2 = mol.GetSubstructMatches(c["p2"])
    m_p3 = mol.GetSubstructMatches(c["p3"])
    n_ring = ri.NumRings()
    n_aro = sum(1 for r in ri.AtomRings()
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    # ★★S6：★环里【≥3 个原子】落在糖环 O 的命中集合 ⇒ 算糖环；★否则算非糖环。
    sug_atoms = {i for mm in m_s1 for i in mm}
    n_sugar_ring = n_other_ring = 0
    for ring in ri.AtomRings():
        if len(set(ring) & sug_atoms) >= 3:
            n_sugar_ring += 1
        else:
            n_other_ring += 1
    prof = ionizable_profile(smiles)
    n_ion = (prof["n_acidic"] + prof["n_basic"]) if prof["n_acidic"] != "" else None

    # ---- 规则 S（糖类）----
    sc = [("S1 糖环 O", len(m_s1), len(m_s1) >= 1),
          ("S2 醇羟基", len(m_s2), len(m_s2) >= 3),
          ("S3 可电离基团", n_ion, (n_ion == 0) if n_ion is not None else False),
          ("S4 无 β-内酰胺", n_bl, n_bl == 0),
          ("S5 芳香环(★保守项)", n_aro, n_aro == 0)]
    if with_S6:
        sc.append(("S6 非糖环(★Lang 明令增补)", n_other_ring, n_other_ring == 0))
    s_hit = all(ok for _, _, ok in sc)

    # ---- 规则 P（寡肽）----
    pc = [("P0 无 β-内酰胺(★先排除)", n_bl, n_bl == 0),
          ("P1 主链肽键", len(m_p1), len(m_p1) >= 1),
          ("P2 游离 α-氨基", len(m_p2), len(m_p2) >= 1),
          ("P3 游离羧基", len(m_p3), len(m_p3) >= 1),
          ("P4 环个数(★保守项)", n_ring, n_ring == 0),
          ("P5 肽键数<=4(★保守项)", len(m_p1), len(m_p1) <= 4)]
    p_hit = all(ok for _, _, ok in pc)

    def fmt(conds):
        return "；".join(f"{lab}={v}{'✓' if ok else '✗'}" for lab, v, ok in conds)

    # ★★保守项的代价：★仅差保守项就会被删的（★用于如实报"漏删了多少"）
    # ★★补 S6 后必须把【两个保守项一起】扣掉才算得准 ——
    #   ★只扣 S5 的话，同时不满足 S6 的分子会被这个指标漏掉（★S34-12 裁定后实测发现）。
    out["近似命中S_仅差保守项S5"] = bool(
        all(ok for lab, _, ok in sc if not (lab.startswith("S5") or lab.startswith("S6")))
        and not s_hit)
    out["近似命中P_仅差保守项P4P5"] = bool(all(ok for lab, _, ok in pc
                                               if not (lab.startswith("P4") or lab.startswith("P5")))
                                            and not p_hit)
    if s_hit:
        out.update(excluded=True, rule="S",
                   atoms=sorted({i for m in m_s1 for i in m} | {i for m in m_s2 for i in m}))
    elif p_hit:
        out.update(excluded=True, rule="P",
                   atoms=sorted({i for m in m_p1 for i in m}))
    out["n_sugar_ring"] = n_sugar_ring
    out["n_other_ring"] = n_other_ring
    out["why"] = (f"[规则S 糖类] {fmt(sc)} ⇒ {'★命中(删)' if s_hit else '不命中'}  ‖  "
                  f"[规则P 寡肽] {fmt(pc)} ⇒ {'★命中(删)' if p_hit else '不命中'}")
    return out


# ══════════════════ 统计（★与 [S34-8]/[S34-9] 逐字同实现） ══════════════════
def spearman(x, y):
    n = len(x)
    if n < 3:
        return None

    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def boot_ci(x, y, rng, nboot=NBOOT):
    base = spearman(x, y)
    if base is None:
        return None, None, None
    bs = []
    for _ in range(nboot):
        idx = [rng.randrange(len(x)) for _ in range(len(x))]
        r = spearman([x[i] for i in idx], [y[i] for i in idx])
        if r is not None:
            bs.append(r)
    if len(bs) < nboot * 0.5:
        return base, None, None
    bs.sort()
    return base, bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]


def strat_split(rows, cls, rng):
    """★分层键 = (chem_class, 终点三分位档)；★逐字沿用 [S34-9] s34i 的实现。"""
    vals = sorted(float(r["value"]) for r in rows)
    n = len(vals)
    lo, hi = vals[n // 3], vals[2 * n // 3]
    strata = collections.defaultdict(list)
    for r in rows:
        v = float(r["value"])
        tier = "hi" if v >= hi else ("lo" if v < lo else "mid")
        strata[(cls.get(r["inchikey_full"], "★未判"), tier)].append(r)
    out = {}
    flip = 0
    for key in sorted(strata):
        grp = strata[key][:]
        rng.shuffle(grp)
        for i, r in enumerate(grp):
            out[r["inchikey_full"]] = "S1" if (i + flip) % 2 == 0 else "S2"
        flip = (flip + len(grp)) % 2
    return out


# ══════════════════════════════════════════════════════════════════════════
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(S34, exist_ok=True)

    print("=" * 80)
    print("S34-12 · 入组标准（抗生素样分子）＋ 排除前后并列重算")
    print("=" * 80)
    print("★★★入组标准的先验理由（Lang 2026-08-18，★逐字）：")
    print("    「本 benchmark 的评价对象是【抗生素样分子】；糖与寡肽属于营养物/代谢物，")
    print("      其跨膜以不同机制为主，故【预先排除】。」")
    print("  ⚠️★★★这是【先验的机制理由】，★不是「因为信号方向相反才删」。")
    print("     ⇒ 排除后信号变好只作【佐证】，★不作理由。")
    print("★★★误差方向不对称（写死）：★宁可漏删，★★绝不误删 ⇒ 标准写得保守。")
    print()
    print("★★★★【Lang 2026-08-18 裁定 ①】规则 S 增补子条件 ★★S6：非糖环数 == 0")
    print("   ★理由（★逐字）：[S34-12] 候选库实测 —— 被删的 24 个里 23 个含非糖环")
    print("     ⇒ 它们不是「一个糖」，是【非糖骨架挂了糖环】（糖苷化聚酮/内酯样）")
    print("     ⚠️★而氨基糖苷本身就是典型抗生素，正是「糖环挂在非糖骨架上」")
    print("     ⇒ ★★现写法会在候选库上【成批误删真抗生素样分子】⇒ 撞上「绝不误删」")
    print("   ★规则 S 的【意图】是「整个分子就是糖」 ⇒ ★补上 S6 才与意图一致")
    print("   ⚠️★★代价接受：★漏删会变多 ⇒ ★如实报新的漏删数，★不再放宽别的")
    print("★★★这是【Lang 明令】补的子条件，★★不是助手看了结果自行放宽。")
    print("   ⇒ ★本次改动【只加 S6 这一条】，★其余 SMARTS 与阈值一个字未动。")

    # ================= 件一 =================
    print("\n" + "─" * 80)
    print("件一 · 排除标准（★结构 SMARTS，★★绝不按名字/类别名判）")
    print("─" * 80)
    print("★★★本模块【完全不看 chem_class】—— [S34-11] 已实测它会误删 fosfomycin。\n")
    print("▎必答问 1 —— 两条规则的 SMARTS 逐条报")
    print("\n★规则 S（糖类）—— ★★下列 5 条【全部满足】才删：")
    for lab, sma, cond in RULE_S:
        print(f"    {lab:28s} {cond:5s}  {sma}")
    print("\n★规则 P（寡肽）—— ★★P0 先排除，其余【全部满足】才删：")
    for lab, sma, cond in RULE_P:
        print(f"    {lab:38s} {cond:5s}  {sma}")
    print("\n  ⇒ ★★P0 就是任务书点名的那一步：★★β-内酰胺环本身含 C(=O)N，")
    print("     ★不先把它排除，规则 P 会把所有 β-内酰胺当成肽删掉。")

    # ---- 在 94 个分子上跑 ----
    fn = load_tsv(os.path.join(S34, "features_new.tsv"))
    smi_g, smi_any = {}, {}
    for r in fn:
        s = (r.get("smiles_used") or "").strip()
        if s:
            smi_g[(r["group"], r["inchikey_full"])] = s
            smi_any.setdefault(r["inchikey_full"], s)
    gml = load_tsv(os.path.join(S33, "group_molecule_list.tsv"))
    cls = {}
    for r in gml:
        ik = (r.get("inchikey_full") or "").strip()
        cc = (r.get("chem_class") or "").strip()
        if ik and cc:
            cls.setdefault(ik, cc)

    rows = []
    scales = {}
    for g in "ADB":
        sc = load_tsv(os.path.join(S34, f"scale_{g}_v2.tsv"))
        scales[g] = sc
        for x in sc:
            ik = x["inchikey_full"]
            smi = smi_g.get((g, ik)) or smi_any.get(ik, "")
            res = scope_gate(smi, with_S6=True)
            old6 = scope_gate(smi, with_S6=False)
            rows.append({
                "scale": g, "molecule": x["molecule"], "inchikey_full": ik,
                "chem_class_仅供对照": cls.get(ik, "★未在 group_molecule_list"),
                "value": x.get("value", ""),
                "判定": "★删" if res["excluded"] else "保留",
                "★判定_旧规则S(无S6)": "★删" if old6["excluded"] else "保留",
                "★S6改变了判定": "★★是" if res["excluded"] != old6["excluded"] else "",
                "n_sugar_ring": res["n_sugar_ring"], "n_other_ring": res["n_other_ring"],
                "命中规则": res["rule"], "为什么": res["why"],
                "命中原子索引": ",".join(map(str, res["atoms"])),
                "仅差保守项S5或S6就会删": "★是" if res["近似命中S_仅差保守项S5"] else "",
                "仅差保守项P4P5就会删": "★是" if res["近似命中P_仅差保守项P4P5"] else "",
                "smiles_used": smi, "err": res["err"]})

    dele = [r for r in rows if r["判定"] == "★删"]
    keep = [r for r in rows if r["判定"] != "★删"]
    print(f"\n▎必答问 2 —— 在 94 个分子上跑一遍")
    print(f"  ★判为【删】= {len(dele)} 行；★保留 = {len(keep)} 行（★94 行 = 分子×秤）")
    print(f"\n  ★被删的逐个列出：")
    for r in sorted(dele, key=lambda z: (z["命中规则"], z["scale"], z["molecule"])):
        print(f"    [{r['命中规则']}] {r['scale']} {r['molecule'][:38]:38s} "
              f"(chem_class 对照: {r['chem_class_仅供对照']})")

    # ---- ★★★误删/漏删核算 ----
    # ★"抗生素" 一律按【结构判据 chem_class】认（★仅用于核算误删，★★不用于删除决策）
    AB_CLS = {"β-内酰胺", "喹诺酮类（含无氟萘啶酮/吡啶并嘧啶）", "氨基糖苷"}
    # ★另加三个 Lang 已点名的抗生素（★结构上不落在上面三类里）
    AB_EXTRA_IK = set()
    for r in rows:
        nm = r["molecule"].strip().lower()
        if nm in ("fosfomycin", "linezolid"):
            AB_EXTRA_IK.add(r["inchikey_full"])
    mis = [r for r in dele if r["chem_class_仅供对照"] in AB_CLS
           or r["inchikey_full"] in AB_EXTRA_IK]
    # ★漏删 = 结构上是糖/肽（按 chem_class 对照）但没被删
    LEAK_CLS = {"糖·糖醇·糖酸", "肽"}
    leak = [r for r in keep if r["chem_class_仅供对照"] in LEAK_CLS
            and r["inchikey_full"] not in AB_EXTRA_IK]
    print(f"\n  ★★★误删（抗生素被判为删）= {len(mis)}"
          + ("" if not mis else " —— " + "; ".join(f"{r['scale']}/{r['molecule']}" for r in mis)))
    print(f"  ★漏删（chem_class 为糖/肽但未删）= {len(leak)}"
          + ("" if not leak else " —— " + "; ".join(f"{r['scale']}/{r['molecule']}" for r in leak)))
    print("    ⚠️★漏删按设计【可接受】：★宁可漏删，★绝不误删。")
    print("\n  ★★★补一条【结构层面】的漏删核算（★上面那条用 chem_class 对照，★会漏掉 classify() 判错的）：")
    nearS = [r for r in keep if r["仅差保守项S5或S6就会删"]]
    nearP = [r for r in keep if r["仅差保守项P4P5就会删"]]
    print(f"    ★仅因保守项 S5(无芳环)/S6(非糖环=0) 而没删的 = {len(nearS)}"
          + ("" if not nearS else " —— " + "; ".join(f"{r['scale']}/{r['molecule']}" for r in nearS)))
    print(f"    ★仅因保守项 P4/P5（无环、肽键<=4）而没删的 = {len(nearP)}"
          + ("" if not nearP else " —— " + "; ".join(f"{r['scale']}/{r['molecule']}" for r in nearP)))
    print("    ⇒ ★★这几个就是【保守项换来的漏删】，★如实报出，★★本轮不放宽规则。")
    ch6 = [r for r in rows if r["★S6改变了判定"]]
    print(f"\n  ★★★S6 带来的变化（★94 个建秤分子上）：{len(ch6)} 个判定改变"
          + ("" if not ch6 else " —— " + "; ".join(f"{r['scale']}/{r['molecule']}"
                                                   f"（旧{r['★判定_旧规则S(无S6)']}→新{r['判定']}）" for r in ch6)))
    if not ch6:
        print("     ⇒ ★★在这 94 个上，S6 【不改变任何一个判定】——")
        print("        ★因为建秤集里的糖都是【纯糖】（环全是糖环），★非糖环本来就是 0。")

    print(f"\n▎必答问 3 —— 三个硬性验收")
    ok3 = True
    for nm in ("fosfomycin", "linezolid"):
        hits = [r for r in rows if r["molecule"].strip().lower() == nm]
        for r in hits:
            good = r["判定"] != "★删"
            ok3 &= good
            print(f"  {'✅' if good else '❌'} {nm} ({r['scale']}) ⇒ {r['判定']}")
            print(f"       {r['为什么']}")
    bl = [r for r in rows if r["chem_class_仅供对照"] == "β-内酰胺"]
    bl_bad = [r for r in bl if r["判定"] == "★删"]
    ok3 &= not bl_bad
    print(f"  {'✅' if not bl_bad else '❌'} β-内酰胺 {len(bl)} 行 ⇒ 被删 {len(bl_bad)} 行"
          + ("" if not bl_bad else f" —— {[r['molecule'] for r in bl_bad]}"))

    p1 = os.path.join(S34, "scope_rules_y.tsv")
    with open(p1, "w", encoding="utf-8", newline="") as fh:
        fh.write("# scope_rules_y.tsv -- S34-12 件一：一行 = 一个【分子×秤】的入组判定。\n")
        fh.write("# ★★★入组标准的理由是【先验机制理由】（糖/寡肽是营养物代谢物，跨膜机制不同），\n")
        fh.write("#    ★不是「因为信号方向相反才删」。★排除后信号变好只作佐证。\n")
        fh.write("# ★★判定一律用结构 SMARTS，★★【完全不看 chem_class】；\n")
        fh.write("#    chem_class_仅供对照 一列只用于事后核算误删/漏删，★不参与删除决策。\n")
        fh.write("# ★规则 S（糖类）5 条全满足才删；★规则 P（寡肽）P0 先排除 β-内酰胺后其余全满足才删。\n")
        fh.write("# ★★错误方向不对称：★宁可漏删，★★★绝不误删 ⇒ S5/P4/P5 三条为【保守项】。\n")
        fh.write(f"# 行数：{len(rows)}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p1}  ({len(rows)} 行)")

    if mis:
        print("\n★★★ 误删 > 0 ⇒ ★停下报出，★不继续（件二/件三/件四不跑）")
        return 3
    print("\n★★★ 误删 = 0 ⇒ 硬性红线通过，继续。")
    if not ok3:
        print("★★★ 三个硬性验收未全通过 ⇒ 停下报出。")
        return 3
    if stage == "rules":
        return 0

    # ================= 件二 =================
    print("\n" + "─" * 80)
    print("件二 · 三把秤各删掉几个、剩几个")
    print("─" * 80)
    drop_ik = {r["inchikey_full"] for r in dele}
    kept = {}
    for g in "ADB":
        before = scales[g]
        after = [x for x in before if x["inchikey_full"] not in drop_ik]
        kept[g] = after
        gone = [x for x in before if x["inchikey_full"] in drop_ik]
        print(f"\n★秤 {g}：删前 n={len(before)} ⇒ 删掉 {len(gone)} ⇒ ★★剩 {len(after)}")
        for x in gone:
            rr = next(r for r in rows if r["scale"] == g and r["inchikey_full"] == x["inchikey_full"])
            print(f"    − {x['molecule'][:38]:38s} [规则{rr['命中规则']}]")
        if len(after) < N_MIN_SCALE:
            print(f"    ★★★剩 {len(after)} < {N_MIN_SCALE} ⇒ ★★【该秤无法保留】")
            print("       ⇒ 如实记：「该尺的分子构成与主评价对象不同类，★故不纳入定量层」。")
            print("       ⇒ ★这不是损失，★是把一把本来就撑不住的秤如实取消。")
        elif len(after) < N_NOPOWER:
            print(f"    ⚠️★剩 {len(after)} < {N_NOPOWER} ⇒ ★无统计功效，★件三不重算")
    print("\n★去核任务书点名的三处牵动：")
    for nm in ("glucose", "sucrose", "maltose", "linezolid"):
        hit = [r for r in rows if r["molecule"].strip().lower().strip("()") == nm]
        if hit:
            print(f"  {nm}: " + "；".join(f"{r['scale']}秤→{r['判定']}" for r in hit))
    shared_ab = [r for r in rows if r["scale"] == "A" and any(
        z["scale"] == "B" and z["inchikey_full"] == r["inchikey_full"] for z in rows)]
    print(f"  ★A∩B 共有分子 {len(shared_ab)} 个 ⇒ "
          + "；".join(f"{r['molecule']}({r['判定']})" for r in shared_ab)
          + "  ⇒ ★删 B 确实同时牵动 A")
    print(f"\n▎必答 —— A/D/B 删后各剩几个："
          + "  ".join(f"{g}={len(kept[g])}" for g in "ADB"))
    print("\n★★★硬闸（Lang 裁定 ①-2）：★D 必须仍是 28")
    if len(kept["D"]) != 28:
        print(f"  ❌★★★D = {len(kept['D'])} ≠ 28 ⇒ ★停下报出：S6 误伤了 D")
        return 3
    print("  ✅ D = 28 ⇒ 通过")
    print("\n★★★【Lang 裁定 ②】B 秤的处置 —— ★不动手删，★只写死结论：")
    print("  ▎该尺（B / arabinose 糖肽尺）的分子构成与本 benchmark 的主评价对象不同类，")
    print("    ★故【不纳入定量层】。")
    print("  ⚠️★★B 的全部旧读数【一律保留，一个字都不删】——")
    print("     scale_B_v2 / _v3 / _w 里的 min_proj_area −0.525（有信号）、")
    print("     b_rotR −0.916（已定为类别共线）★★要原样留着。")
    print("     ★理由：论文里「我们试过这把尺、发现其分子构成与主评价对象不同类，故不纳入」")
    print("       这句话★靠的就是这些数当证据 ⇒ ★★删了反而说不清。")
    print("\n★★★【Lang 裁定 ③】氨基酸（单体）★仍【不删】—— ★Lang 未裁，★本轮不动。")
    aa = [r for r in rows if r["scale"] == "A" and r["判定"] != "★删"
          and r["chem_class_仅供对照"] == "氨基酸·多胺"]
    print(f"  ★A 剩下的 {len(kept['A'])} 个里仍有 {len(aa)} 个『氨基酸·多胺』")
    print(f"  ⚠️★★若将来认定它们也属营养物/代谢物 ⇒ A 会从 {len(kept['A'])} 降到 {len(kept['A'])-len(aa)}")
    print("     ⇒ ★★伤筋动骨 ⇒ ★需要一条【独立于糖肽】的理由，★不能顺手并进规则 S/P。")

    # ================= 件三 =================
    print("\n" + "─" * 80)
    print("件三 · 排除前后【并列】重算（★★不许只报排除后）")
    print("─" * 80)
    oldv3 = {(r["scale"], r["feature"]): r
             for r in load_tsv(os.path.join(S34, "scale_feature_stats_v3.tsv"))}
    stats = []
    for g in "ADB":
        after = kept[g]
        n = len(after)
        print(f"\n★★秤 {g}  排除后 n={n}")
        for f in FEATS:
            o = oldv3.get((g, f), {})
            rec = dict(scale=g, feature=f, 主或参照=("主" if f in MAIN else "参照"),
                       n_排除前=len(scales[g]), n_排除后=n,
                       rho_排除前=o.get("spearman_rho_沿用v2", ""),
                       ci_lo_排除前=o.get("ci_lo_沿用v2", ""), ci_hi_排除前=o.get("ci_hi_沿用v2", ""),
                       判据1_排除前=o.get("判据1_CI跨0_沿用v2", ""),
                       判据3_排除前=o.get("新划法_判据3", ""),
                       结论_排除前=o.get("新划法_结论", ""))
            if n < N_MIN_SCALE:
                rec.update(rho_排除后="", ci_lo_排除后="", ci_hi_排除后="",
                           判据1_排除后="", 判据3_排除后="",
                           结论_排除后=f"★★该秤剩 {n} < {N_MIN_SCALE} ⇒ ★不纳入定量层，★不重算",
                           方向是否翻转="—", 结论是否变了="★该秤已取消")
                stats.append(rec); continue
            xs = [x[f] for x in after]; ys = [x["value"] for x in after]
            if not all(a.strip() and b.strip() for a, b in zip(xs, ys)):
                rec.update(rho_排除后="★有空值", ci_lo_排除后="", ci_hi_排除后="",
                           判据1_排除后="", 判据3_排除后="", 结论_排除后="★算不出",
                           方向是否翻转="—", 结论是否变了="—")
                stats.append(rec); continue
            x = [float(a) for a in xs]; y = [float(b) for b in ys]
            rho, lo, hi = boot_ci(x, y, random.Random(SEED))
            if rho is None:
                rec.update(rho_排除后="★算不出", ci_lo_排除后="", ci_hi_排除后="",
                           判据1_排除后="", 判据3_排除后="",
                           结论_排除后="★判不了（★ρ 算不出：方差为 0 / 全并列）",
                           方向是否翻转="—")
            else:
                c1 = "不跨0" if (lo is not None and (lo > 0 or hi < 0)) else "★跨0"
                # ★判据③：★承 [S34-9] 分层划法，NREP=200
                rng = random.Random(SEED)
                nj = na = 0
                for _ in range(NREP):
                    asg = strat_split(after, cls, rng)
                    s1 = [z for z in after if asg[z["inchikey_full"]] == "S1"]
                    s2 = [z for z in after if asg[z["inchikey_full"]] == "S2"]
                    if len(s1) < 4 or len(s2) < 4:
                        continue
                    r1 = spearman([float(z[f]) for z in s1], [float(z["value"]) for z in s1])
                    r2 = spearman([float(z[f]) for z in s2], [float(z["value"]) for z in s2])
                    if r1 is None or r2 is None:
                        continue
                    nj += 1
                    if r1 * r2 > 0:
                        na += 1
                frac = (na / nj) if nj else None
                if nj == 0:
                    c3 = "★判不了（★分层后子集 n<4 或 ρ 算不出）"
                elif frac >= 0.95:
                    c3 = f"一致（{na}/{nj} = {frac:.0%}）"
                elif frac <= 0.05:
                    c3 = f"★不一致（仅 {na}/{nj} = {frac:.0%} 同号）"
                else:
                    c3 = f"★不稳（{na}/{nj} = {frac:.0%} 同号）"
                if c1 == "不跨0" and c3.startswith("一致"):
                    v = "★有信号（正向）" if rho > 0 else "★有信号（反向）"
                elif c3.startswith("★判不了"):
                    v = "★判不了（子集不足）"
                elif c3.startswith("★不一致") or c3.startswith("★不稳"):
                    v = "★判不了（两子集方向不稳/相反）"
                elif n < N_NOPOWER:
                    v = "★判不了（n<10 无统计功效）"
                else:
                    v = "★无信号"
                old_rho = rec["rho_排除前"]
                flip = "—"
                if old_rho not in ("", "★算不出"):
                    try:
                        flip = "★★是（翻了）" if float(old_rho) * rho < 0 else "否"
                    except ValueError:
                        flip = "—"
                rec.update(rho_排除后=f"{rho:+.4f}",
                           ci_lo_排除后=("" if lo is None else f"{lo:+.4f}"),
                           ci_hi_排除后=("" if hi is None else f"{hi:+.4f}"),
                           判据1_排除后=c1, 判据3_排除后=c3, 结论_排除后=v,
                           方向是否翻转=flip)
            rec["结论是否变了"] = ("★★变了" if rec["结论_排除前"] != rec["结论_排除后"]
                                   else "没变")
            if n < N_NOPOWER:
                rec["结论_排除后"] += f"｜⚠️★n={n}<10 无统计功效"
            stats.append(rec)
            print(f"   {f:26s} 前 ρ={rec['rho_排除前']:>9s} → 后 ρ={rec['rho_排除后']:>9s} "
                  f"| 前:{rec['结论_排除前'][:20]:20s} → 后:{rec['结论_排除后'][:20]:20s} "
                  f"| 翻转={rec['方向是否翻转']}")

    p2 = os.path.join(S34, "scale_stats_excluded_y.tsv")
    with open(p2, "w", encoding="utf-8", newline="") as fh:
        fh.write("# scale_stats_excluded_y.tsv -- S34-12 件三：一行 = 一（秤, 构件）。\n")
        fh.write("# ★★★排除前 / 排除后【两套数并列】—— ★排除前一律沿用 [S34-9] scale_feature_stats_v3.tsv，\n")
        fh.write("#    ★★未重算、未改动；★排除后为本轮新算。\n")
        fh.write(f"# ★排除后：bootstrap nboot={NBOOT} seed={SEED}；判据③沿用 [S34-9] 分层划法 NREP={NREP}。\n")
        fh.write(f"# ★秤剩 <{N_MIN_SCALE} 个 ⇒ 不纳入定量层、不重算；★剩 <{N_NOPOWER} 标『无统计功效』。\n")
        fh.write("# ⚠️『无信号』只在【CI 确实跨 0 ＋ n>=10 ＋ 两子集可判且一致】时才用；★判不了≠无信号。\n")
        fh.write(f"# 行数：{len(stats)}\n")
        w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(stats)
    print(f"\nwrote {p2}  ({len(stats)} 行)")

    print("\n▎必答问 1 —— 排除后哪些格的结论变了（★逐格报）")
    ch = [r for r in stats if r["结论是否变了"].startswith("★")]
    if not ch:
        print("  ★一格都没变")
    for r in ch:
        print(f"  {r['scale']}/{r['feature']:26s} {r['结论_排除前']} → {r['结论_排除后']}")
    print("\n▎必答问 2 —— D 的 net_charge（排除前 ρ=+0.716）排除后是多少")
    d = next(r for r in stats if r["scale"] == "D" and r["feature"] == "net_charge_v2")
    print(f"  排除前 ρ={d['rho_排除前']}  ⇒  排除后 ρ={d['rho_排除后']}  "
          f"(n {d['n_排除前']}→{d['n_排除后']})")
    # ★★数值比对（★不比字符串 —— "0.7162" 与 "+0.7162" 是同一个数）
    try:
        same = abs(float(d["rho_排除前"]) - float(d["rho_排除后"])) < 1e-9
    except ValueError:
        same = False
    print("  ⇒ " + ("★★数值完全相同（差 <1e-9）⇒ ★规则未误伤 D（★D 一个都没删，n 28→28）"
                    if same else "★★★变了 ⇒ ★停下报出：规则可能误伤了 D"))
    print("\n▎必答问 3 —— A 的三个主构件排除后各是多少")
    for f in FEATS[:3]:
        r = next(z for z in stats if z["scale"] == "A" and z["feature"] == f)
        print(f"  {f:26s} 前 {r['rho_排除前']:>9s} [{r['ci_lo_排除前']},{r['ci_hi_排除前']}] "
              f"→ 后 {r['rho_排除后']:>9s} [{r['ci_lo_排除后']},{r['ci_hi_排除后']}]")
    print("\n▎必答问 4 —— 排除前后有没有构件方向翻了")
    fl = [r for r in stats if r["方向是否翻转"].startswith("★")]
    print(f"  {'★一个都没有' if not fl else ''}")
    for r in fl:
        print(f"  ★★{r['scale']}/{r['feature']}: {r['rho_排除前']} → {r['rho_排除后']}")

    # ================= 件四 =================
    print("\n" + "─" * 80)
    print("件四 · 把标准做成可复用的门 ＋ 在候选库上试跑（★只报，★不据此筛选）")
    print("─" * 80)
    print("★门 = 本脚本的 scope_gate(smiles) ⇒ 返回 excluded / rule / why / atoms / err")
    print("★★why 一定填 —— ★不许只输出布尔值。")
    cp = os.path.join(HOME, "antibiotic-benchmark/candidates/features_aibiotics.tsv")
    with open(cp, encoding="utf-8", errors="replace", newline="") as fh:
        cand = list(csv.DictReader(fh, delimiter="\t"))
    print(f"\n★候选库：candidates/features_aibiotics.tsv  n={len(cand)}")
    crows = []
    nS = nP = nerr = 0
    for i, r in enumerate(cand, 1):
        s = (r.get("smiles") or "").strip()
        res = scope_gate(s)
        if res["err"]:
            nerr += 1
        if res["excluded"]:
            nS += res["rule"] == "S"
            nP += res["rule"] == "P"
            d = _diag(s)
            crows.append({"mol_id": r.get("mol_id", ""), "smiles": s,
                          "命中规则": res["rule"], "为什么": res["why"],
                          "命中原子索引": ",".join(map(str, res["atoms"])),
                          "MW": d["MW"], "n_heavy": d["n_heavy"],
                          "环数": d["n_ring"], "糖环数": d["n_sugar_ring"],
                          "★非糖环数": d["n_other_ring"],
                          "★★可疑(有非糖环⇒可能是糖苷化的非糖骨架)":
                              "★★是" if d["n_other_ring"] > 0 else ""})
        if i % 500 == 0:
            print(f"   …{i}/{len(cand)}", flush=True)
    ndel = len(crows)
    print(f"\n▎必答问 1 —— 3553 个里会删掉几个")
    print(f"  ★★删掉 {ndel} / {len(cand)} = {ndel/len(cand):.2%}"
          f"（规则S {nS} 个；规则P {nP} 个）；★SMILES 解析不了 {nerr} 个（★一律不删）")
    if ndel == 0:
        print("  ⇒ ★★★【一个都不删了】—— 明说 0。")
    print(f"\n▎必答问 2 —— 随机抽 20 个被删的，逐个报（★由 Lang 肉眼核有没有误删）")
    rng = random.Random(SEED)
    samp = rng.sample(crows, min(20, len(crows))) if crows else []
    for k, r in enumerate(samp, 1):
        print(f"  {k:2d}. [规则{r['命中规则']}] {r['mol_id']}  MW={r['MW']} 环={r['环数']}"
              f"（糖环{r['糖环数']} / ★非糖环{r['★非糖环数']}）"
              f"  {r['★★可疑(有非糖环⇒可能是糖苷化的非糖骨架)']}")
        print(f"      {r['smiles']}")
    nsus = sum(1 for r in crows if r["★★可疑(有非糖环⇒可能是糖苷化的非糖骨架)"])
    print(f"\n  ★★★事后诊断（★只报，★不改规则，★不参与删除决策）：")
    print(f"     被删的 {ndel} 个里，含非糖环的 = {nsus}")
    if nsus == 0:
        print("     ⇒ ★★★S6 生效后，【被删的分子全部是「环全是糖环」的纯糖骨架】——")
        print("        ★[S34-12] 那一批「糖苷化的非糖骨架」已被 S6 全部挡回。")
    else:
        print("     ⇒ ★★仍有【糖苷化的非糖骨架】漏进删除名单 ⇒ ★★★停下报出，交 Lang 裁。")
    print(f"     ⚠️★对照：★补 S6 前删 24 个（23 个含非糖环）⇒ ★补 S6 后删 {ndel} 个（{nsus} 个含非糖环）。")

    p3 = os.path.join(S34, "scope_filter_on_candidates_y.tsv")
    with open(p3, "w", encoding="utf-8", newline="") as fh:
        fh.write("# scope_filter_on_candidates_y.tsv -- S34-12 件四：一行 = 一个【被门判为删】的候选分子。\n")
        fh.write("# ★★只报，★★★不据此做任何筛选 —— 本表不改动任何候选库产物。\n")
        fh.write(f"# ★来源：candidates/features_aibiotics.tsv（n={len(cand)}）；"
                 f"★被删 {ndel}（S {nS} / P {nP}）；★解析不了 {nerr}（一律不删）。\n")
        fh.write("# ★『为什么』列逐条列出每个子条件的实测值与是否满足。\n")
        fh.write(f"# 行数：{len(crows)}\n")
        if crows:
            w = csv.DictWriter(fh, fieldnames=list(crows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(crows)
    print(f"\nwrote {p3}  ({len(crows)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
