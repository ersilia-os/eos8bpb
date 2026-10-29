#!/usr/bin/env python
"""S34-11 查证轮 —— B 与 A/D 的分界到底绑在哪个【可计算的性质】上。

★★★【查证轮】：★不重造秤、★不改权重、★不定分组、★不写任何既有产物。
★★★ 本脚本【不下】"MoE 该按什么分组"的结论 —— ★判决权在 Lang。
★★ 新产物一律 _x 后缀；★_v2 / _v3 / _w 一个都不碰。

【为什么查这一件（承 S34-10 实测）】
  跨秤权重转移：D↔A 可互转（0.662 / 0.661），涉及 B 的四对全部失败。
  直接成因：n_prim_amine / b_rotR 在 B 上是负、在 A/D 上是正 ⇒ 搬权重等于搬反朝向。
  ⚠️ 两个解释在现有数据上【完全混淆】：
     甲 = 分界是"带不带电"（B 组 net_charge 方差为 0，18 个全中性）
     乙 = 分界是"是不是抗生素"（B 是糖+肽，A/D 是抗生素）
  ⇒ 本轮用【A 组内部现成的净电荷≈0 抗生素】做对照，试着把两者拆开。

【口径 —— ★跑前写死，★逐条报出】
  net_charge  ⇒ ★★直接取 scale_{A,D,B}_v2.tsv 的 net_charge_v2 列（★已有的不重算）
                 该列口径 = net_charge_pH74_v2(rules=('sulfonamide','formamidine'))
  "≈0" 阈值   ⇒ ★★|q| < 0.5（★net_charge_v2 是整数 ⇒ 等价于 q == 0，★阈值仍写死报出）
  电荷型四档（★跑前写死，★互斥且穷尽）：
     anionic       q < -0.5
     cationic      q > +0.5
     zwitterionic  |q| < 0.5 且 n_acidic >= 1 且 n_basic >= 1
     neutral       |q| < 0.5 且 上一条不成立（★即酸碱基团至少缺一边）
     ⇒ ★neutral 档内再细分报 n_ionizable == 0 与 > 0 两小类（★不另立档）
  可电离基团 ⇒ ★★一律 SMARTS 结构判据，★绝不按分子名猜；
     ★匹配在【输入 SMILES 原态】上做，SMARTS 一律同时写中性态与已解离/已质子化态；
     ★★原子级去重：按优先级顺序匹配，中心原子已被占则跳过，命中后占住该 match 全部原子。
  ★★n_acidic / n_basic 只计【任务书点名的那几类】（羧基/磺酸/酚/磷酸/四氮唑/酸性磺酰胺；
     伯胺/仲胺/叔胺/胍基/脒/咪唑/季铵）；
     ★弱酸弱碱（硫醇、苯胺型 NH2、吡啶型 n、噻唑/噁唑型 n）单列 n_*_weak 两列，
     ★★【不进】四档分类 —— 它们在 pH 7.4 基本不解离。
  chem_class  ⇒ ★承 S33-19/S34-9 的结构判据（s33r_group_panorama.classify），
                 ★经 group_molecule_list.tsv 按 inchikey_full 取回。
  bootstrap   ⇒ nboot=2000, seed=42, 整行（分子）重抽；ρ = Spearman。
  ★n<10 标"无统计功效"；★n<4 不做统计、明说做不了。
  ★★★不许因 n 小而合并分组或改阈值。

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

NBOOT, SEED = 2000, 42
Q_EPS = 0.5          # ★"净电荷≈0"的写死阈值：|q| < 0.5
N_NOPOWER = 10       # ★n < 10 ⇒ 无统计功效
N_NOSTAT = 4         # ★n < 4  ⇒ 不做统计


def load_tsv(path):
    if any(f in path for f in FORBIDDEN):
        raise RuntimeError(f"禁读文件: {path}")
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = [l.rstrip("\r") for l in fh.read().split("\n")
                 if not l.startswith("#") and l.strip()]
    return list(csv.DictReader(lines, delimiter="\t"))


# ══════════════════ 件一：可电离基团的 SMARTS（★跑前写死，★逐条报出） ══════════════════
# 每条 = (标签, SMARTS, 中心原子在 SMARTS 里的序号)。★按列出顺序作为优先级。
ACID_MAIN = [
    ("A1 羧酸/羧酸根",      "[CX3](=[OX1])[OX2H1,OX1-]", 0),
    ("A2 磺酸/磺酸根",      "[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]", 0),
    ("A3 磷酸/膦酸(根)",    "[PX4](=[OX1])[OX2H1,OX1-]", 0),
    ("A4 四氮唑",           "[$(c1nnn[nH]1),$(c1nnn[n-]1)]", 0),
    # ★A5 三条 = _charge_rules.py 规则 1 的 R1a/R1b/R1c 逐字照抄（★同一口径，不另立）
    ("A5a 酸性磺酰胺 N-酰基",   "[NX3;H1,H0;!+](-[CX3]=[OX1,SX1])-[SX4](=[OX1])(=[OX1])", 0),
    ("A5b 酸性磺酰胺 N-杂芳基", "[NX3;H1,H0;!+](-[SX4](=[OX1])(=[OX1]))-[c;$(c~[n,o,s;a])]", 0),
    ("A5c 双磺酰亚胺",          "[NX3;H1,H0;!+](-[SX4](=[OX1])(=[OX1]))-[SX4](=[OX1])(=[OX1])", 0),
    ("A6 酚(酚氧负离子)",   "[OX2H1,OX1-;$([O][c])]", 0),
]
ACID_WEAK = [   # ★单列，★不进四档分类
    ("Aw1 硫醇(pKa~10)",    "[SX2H1,SX1-;$([S][CX4])]", 0),
]

BASE_MAIN = [   # ★优先级由高到低
    ("B1 季铵(永久正电)",   "[NX4;H0;+;$([N]([#6])([#6])([#6])[#6])]", 0),
    ("B2 胍基",             "[NX3;!$(N[!#6;!#1])][CX3](=[NX2,NX3;+0,+1])[NX3;!$(N[!#6;!#1])]", 1),
    ("B3 脒(非胍)",         "[NX3;!$(N[!#6;!#1])][CX3]=[NX2,NX3;+0,+1]", 1),
    ("B4 咪唑型芳香 N",     "[nX2;H0;r5;$([n]:[c]:[nX3])]", 0),
    ("B5 脂肪伯胺",         "[$([NX3;H2;$(N[CX4]);!$(NC=[O,S,N]);!$(N[!#6;!#1])]),"
                            "$([NX4;H3;+;$(N[CX4])])]", 0),
    ("B6 脂肪仲胺",         "[$([NX3;H1;$(N([CX4])[CX4]);!$(NC=[O,S,N]);!$(N[!#6;!#1])]),"
                            "$([NX4;H2;+;$(N([CX4])[CX4])])]", 0),
    ("B7 脂肪叔胺",         "[$([NX3;H0;$(N([CX4])([CX4])[CX4]);!$(NC=[O,S,N]);!$(N[!#6;!#1])]),"
                            "$([NX4;H1;+;$(N([CX4])([CX4])[CX4])])]", 0),
]
BASE_WEAK = [   # ★单列，★不进四档分类（★pH 7.4 基本不质子化）
    ("Bw1 苯胺型 NH2(pKa~4.6)", "[NX3;H2;$(N[a]);!$(NC=[O,S,N]);!$(N[!#6;!#1])]", 0),
    ("Bw2 吡啶型芳香 N",        "[nX2;H0;r6]", 0),
    ("Bw3 其他五元芳杂环 N",    "[nX2;H0;r5]", 0),
]


def _compile(defs):
    from rdkit import Chem
    out = []
    for lab, sma, ci in defs:
        p = Chem.MolFromSmarts(sma)
        if p is None:
            raise RuntimeError(f"SMARTS 编译失败: {lab} {sma}")
        out.append((lab, sma, ci, p))
    return out


def count_groups(mol, compiled, claimed):
    """★原子级去重计数：中心原子已被占则跳过；命中后占住该 match 全部原子。"""
    hits = collections.Counter()
    for lab, _sma, ci, patt in compiled:
        for m in mol.GetSubstructMatches(patt):
            c = m[ci]
            if c in claimed:
                continue
            claimed.update(m)
            hits[lab] += 1
    return hits


def ionizable_profile(smiles, cache={}):
    """-> dict(n_acidic, n_basic, n_acidic_weak, n_basic_weak, detail_酸, detail_碱, err)"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if smiles in cache:
        return cache[smiles]
    if not cache.get("_c"):
        cache["_c"] = (_compile(ACID_MAIN), _compile(ACID_WEAK),
                       _compile(BASE_MAIN), _compile(BASE_WEAK))
    am, aw, bm, bw = cache["_c"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            from _smiles_syntax import normalize_smiles_syntax
            mol = Chem.MolFromSmiles(normalize_smiles_syntax(smiles))
        except Exception:
            mol = None
    if mol is None:
        r = dict(n_acidic="", n_basic="", n_acidic_weak="", n_basic_weak="",
                 detail_酸="", detail_碱="", err="★SMILES 解析不了")
        cache[smiles] = r
        return r
    # ★酸与碱各自独立占位（一个 N 可以既不是酸中心也不是碱中心；酸碱互不抢原子）
    ca, cb = set(), set()
    ha = count_groups(mol, am, ca)
    haw = count_groups(mol, aw, ca)
    hb = count_groups(mol, bm, cb)
    hbw = count_groups(mol, bw, cb)
    r = dict(n_acidic=sum(ha.values()), n_basic=sum(hb.values()),
             n_acidic_weak=sum(haw.values()), n_basic_weak=sum(hbw.values()),
             detail_酸="; ".join(f"{k}×{v}" for k, v in sorted(ha.items()))
                       + ("｜弱: " + "; ".join(f"{k}×{v}" for k, v in sorted(haw.items())) if haw else ""),
             detail_碱="; ".join(f"{k}×{v}" for k, v in sorted(hb.items()))
                       + ("｜弱: " + "; ".join(f"{k}×{v}" for k, v in sorted(hbw.items())) if hbw else ""),
             err="")
    cache[smiles] = r
    return r


def charge_type(q, n_acid, n_base):
    if q is None:
        return "★算不出"
    if q < -Q_EPS:
        return "anionic"
    if q > Q_EPS:
        return "cationic"
    return "zwitterionic" if (n_acid >= 1 and n_base >= 1) else "neutral"


# ══════════════════ 统计 ══════════════════
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


# ══════════════════ 件三：候选分界性质（★都从 SMILES 直接算） ══════════════════
def sep_feats(smiles, cache={}):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    RDLogger.DisableLog("rdApp.*")
    if smiles in cache:
        return cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        cache[smiles] = {}
        return {}
    ri = mol.GetRingInfo()
    naro = sum(1 for r in ri.AtomRings()
               if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    d = {"MW": Descriptors.MolWt(mol),
         "n_heavy": float(mol.GetNumHeavyAtoms()),
         "HBD": float(Lipinski.NumHDonors(mol)),
         "HBA": float(Lipinski.NumHAcceptors(mol)),
         "logP_crippen": Crippen.MolLogP(mol),
         "TPSA": Descriptors.TPSA(mol),
         "n_ring": float(ri.NumRings()),
         "n_aro_ring": float(naro)}
    cache[smiles] = d
    return d


def quart(vals):
    v = sorted(vals); n = len(v)
    return v[0], v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2, v[-1]


def main():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    os.makedirs(S34, exist_ok=True)
    rng = random.Random(SEED)

    print("=" * 78)
    print("S34-11 查证轮 · B 与 A/D 的分界绑在哪个可计算性质上（★只查不用）")
    print("=" * 78)
    print(f"★阈值写死：'净电荷≈0' = |q| < {Q_EPS}；n<{N_NOPOWER} 无统计功效；n<{N_NOSTAT} 不做统计")
    print(f"★bootstrap：nboot={NBOOT}, seed={SEED}, 整行(分子)重抽；ρ = Spearman")

    # ---------- 取 SMILES / chem_class ----------
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
        c = (r.get("chem_class") or "").strip()
        if ik and c:
            cls.setdefault(ik, c)

    # ---------- 件一 ----------
    print("\n" + "─" * 78)
    print("件一 · 给三把秤每个分子打【电荷型】标签（★只标，★不分组）")
    print("─" * 78)
    print("★酸性 SMARTS（★进分类）：")
    for lab, sma, _ in ACID_MAIN:
        print(f"    {lab:24s} {sma}")
    print("★酸性 SMARTS（★弱酸，单列，★不进分类）：")
    for lab, sma, _ in ACID_WEAK:
        print(f"    {lab:24s} {sma}")
    print("★碱性 SMARTS（★进分类，★优先级由上到下）：")
    for lab, sma, _ in BASE_MAIN:
        print(f"    {lab:24s} {sma}")
    print("★碱性 SMARTS（★弱碱，单列，★不进分类）：")
    for lab, sma, _ in BASE_WEAK:
        print(f"    {lab:24s} {sma}")

    rows = []
    for g in "ADB":
        for x in load_tsv(os.path.join(S34, f"scale_{g}_v2.tsv")):
            ik = x["inchikey_full"]
            smi = smi_g.get((g, ik)) or smi_any.get(ik, "")
            prof = ionizable_profile(smi) if smi else dict(
                n_acidic="", n_basic="", n_acidic_weak="", n_basic_weak="",
                detail_酸="", detail_碱="", err="★取不到 SMILES")
            qs = (x.get("net_charge_v2") or "").strip()
            q = int(float(qs)) if qs else None
            na = prof["n_acidic"] if prof["n_acidic"] != "" else None
            nb = prof["n_basic"] if prof["n_basic"] != "" else None
            ct = charge_type(q, na, nb) if (q is not None and na is not None) else "★算不出"
            rows.append({
                "scale": g, "molecule": x["molecule"], "inchikey_full": ik,
                "chem_class": cls.get(ik, "★未在 group_molecule_list"),
                "source": x.get("source", ""), "value": x.get("value", ""),
                "net_charge_v2": qs,
                "n_acidic": prof["n_acidic"], "n_basic": prof["n_basic"],
                "n_ionizable": ("" if na is None else na + nb),
                "n_acidic_weak": prof["n_acidic_weak"], "n_basic_weak": prof["n_basic_weak"],
                "charge_type": ct,
                "n_prim_amine_entry": x.get("n_prim_amine_entry", ""),
                "b_rotR": x.get("b_rotR", ""),
                "min_proj_area_v2_lowest": x.get("min_proj_area_v2_lowest", ""),
                "detail_酸": prof["detail_酸"], "detail_碱": prof["detail_碱"],
                "smiles_used": smi, "算不出原因": prof["err"],
            })

    print(f"\n★共 {len(rows)} 行（分子×秤）；★唯一分子 = "
          f"{len({r['inchikey_full'] for r in rows})}（★A∩D=5, A∩B=2 ⇒ 有分子在两把秤里各占一行）")

    print("\n▎必答问 1 —— 三把秤各自的电荷型构成（★逐把逐档报 n）")
    ORD = ["anionic", "cationic", "zwitterionic", "neutral", "★算不出"]
    for g in "ADB":
        sub = [r for r in rows if r["scale"] == g]
        c = collections.Counter(r["charge_type"] for r in sub)
        print(f"  秤 {g}（n={len(sub)}）： " + "  ".join(f"{k}={c.get(k,0)}" for k in ORD))
        nz = [r for r in sub if r["charge_type"] == "neutral"]
        z0 = sum(1 for r in nz if r["n_ionizable"] == 0)
        print(f"      └ neutral 档细分：n_ionizable==0 → {z0} 个；>0 → {len(nz)-z0} 个")

    print("\n▎必答问 2 —— A/D 里的 zwitterionic 与 neutral（★逐个列出分子名）")
    for g in "AD":
        for t in ("zwitterionic", "neutral"):
            sub = [r for r in rows if r["scale"] == g and r["charge_type"] == t]
            print(f"  秤 {g} · {t}（n={len(sub)}）：")
            if not sub:
                print("      ★★一个都没有")
            for r in sub:
                print(f"      {r['molecule'][:30]:30s} q={r['net_charge_v2']:>3s} "
                      f"酸={r['n_acidic']} 碱={r['n_basic']} 弱酸={r['n_acidic_weak']} "
                      f"弱碱={r['n_basic_weak']}  [{r['chem_class']}]")

    p1 = os.path.join(S34, "molecule_charge_type_x.tsv")
    with open(p1, "w", encoding="utf-8", newline="") as fh:
        fh.write("# molecule_charge_type_x.tsv -- S34-11 件一：一行 = 一个【分子×秤】。\n")
        fh.write("# ★★查证轮产物，★_x 后缀，★不覆盖任何 _v2/_v3/_w。\n")
        fh.write("# ★net_charge_v2 直接取自 scale_{A,D,B}_v2.tsv（★已有的不重算）。\n")
        fh.write(f"# ★'净电荷≈0' 阈值写死 |q| < {Q_EPS}（net_charge_v2 为整数 ⇒ 等价 q==0）。\n")
        fh.write("# ★电荷型四档：anionic q<-0.5 / cationic q>+0.5 /\n")
        fh.write("#   zwitterionic |q|<0.5 且 n_acidic>=1 且 n_basic>=1 / neutral |q|<0.5 且不满足上一条。\n")
        fh.write("# ★n_acidic/n_basic 只计任务书点名的强档；弱酸弱碱单列 n_*_weak，★不进分类。\n")
        fh.write("# ★一切分类用 SMARTS 结构判据，★绝不按分子名猜；SMARTS 逐条见 charge_boundary_report.md。\n")
        fh.write(f"# 行数：{len(rows)}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p1}  ({len(rows)} 行)")

    # ---------- 件二 ----------
    print("\n" + "─" * 78)
    print("件二 · A/D 里净电荷≈0 的那些，符号翻不翻")
    print("─" * 78)
    print("⚠️★A 组 4 个来源、D 组 1 个来源，两把秤的终点刻度不同（A −1.18~137.28 / D 5~216）")
    print("  ⇒ ★★【不】把 A+D 并成一池算 ρ（★跨 l1_unit 不可并列，★纪律不变）；★A 与 D 各自分开做。")

    # ★★"抗生素" 一律用【结构判据 chem_class】认，★绝不按分子名猜：
    #   抗生素侧 = β-内酰胺 / 喹诺酮类 / 氨基糖苷；非抗生素侧 = 氨基酸·多胺 / 糖·糖醇·糖酸 / 肽 / 核苷酸 / 其他
    AB_CLS = {"β-内酰胺", "喹诺酮类（含无氟萘啶酮/吡啶并嘧啶）", "氨基糖苷"}

    sign_rows = []
    for g in "ADB":
        sub = [r for r in rows if r["scale"] == g]
        z0 = [r for r in sub if r["charge_type"] in ("zwitterionic", "neutral")]
        grp = {"带电(anionic+cationic)": [r for r in sub if r["charge_type"] in ("anionic", "cationic")],
               "净额≈0(zwitter+neutral)": z0,
               "└ zwitterionic": [r for r in sub if r["charge_type"] == "zwitterionic"],
               "└ neutral": [r for r in sub if r["charge_type"] == "neutral"],
               # ★★备查：任务书立论所依赖的那组 —— "净额≈0 且【是抗生素】"（★A 组内部现成的对照）
               "★备查 净额≈0 且是抗生素": [r for r in z0 if r["chem_class"] in AB_CLS],
               "★备查 净额≈0 且非抗生素": [r for r in z0 if r["chem_class"] not in AB_CLS]}
        for gname, rs in grp.items():
            for feat in ("n_prim_amine_entry", "b_rotR"):
                n = len(rs)
                rec = {"scale": g, "组": gname, "feature": feat, "n": n}
                xs = [r[feat] for r in rs]
                ys = [r["value"] for r in rs]
                ok = all(a.strip() and b.strip() for a, b in zip(xs, ys))
                if n < N_NOSTAT or not ok:
                    rec.update({"rho": "", "ci_lo": "", "ci_hi": "", "符号": "",
                                "统计功效": f"★★n={n} < {N_NOSTAT} ⇒ ★不做统计，做不了"
                                            if n < N_NOSTAT else "★有空值 ⇒ 做不了"})
                else:
                    x = [float(a) for a in xs]; y = [float(b) for b in ys]
                    if len(set(x)) < 2:
                        rec.update({"rho": "", "ci_lo": "", "ci_hi": "", "符号": "",
                                    "统计功效": f"★该组内 {feat} 方差为 0 ⇒ ρ 算不出"})
                    else:
                        rho, lo, hi = boot_ci(x, y, random.Random(SEED))
                        rec.update({"rho": f"{rho:+.4f}",
                                    "ci_lo": "" if lo is None else f"{lo:+.4f}",
                                    "ci_hi": "" if hi is None else f"{hi:+.4f}",
                                    "符号": "正" if rho > 0 else ("负" if rho < 0 else "0"),
                                    "统计功效": "★★无统计功效(n<10)" if n < N_NOPOWER else "n>=10"})
                sign_rows.append(rec)

    print("\n▎必答问 1 —— 净额≈0 组里 n_prim_amine 与 b_rotR 的符号（★逐个报 ρ 与 CI）")
    print("  ★B 秤一并列出（★B 全部 |q|<0.5，★但内部裂成 糖=neutral / 肽=zwitterionic 两半）")
    print("  ★带★备查 的两行 = 净额≈0 组再按【结构判据 chem_class】切抗生素/非抗生素，★只报不立论")
    print(f"  {'秤':3s} {'组':26s} {'构件':20s} {'n':>3s} {'ρ':>9s} {'95%CI':>21s}  功效")
    for r in sign_rows:
        ci = f"[{r['ci_lo']}, {r['ci_hi']}]" if r["rho"] else "—"
        print(f"  {r['scale']:3s} {r['组']:26s} {r['feature']:20s} {r['n']:>3d} "
              f"{r['rho'] or '—':>9s} {ci:>21s}  {r['统计功效']}")

    p2 = os.path.join(S34, "sign_by_chargetype_x.tsv")
    with open(p2, "w", encoding="utf-8", newline="") as fh:
        fh.write("# sign_by_chargetype_x.tsv -- S34-11 件二：一行 = 一个（秤,组,构件）。\n")
        fh.write("# ★★查证轮产物，★_x 后缀。★A 与 D 分开做，★★不把 A+D 并成一池（跨 l1_unit 不可并列）。\n")
        fh.write(f"# ★bootstrap nboot={NBOOT} seed={SEED} 整行重抽；ρ = Spearman(构件, 终点 value)。\n")
        fh.write(f"# ★n<{N_NOPOWER} 标'无统计功效'；★n<{N_NOSTAT} 不做统计。★★不因 n 小而合并分组或改阈值。\n")
        fh.write(f"# 行数：{len(sign_rows)}\n")
        w = csv.DictWriter(fh, fieldnames=list(sign_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(sign_rows)
    print(f"\nwrote {p2}  ({len(sign_rows)} 行)")

    # ---------- 件三 ----------
    print("\n" + "─" * 78)
    print("件三 · 分界还能绑在哪个可计算性质上（★候选清单，★不是结论）")
    print("─" * 78)
    ad_ik, b_ik = {}, {}
    for r in rows:
        (b_ik if r["scale"] == "B" else ad_ik).setdefault(r["inchikey_full"], r)
    shared = sorted(set(ad_ik) & set(b_ik))
    print(f"★A∪D 唯一分子 = {len(ad_ik)}；B 唯一分子 = {len(b_ik)}；"
          f"★★两边【都在】的 = {len(shared)}：{[b_ik[k]['molecule'] for k in shared]}")
    print("  ⇒ ★★这几个分子同时属于两边 ⇒ ★★★任何性质都【不可能】把两边完全分开（逻辑上）")
    print("  ⇒ ★故完全分离判定跑两遍：★口径甲=照原样（含重叠分子）；★口径乙=把重叠分子从两边都剔除")

    CAND = [("MW", "分子量 (Descriptors.MolWt)"),
            ("n_heavy", "重原子数 (GetNumHeavyAtoms)"),
            ("n_ionizable", "可电离基团总数 (承件一 n_acidic+n_basic)"),
            ("HBD", "氢键供体 (Lipinski.NumHDonors)"),
            ("HBA", "氢键受体 (Lipinski.NumHAcceptors)"),
            ("logP_crippen", "logP (Crippen.MolLogP，★输入串原态；★logD 见表头 ⚠️ 两条)"),
            ("TPSA", "TPSA (Descriptors.TPSA)"),
            ("n_ring", "环个数 (RingInfo.NumRings)"),
            ("n_aro_ring", "芳香环个数 (全芳香原子的环)"),
            ("b_rotR", "旋转键比例 b_rotR (★承既有，取自 scale_*_v2.tsv)")]

    def val_of(rec, key):
        if key == "n_ionizable":
            return None if rec["n_ionizable"] == "" else float(rec["n_ionizable"])
        if key == "b_rotR":
            s = (rec["b_rotR"] or "").strip()
            return float(s) if s else None
        d = sep_feats(rec["smiles_used"]) if rec["smiles_used"] else {}
        return d.get(key)

    cand_rows = []
    for key, desc in CAND:
        for mode, drop in (("甲 含重叠分子", set()), ("乙 剔除重叠分子", set(shared))):
            ad = [v for k, r in ad_ik.items() if k not in drop
                  for v in [val_of(r, key)] if v is not None]
            bb = [v for k, r in b_ik.items() if k not in drop
                  for v in [val_of(r, key)] if v is not None]
            if len(ad) < 2 or len(bb) < 2:
                cand_rows.append({"候选性质": key, "口径": mode, "说明": desc,
                                  "n_AD": len(ad), "n_B": len(bb),
                                  "AD_min": "", "AD_中位": "", "AD_max": "",
                                  "B_min": "", "B_中位": "", "B_max": "",
                                  "完全分开": "★算不出", "B嵌在A∪D区间内": "", "重叠区分子数": "",
                                  "最佳单阈值": "", "阈值方向": "", "最少错分数": "",
                                  "总分子数": len(ad) + len(bb),
                                  "基线错分(全判A∪D)": "", "比基线少错": "", "重叠区分子": ""})
                continue
            a0, am, a1 = quart(ad); b0, bm, b1 = quart(bb)
            full = (a1 < b0) or (b1 < a0)
            # ★★补一列（★只报不立论）：单阈值最好能分到什么地步 —— 扫所有中点，取错分最少
            cuts = sorted(set(ad + bb))
            best = None
            for i in range(len(cuts)):
                t = cuts[i]
                for direc in ("B<=t", "B>t"):
                    if direc == "B<=t":
                        err = sum(1 for v in bb if v > t) + sum(1 for v in ad if v <= t)
                    else:
                        err = sum(1 for v in bb if v <= t) + sum(1 for v in ad if v > t)
                    if best is None or err < best[0]:
                        best = (err, t, direc)
            best_err, best_t, best_dir = best
            # 重叠区 = 两边取值区间的交集
            lo, hi = max(a0, b0), min(a1, b1)
            ov_names = []
            if not full:
                for k, r in list(ad_ik.items()) + list(b_ik.items()):
                    if k in drop:
                        continue
                    v = val_of(r, key)
                    if v is not None and lo <= v <= hi:
                        ov_names.append(f"{r['molecule']}[{r['scale']}]")
            cand_rows.append({
                "候选性质": key, "口径": mode, "说明": desc,
                "n_AD": len(ad), "n_B": len(bb),
                "AD_min": f"{a0:.4g}", "AD_中位": f"{am:.4g}", "AD_max": f"{a1:.4g}",
                "B_min": f"{b0:.4g}", "B_中位": f"{bm:.4g}", "B_max": f"{b1:.4g}",
                "完全分开": "★★是(无重叠)" if full else "否",
                "B嵌在A∪D区间内": "★★是(B 完全被包住)" if (a0 <= b0 and b1 <= a1) else "否",
                "重叠区分子数": len(ov_names),
                "最佳单阈值": f"{best_t:.4g}", "阈值方向": best_dir,
                "最少错分数": best_err, "总分子数": len(ad) + len(bb),
                "基线错分(全判A∪D)": min(len(ad), len(bb)),
                "比基线少错": min(len(ad), len(bb)) - best_err,
                "重叠区分子": "; ".join(sorted(ov_names))})

    print(f"\n{'候选':16s} {'口径':16s} {'nAD':>4s} {'nB':>3s} "
          f"{'A∪D min/中/max':>26s} {'B min/中/max':>26s}  完全分开  重叠数")
    for r in cand_rows:
        print(f"{r['候选性质']:16s} {r['口径']:16s} {r['n_AD']:>4} {r['n_B']:>3} "
              f"{r['AD_min']+'/'+r['AD_中位']+'/'+r['AD_max']:>26s} "
              f"{r['B_min']+'/'+r['B_中位']+'/'+r['B_max']:>26s}  "
              f"{r['完全分开']:>10s}  {r['重叠区分子数']:>3}  "
              f"最佳阈值 {r['最佳单阈值']:>8s} {r['阈值方向']:>5s} ⇒ 错分 {r['最少错分数']}/{r['总分子数']}"
              f"（基线 {r['基线错分(全判A∪D)']} ⇒ 比基线少错 {r['比基线少错']}）")

    print("\n★★附带一条事实（★只报不立论）：B 的取值区间是否【完全嵌在】A∪D 区间内")
    for r in cand_rows:
        print(f"    [{r['口径']}] {r['候选性质']:14s} {r['B嵌在A∪D区间内']}")

    print("\n▎必答问 1 —— 有几个候选能把 B 与 A/D 完全分开（无重叠）")
    for mode in ("甲 含重叠分子", "乙 剔除重叠分子"):
        wins = [r["候选性质"] for r in cand_rows if r["口径"] == mode and r["完全分开"].startswith("★★是")]
        print(f"  口径{mode}：{len(wins)} 个 —— {wins if wins else '★一个都没有'}")

    print("\n▎必答问 2 —— 只能部分分开的，重叠区有几个分子（★逐个报）")
    for r in cand_rows:
        if r["完全分开"] == "否":
            print(f"  [{r['口径']}] {r['候选性质']:14s} 重叠区 n={r['重叠区分子数']}  "
                  f"{r['重叠区分子'][:150]}")

    p3 = os.path.join(S34, "separator_candidates_x.tsv")
    with open(p3, "w", encoding="utf-8", newline="") as fh:
        fh.write("# separator_candidates_x.tsv -- S34-11 件三：一行 = 一个（候选性质 × 口径）。\n")
        fh.write("# ★★查证轮产物，★_x 后缀。★候选一律从 SMILES 直接算，★不用任何功能标签。\n")
        fh.write("# ★★A∪D 与 B 有共有分子（glucose / sucrose 两边都在）\n")
        fh.write("#   ⇒ 口径甲=照原样（★完全分离在逻辑上不可能）；口径乙=把共有分子从两边都剔除。\n")
        fh.write("# ★'完全分开' = 两边取值区间无交集；'重叠区分子' = 落在两区间交集里的分子。\n")
        fh.write("# ⚠️ logD 的盘上现状（★实测，★S34-11 件三改口）：\n")
        fh.write("#   ① 盘上【有】一列 logD —— candidates/features_serf5000.tsv 的 `cx_logD`（4997/5000 有值）；\n")
        fh.write("#      ★但它覆盖本轮 94 个分子中的 **0 个**（★按规范化 SMILES 逐个比对），★且 pH 口径盘上无任何\n")
        fh.write("#      脚本或报告引用过 ⇒ ★★不知道是不是 7.4 ⇒ ★本表不用它。\n")
        fh.write("#   ② 盘上【没有】能给新分子算 logD 的工具（c7_run_l_step0_gate 已核：OpenChemLib 有\n")
        fh.write("#      CLogPPredictor，★无 pKa、★无 logD）。\n")
        fh.write("#   ⇒ 本表 logP 用 Crippen.MolLogP（★输入串原态，★与既有 #9 口径同实现）。\n")
        fh.write(f"# 行数：{len(cand_rows)}\n")
        w = csv.DictWriter(fh, fieldnames=list(cand_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(cand_rows)
    print(f"\nwrote {p3}  ({len(cand_rows)} 行)")

    # ---------- 件四 ----------
    print("\n" + "─" * 78)
    print('件四 · "是不是抗生素"这条路的现状（★只报事实，★不立论）')
    print("─" * 78)
    cc = collections.Counter(r["chem_class"] for r in rows)
    print("★三把秤合起来的 chem_class 构成（分子×秤 计）：")
    for k, v in cc.most_common():
        print(f"    {k:24s} {v}")
    print("\n★逐把秤：")
    for g in "ADB":
        c = collections.Counter(r["chem_class"] for r in rows if r["scale"] == g)
        print(f"  {g}: " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
    nocls = [r["molecule"] for r in rows if r["chem_class"] in ("其他", "未判（无结构）",
                                                               "★未在 group_molecule_list")]
    print(f"\n★chem_class 落到「其他 / 未判 / 取不到」的：{len(nocls)} —— {sorted(set(nocls))}")

    print(f"\nwrote 件四（无独立 tsv，答写进 charge_boundary_report.md）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
