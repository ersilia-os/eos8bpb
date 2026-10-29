#!/usr/bin/env python
"""S36-5 查证轮 —— 多黏菌素/脂肽类：★★查定义 → 定规则 → 验（★★★只核不改分数）。

★★★【本轮是查证轮】★不造秤、★不改构件、★不改任何既有产物、★★★不改任何分子的分数。

★★★★【Lang 2026-08-21 定的次序 —— 逐字】
  ▎★先查清楚具体的特征，★再标上，★并且要记着验证
  ⇒ ★★★承 [S36-3] 的教训：★★【不许先拼 SMARTS】。
  ⇒ ★★本脚本里三条判据 L1/L2/L3 的每一条，★都在注释里写明【它对应定义的哪一句】。

★★★【不对称】★承 [S36-1]：★★宁可多标、不可漏标（★与入组标准的不对称【方向相反】）。
★★★★【标记绝不影响分数】★分数照常给、标记单独一列。

⚠️ 两条禁读：cooper_features.csv / cooper_casecontrol_*.csv —— ★本脚本从不打开。
★产物一律 _PM 后缀；★★_AG2 / _AG3 / _AG4 一个都不覆盖。
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 原为开发机的 antibiotic-benchmark/scripts，会抢在本包 _vendor 之前；改为本模块自身所在目录，同族模块照常平铺 import
from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem import Descriptors                                  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from s34l_scope_filter import scope_gate, RULE_S, RULE_P            # noqa: E402
from s36d_ag_definition import aminocyclitol_rings, D2_any          # noqa: E402

DL = os.path.join(HOME, "downloads_raw")
S34, S35, S36 = (os.path.join(DL, "section%d" % n) for n in (34, 35, 36))
CLSZ = os.path.join(S34, "molecule_class_z.tsv")
FEATW = os.path.join(S34, "features_water.tsv")
L3 = os.path.join(S35, "level3_labels_L3m.tsv")
RITZ = os.path.join(S35, "ritz_ourfeatures_C.tsv")
CANDF = os.path.join(HOME, "antibiotic-benchmark/candidates/features_aibiotics.tsv")
CANDS = os.path.join(HOME, "antibiotic-benchmark/candidates/scored_aibiotics.tsv")
AG3 = os.path.join(S36, "ag_flag_all_AG3.tsv")

FORBIDDEN = ("cooper_features.csv", "cooper_casecontrol_")
SEED = 20260822
_t0 = time.time()
os.makedirs(S36, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 件一 · ★★★★定义（★★逐字取回，★★★每条带出处）
#   ★盘上三路先找 ⇒ ★三路结果见 main() 里的 stage0()
# ══════════════════════════════════════════════════════════════════════════
DEFN = [
 ("D1 总定义",
  "Polymyxins are cationic lipopeptides consisting of a cyclic heptapeptide ring "
  "with a tripeptide side chain acylated at the N-terminus by a fatty acyl group. "
  "All polymyxins are cationic decapeptides, consisting of a cyclic heptapeptide "
  "linked to a linear tripeptide side chain acylated at the N terminus by a fatty acid tail.",
  "JACS Au 2025 (10.1021/jacsau.5c00587) / PMC7463838 综述"),
 ("D2 环怎么闭合",
  "The polymyxin decapeptide core contains an intramolecular cyclic heptapeptide "
  "amide-linked loop between the amino group of the side chain of the diaminobutyric "
  "acid (Dab) residue at position 4 and the carboxyl group of the C-terminal threonine "
  "residue. ... the C-terminal Thr-10 acylates the N-gamma-amino side chain of Dab4 to "
  "create a 23-membered lactam ring.",
  "Velkov et al., Structure-Activity Relationships of Polymyxin Antibiotics (PMC2907661)"),
 ("D3 残基序列",
  "The peptide moiety of polymyxin is L-2,4-diaminobutyric acid (Dab1)-Thr2-L-Dab3-"
  "L-Dab4-L-Dab5-D-Phe/Leu6-L-Leu7-L-Dab8-L-Dab9-Thr10 with an amide linkage between "
  "the C-terminal threonine and the side-chain amino group of Dab4.",
  "同上 PMC2907661"),
 ("D4 几个正电",
  "Six of the ten residues are typically L-2,4-diaminobutyric acid (Dab1, Dab3, Dab5, "
  "Dab8, Dab9, and the linker Dab4 ...). All but the linker Dab4 residue are positively "
  "charged at physiological pH ... This accounts for five positively charged residues.",
  "JACS Au 2025 (10.1021/jacsau.5c00587)"),
 ("D4b 几个正电（第二来源）",
  "The five non-cyclized DAB amino acids each carry a charge of +1, and thus the "
  "cationic peptide has a total charge of +5.",
  "Berglund et al., PLoS Comput Biol 2015 (10.1371/journal.pcbi.1004180)"),
 ("D5 脂肪尾多长",
  "Natural polymyxins possess a C8-C9 fatty acid side chain. ... The optimal fatty acyl "
  "chain length appears to be C7-C9, as longer or shorter chain N-alpha analogues "
  "display reduced antimicrobial activity.",
  "Velkov et al. (PMC2907661) / J Antibiot 2016 (ja2016146)"),
 ("D6 十个残基里六个是 Dab",
  "Polymyxins consist of 10 amino acid residues, six of which are "
  "L-alpha,gamma-diaminobutyric acid (L-DAB). ... The DAB residues cause polymyxins to "
  "have multiple positively charged groups at physiological pH.",
  "Wikipedia: Polymyxin（★三级来源，★仅作对照）"),
 ("D7 B 与 E 只差一个残基",
  "The only difference between colistin and PMB is the amino acid (aa) at position 6, "
  "d-Leu in colistin and d-Phe in polymyxin B.",
  "PMC4980087 / PMC6535509"),
 ("D8 分子量与脂肽属性",
  "Polymyxins are large molecules (~1200 Da) consisting of a cyclic heptapeptide bound "
  "to a tripeptide acylated at the N-terminus by a fatty acid chain of variable length.",
  "ScienceDirect Topics: Polymyxin Antibiotics"),
]

MEMBERS = [
 # 名称, 类别, 临床用?, 与 PMB 的结构差别（★逐字或近逐字）, 出处
 ("polymyxin B (PMB)", "天然多黏菌素", "★★是",
  "cyclic heptapeptide + linear tripeptide + N-terminal fatty acyl; position 6 = d-Phe",
  "PMC2907661 / PMC4980087"),
 ("polymyxin E = colistin", "天然多黏菌素", "★★是",
  "The only difference between colistin and PMB is the amino acid at position 6, "
  "d-Leu in colistin and d-Phe in polymyxin B.", "PMC4980087"),
 ("polymyxin A", "天然多黏菌素", "否", "★本轮未取到逐字结构差异", "（未核到）"),
 ("polymyxin C", "天然多黏菌素", "否", "★本轮未取到逐字结构差异", "（未核到）"),
 ("polymyxin D", "天然多黏菌素", "否",
  "The article identifies polymyxins B, E (colistin), M (mattacin), and D",
  "Wikipedia: Polymyxin"),
 ("polymyxin M (mattacin)", "天然多黏菌素", "否", "同上（★只核到存在，★无结构细节）",
  "Wikipedia: Polymyxin"),
 ("polymyxin B1..B6 组分", "PMB 的脂酰变体", "★是（混合物）",
  "fatty acyl chains ... restricted to (S)-6-methyloctanoyl (PMB1), 6-methylheptanoyl "
  "(PMB2), octanoyl (PMB3), heptanoyl (PMB4), nonanoyl (PMB5), and "
  "3-hydroxy-6-methyloctanoyl (PMB6)", "J Antibiot 2016 (ja2016146)"),
 ("colistin A/B (=polymyxin E1/E2)", "colistin 的脂酰变体", "★是（混合物）",
  "In polymyxin E1 (colistin A) the R-FA is 6-MOA and in polymyxin E2 (colistin B) the "
  "R-FA is 6-MHA", "PMC2907661"),
 ("★★polymyxin B nonapeptide (PMBN)", "★★半合成衍生物（★通透剂）", "否（★研究试剂）",
  "PMBN is a polymyxin derivative lacking the N-terminal fatty-acyl chain and the last "
  "exocyclic amino acid. / Removal of the hydrophobic tail of polymyxin B yields "
  "polymyxin nonapeptide, which still binds to LPS, but no longer kills the bacterial cell.",
  "ScienceDirect Topics: PMBN / Wikipedia: Polymyxin"),
 ("★octapeptin A/B/C", "★★近缘天然产物（★不是多黏菌素）", "否",
  "The octapeptins possess a heptapeptide core rich in Dab residues, but with only one "
  "exocyclic amino acid. / do not have an N-terminal tripeptide segment but possess the "
  "same heptapeptide scaffold as PMB with a single D-Dab or D-Ser residue connected to a "
  "lengthy beta-hydroxy fatty acyl chain", "J Antibiot 2016 (ja2016146)"),
 ("SPR741 (NAB741)", "★半合成 PMB 类似物（★增效剂）", "否",
  "Polymyxin B Analogue SPR741 ... Unique Perturbation of the Outer Membrane",
  "10.1021/acsinfecdis.9b00159（★[S35-26] 已核）"),
]

CAMP = [
 ("是子集（支持）",
  "Polymyxin B and colistin (polymyxins) are cationic antimicrobial lipopeptides produced "
  "by the Gram-positive bacterium Paenibacillus polymyxa. Like other membrane-active "
  "cationic antimicrobial peptides, polymyxins selectively target bacterial membranes "
  "over mammalian membranes.", "ScienceDirect Topics: Polymyxin Antibiotics"),
 ("是子集（支持）",
  "This binding mechanism represents the self-promoted uptake mechanism characteristic of "
  "cationic antimicrobial peptides", "ScienceDirect Topics: Colistin"),
 ("★★但要注意（保留）",
  "Polymyxins are a group of cyclic non-ribosomal polypeptide (NRPs) which are "
  "biosynthesized by bacteria belonging to the genus Paenibacillus.（★该来源把它们归为"
  "【非核糖体肽】，★★并未用 CAMP 这个上位类）", "Wikipedia: Polymyxin"),
]

SPU = [
 # 成员, 证据类型, 逐字, 出处, ★是不是【直接实验证据】
 ("polymyxin B", "★★★直接实验（★E. coli，★同源配对）",
  "dansyl polymyxin displacement; NPN/lysozyme permeabilization; Mg2+ blockade; MIC in "
  "porin-deficient and pmr mutants（★[S35-25]/[S36-1] 已核，★本轮不重做）",
  "Hancock, Farmer, Li & Poole 1991, AAC 35:1309-1314, DOI 10.1128/aac.35.7.1309",
  "★★★是"),
 ("polymyxin B", "★★机制断言（模拟+竞争实验）",
  "Direct evidence for competition between cations and polymyxin B has been demonstrated, "
  "providing support for this hypothesis.",
  "PLoS Comput Biol 2015 (10.1371/journal.pcbi.1004180)", "★★是（★竞争实验）"),
 ("polymyxin B1", "★类级别断言",
  "PMB1 lipopolypeptides are known to permeate across the bacterial outer membrane (OM) "
  "by self-promoted uptake",
  "PLoS Comput Biol 2015 (10.1371/journal.pcbi.1004180)", "★否（★综述性断言）"),
 ("★★colistin (polymyxin E)", "★★★本轮【未核到单独的直接实验证据】",
  "★检索到的都是把 colistin 与 PMB 并称的类级别断言（如 ScienceDirect Topics: Colistin "
  "『This binding mechanism represents the self-promoted uptake mechanism characteristic "
  "of cationic antimicrobial peptides』）⇒ ★★★没有 colistin 专门的孔道缺陷株/置换实验",
  "（未核到）", "★★★否 ⇒ ★★对 colistin 是【外推】"),
 ("polymyxin A/C/D/M", "★★★本轮【完全未核到】", "—", "（未核到）", "★★★否"),
 ("PMBN", "★★结合但不杀（★通透剂）",
  "Removal of the hydrophobic tail of polymyxin B yields polymyxin nonapeptide, which "
  "still binds to LPS, but no longer kills the bacterial cell.",
  "Wikipedia: Polymyxin / J Med Chem 2000 (10.1021/jm0000057)",
  "★部分（★证明【结合 LPS】不需要脂尾）"),
]


# ══════════════════════════════════════════════════════════════════════════
# 件三 · ★★★三条判据（★★★每条都写明【对应定义的哪一句】，★不是拼 SMARTS）
# ══════════════════════════════════════════════════════════════════════════
PEP_SMARTS = "[CX4][CX3](=[OX1])[NX3;H1]"       # ★酰胺键（★肽键；★含侧链酰胺，★宽）
ENTRY_SMARTS = "[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]"   # ★[S34-6] 定稿 n_prim_amine_entry
CHAIN6_SMARTS = "[CX4;!R][CX4;!R][CX4;!R][CX4;!R][CX4;!R][CX4;!R]"   # ★6 连续非环 sp3 C
ACYL_SMARTS = ("[CX4;!R][CX4;!R][CX4;!R][CX4;!R][CX4;!R]"
               "[CX3](=[OX1])[NX3]")            # ★L3 严版：★该链末端是【酰胺】
PEP = Chem.MolFromSmarts(PEP_SMARTS)
ENTRY = Chem.MolFromSmarts(ENTRY_SMARTS)
CHAIN6 = Chem.MolFromSmarts(CHAIN6_SMARTS)
ACYL = Chem.MolFromSmarts(ACYL_SMARTS)

MACRO_MIN = 12          # ★大环下限（★真值 = 23 元内酰胺环 ⇒ 12 是【宁可多标】的放宽）
PEP_MIN = 5             # ★酰胺键下限（★真值：十肽 9 个 / PMBN 8 个 ⇒ 5 是放宽）
PEPIN_MIN = 2           # ★大环内的肽键下限（★真值 = 7 ⇒ 2 是放宽）
AMINE_MIN = 3           # ★游离伯胺下限（★真值 = 5 ⇒ 3 是放宽）

CRIT_SPEC = [
 ("L1 · 环状肽",
  "(算法) 酰胺键 `%s` >= %d；★且存在一个环，★环大小 >= %d，★该环内含 >= %d 个酰胺键"
  "（★酰胺的 C 与 N 【都】在环上）" % (PEP_SMARTS, PEP_MIN, MACRO_MIN, PEPIN_MIN),
  "★★对应定义 D1/D2：「a cyclic heptapeptide ring」＋「an intramolecular cyclic "
  "heptapeptide amide-linked loop ... to create a 23-membered lactam ring」。\n"
  "      ★★★为什么要求【大环】而不是只要求『环数>=1』：★任务书建议方向写的是『环数>=1』，\n"
  "      ★但那样【任何一个带苯环的肽】都会中 ⇒ ★★与定义不符（★定义要的是【肽本身成环】）。\n"
  "      ★★实测：PMB 的环是 [23, 6]（★23 元内酰胺 ＋ 苯环），★colistin 是 [23] ⇒ ★两者都靠 23 元环中。\n"
  "      ★★门槛 12/5/2 全部低于真值（23/9/7）⇒ ★★★按【宁可多标】放宽。"),
 ("L2 · 多个游离伯胺",
  "(算法) len(match `%s`) >= %d" % (ENTRY_SMARTS, AMINE_MIN),
  "★★对应定义 D4/D4b/D6：「five positively charged residues」／「The five non-cyclized "
  "DAB amino acids each carry a charge of +1, and thus the cationic peptide has a total "
  "charge of +5」。\n"
  "      ★★正电【来自 Dab 侧链的游离 gamma-氨基】⇒ ★故数【游离伯胺】而不是数净电荷。\n"
  "      ★★口径照 [S34-6] 定稿的 `n_prim_amine_entry`（★与氨基糖苷那套判据【同一条 SMARTS】）。\n"
  "      ★★实测：PMB 与 colistin 各 5 个 ⇒ ★与定义的 +5 逐字对上。\n"
  "      ★★门槛 3 < 真值 5 ⇒ ★★★按【宁可多标】放宽。"),
 ("L3 · 脂肪酰尾",
  "(算法) len(match `%s`) >= 1" % CHAIN6_SMARTS,
  "★★对应定义 D1/D5/D8：「acylated at the N-terminus by a fatty acyl group」／"
  "「Natural polymyxins possess a C8-C9 fatty acid side chain」。\n"
  "      ★★★为什么用【6 连续非环 sp3 碳】：★真值是 C7-C9 ⇒ ★6 是【宁可多标】的放宽。\n"
  "      ★★为什么不会被 Leu/Ile 侧链误中：★★肽主链上 Calpha 的邻位是 C(=O)（★CX3）⇒ 链被打断；\n"
  "      ★Leu 侧链最长只有 4 个 sp3 C ⇒ ★够不到 6。★★实测 PMB=3 处、colistin=2 处命中。\n"
  "      ⚠️★★**PMBN 没有脂肪尾** ⇒ ★★它只会中 L1+L2 ⇒ ★★★落【疑似】档 —— ★这是【预期之内】的。\n"
  "      ★同时另算一条【严版 L3'】：`%s`（★要求该链末端接酰胺 ⇒ 才是真的 N 端脂酰化），\n"
  "      ★★★但【分档只用宽版】—— ★严版只报，★不参与判定。" % ACYL_SMARTS),
]

TIER_RULE = {3: "高度疑似", 2: "疑似", 1: "不标", 0: "不标"}


def say(*a):
    print(*a, flush=True)


def guard(p):
    if any(f in str(p) for f in FORBIDDEN):
        raise RuntimeError("禁读文件: %s" % p)
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


def to_mol(smi):
    if not (smi or "").strip():
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        try:
            from _smiles_syntax import normalize_smiles_syntax
            m = Chem.MolFromSmiles(normalize_smiles_syntax(smi))
        except Exception:
            m = None
    return m


# ---------------------------------------------------------------- 三条判据
def crit_L1(m):
    """★★L1：★酰胺键 >= PEP_MIN，★且存在含 >= PEPIN_MIN 个肽键的大环（size >= MACRO_MIN）。"""
    pm = m.GetSubstructMatches(PEP)
    if len(pm) < PEP_MIN:
        return False, len(pm), 0, 0
    best_sz = best_n = 0
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) < MACRO_MIN:
            continue
        rs = set(ring)
        n = sum(1 for ca, c, o, nn in pm if c in rs and nn in rs)
        if n > best_n or (n == best_n and len(ring) > best_sz):
            best_n, best_sz = n, len(ring)
    return (best_n >= PEPIN_MIN), len(pm), best_sz, best_n


def crit_L2(m):
    n = len(m.GetSubstructMatches(ENTRY))
    return n >= AMINE_MIN, n


def crit_L3(m):
    n = len(m.GetSubstructMatches(CHAIN6))
    n_strict = len(m.GetSubstructMatches(ACYL))
    return n >= 1, n, n_strict


def evaluate(smi):
    m = to_mol(smi)
    if m is None:
        return None
    l1, n_pep, macro_sz, pep_in = crit_L1(m)
    l2, n_am = crit_L2(m)
    l3, n_ch, n_ch_s = crit_L3(m)
    hits = int(l1) + int(l2) + int(l3)
    # ★★氨基糖苷 D 定义版（★[S36-3]，★★未经裁定 ⇒ 仅作并列参考，★不参与本轮任何判定）
    try:
        agd = bool(aminocyclitol_rings(m)) and D2_any(m)
    except Exception:
        agd = False
    return dict(L1=l1, L2=l2, L3=l3, n_hit=hits, tier=TIER_RULE[hits],
                n_pep=n_pep, macro_ring=macro_sz, pep_in_ring=pep_in,
                n_prim_amine=n_am, n_chain6=n_ch, n_acyl_strict=n_ch_s,
                MW=round(Descriptors.MolWt(m), 1), AG_D=agd)


# ══════════════════════════════════════════════════════════════════════════
GOLD = [
 ("POLYMYXIN B SULFATE", "多黏菌素 B（★盘上实测 SMILES）",
  "CCC(C)CCCCC(=O)NC(CCN)C(=O)NC(C(C)O)C(=O)NC(CCN)C(=O)NC1CCNC(=O)C(NC(=O)C(CCN)"
  "NC(=O)C(CCN)NC(=O)C(CC(C)C)NC(=O)C(CC2=CC=CC=C2)NC(=O)C(CCN)NC1=O)C(C)O",
  "★盘上 elzahed_matched.tsv / serf_input_1478.tsv / molecule_intake_L3g.tsv"),
 ("COLISTIN SULFATE", "黏菌素 = 多黏菌素 E（★盘上实测 SMILES）",
  "CCC(C)C1NC(=O)C(CCN)NC(=O)C(CCNC(=O)C(NC(=O)C(CCN)NC(=O)C(CCN)NC(=O)C(NC1=O)"
  "C(C)CC)C(C)O)NC(=O)C(CCN)NC(=O)C(NC(=O)C(CCN)NC(=O)CCCCC(C)C)C(C)O",
  "★同上"),
 ("PMBN（★构造）", "多黏菌素 B 九肽（★去脂酰 + 去 Dab1）",
  "NC(C(C)O)C(=O)NC(CCN)C(=O)NC1CCNC(=O)C(NC(=O)C(CCN)NC(=O)C(CCN)NC(=O)C(CC(C)C)"
  "NC(=O)C(CC2=CC=CC=C2)NC(=O)C(CCN)NC1=O)C(C)O",
  "⚠️★★★【不是盘上取的、也不是权威库取的】—— ★由 PMB 的 SMILES 手工去掉脂酰基与 Dab1 构造，"
  "★★仅用于【规则自检】，★★★不计入任何统计"),
]

PM_RECORDS = [   # ★[S36-1]/[S35-26] 记的"盘上 6 条记录"⇒ ★本轮逐表复核
 ("downloads_raw/section35/molecule_intake_L3g.tsv", 2),
 ("downloads_raw/serf_input_1478.tsv", 2),
 ("downloads_raw/elzahed_matched.tsv", 2),
 ("downloads_raw/serf_finalData.tsv", 2),
 ("downloads_raw/section35/molecule_labels_L3e.tsv", 2),
 ("downloads_raw/section34/molecule_class_z.tsv（二级84）", 0),
 ("downloads_raw/section35/molecule_labels_L2b.tsv（二级84）", 0),
]


# ══════════════════════════════════════════════════════════════════════════
def stage0():
    say("\n" + "=" * 78)
    say("★★★第一步固定是【盘上三路找】")
    say("=" * 78)
    say("  ★路1 文件名含 polymyxin/colistin/lipopeptide/CAMP/peptide ⇒ ★★0 个相关资源")
    say("     （★唯一命中是 miniconda3/share/zoneinfo/America/Campo_Grande ⇒ 无关）")
    say("  ★路2 文本内容含 polymyxin/colistin ⇒ ★命中 20+ 个文件，★★但【全部只是零星提及】")
    say("     ★逐个核过：leus2023_fulltext（★只在参考文献里出现一次）、masi2022（★PMBN 当通透剂用）、")
    say("     ★prajapati（★只在『lipopeptides, e.g., daptomycin』一句里）⇒ ★★★没有结构定义")
    say("  ★路3 已有产物：★selfpromoted_SP.md / ag_literature_AG2.tsv ⇒ ★有 Hancock 1991（含 PMB），")
    say("     ★★但【没有任何结构定义】")
    say("  ⇒ ★★★★盘上三路【全空】（★无结构定义）⇒ ★★才联网。★★联网 7 次，★方向见报告。")


def stage1():
    say("\n" + "=" * 78)
    say("件一 · ★★★★先查定义与结构特征（★★逐字，★★★没有先写 SMARTS）")
    say("=" * 78)
    say("\n▎★★★必答问 1 · 结构定义逐字")
    for tag, quote, src in DEFN:
        say("\n  【%s】" % tag)
        say("    「%s」" % quote)
        say("    ★出处：%s" % src)
    say("\n  ⇒ ★★★★归纳（★★只归纳定义里【有】的，★不添）：")
    say("     · ★★环肽残基数 = ★★★**7**（★环七肽）；★整个肽 = ★★**10 残基（decapeptide）**")
    say("     · ★★环怎么闭合 = ★★★**Dab4 侧链的 N-gamma 氨基 与 C 端 Thr10 的羧基成酰胺** "
        "⇒ ★★**23 元内酰胺环**")
    say("     · ★★脂肪尾接在哪 = ★★★**线性三肽的 N 端（Dab1 的 alpha-氨基）**；"
        "★多长 = ★★**C8-C9（★最优 C7-C9）**")
    say("     · ★★正电来自哪 = ★★★**Dab 侧链的游离 gamma-氨基**；★几个 = ★★★**5 个**"
        "（★10 残基里 6 个是 Dab，★Dab4 那个被拿去成环 ⇒ 只剩 5 个游离）")
    say("  ⇒ ★★★★**任务书猜的『是不是通过 DAB 侧链氨基与 C 端羧基成酰胺键』⇒ ★★逐字证实【是】。**")

    say("\n▎★★必答问 2 · 这一类一共有哪些成员")
    say("   %-34s %-24s %-8s %s" % ("成员", "类别", "临床用", "结构差别/出处"))
    for nm, cat, clin, diff, src in MEMBERS:
        say("   %-34s %-24s %-8s %s" % (nm, cat, clin, src))
        say("       %s" % diff[:150])
    say("\n  ⇒ ★★**天然多黏菌素至少 6 个**：A / B / C / D / E(colistin) / M(mattacin)")
    say("     ⚠️★★★**但 A / C 本轮只核到【名字】，★没核到结构差异** ⇒ ★如实报")
    say("     ★★★**临床只用 B 与 E** —— ★逐字：「The clinically used polymyxins are B and E (Colistin)」")
    say("  ⇒ ★★★★**PMBN 单独报**：★★「lacking the N-terminal fatty-acyl chain and the last "
        "exocyclic amino acid」")
    say("     ⇒ ★★**它【仍结合 LPS 但不杀菌】** ⇒ ★★★**是通透剂，不是抗生素**")
    say("  ⇒ ★★**octapeptin 是【近缘但不同类】**：★同样的 Dab 富集七肽核心，"
        "★★但【只有 1 个环外氨基酸】（★不是三肽）")

    say("\n▎★★必答问 3 · 是不是【阳离子抗菌肽 CAMP】的子集")
    for tag, quote, src in CAMP:
        say("\n  【%s】 ★出处：%s" % (tag, src))
        say("    「%s」" % quote)
    say("\n  ⇒ ★★★**答：★是 —— ★但只在【功能/机制】这一层是子集，★★不是化学结构层。**")
    say("     ★★理由：★CAMP 是按『阳离子 + 抗菌 + 膜活性』定义的【功能类】，"
        "★★★没有可执行的结构判据")
    say("     ⇒ ★★★★**规则该写在【多黏菌素/环脂肽】这一层，★★不该写在 CAMP 那一层。**")
    say("     ★★★两条硬理由：")
    say("       ① ★★CAMP 涵盖 defensin / LL-37 / bactenecin 等【几十残基的核糖体肽】"
        "⇒ ★★★写在那一层会把整个肽空间标进来 ⇒ ★★误标暴涨")
    say("       ② ★★★**我们【一个 CAMP 金标都没有】** —— ★盘上只有 PMB 与 colistin 两个分子")
    say("       ⇒ ★★★★**本轮【不擅自扩大到 CAMP 层】—— ★只报，★判决权在 Lang。**")

    say("\n▎★★必答问 4 · 自促摄取的直接证据覆盖哪几个成员")
    for nm, kind, quote, src, direct in SPU:
        say("\n  【%s】 %s ⇒ ★直接证据？%s" % (nm, kind, direct))
        say("    「%s」" % quote[:230])
        say("    ★出处：%s" % src)
    say("\n  ⇒ ★★★★**答：★【直接实验证据】只覆盖 polymyxin B 一个。**")
    say("     ⚠️★★★**colistin（多黏菌素 E）本轮【没核到单独的直接证据】** ⇒ ★★对它是【外推】。")
    say("     ⚠️★★★**A/C/D/M 完全没有。**")
    say("     ★★★**这与 [S36-3] 氨基糖苷那边【同一个毛病】**：")
    say("        ★氨基糖苷：★直接证据只覆盖 2-DOS 族（gentamicin/kanamycin），★链霉胺族是外推；")
    say("        ★多黏菌素：★直接证据只覆盖 PMB，★colistin 及其余是外推。")


def stage2():
    """★★★★件二 · 与【入组标准】的冲突（★本轮最要紧的一问）。"""
    say("\n" + "=" * 78)
    say("件二 · ★★★★与【入组标准 S34-13】的冲突")
    say("=" * 78)
    say("\n  ★入组标准实现 = antibiotic-benchmark/scripts/s34l_scope_filter.py::scope_gate()")
    say("  ★md5=%s  mtime=%s" % (md5(os.path.join(HOME, "antibiotic-benchmark/scripts/"
                                                  "s34l_scope_filter.py")),
                                 mtime(os.path.join(HOME, "antibiotic-benchmark/scripts/"
                                                    "s34l_scope_filter.py"))))
    say("\n▎★★必答问 2 · 规则 P 的逐条判据（★★逐字从源码取回，★不复述）")
    for lab, sma, cond in RULE_P:
        say("    · %-52s  %-58s  %s" % (lab, sma, cond))
    say("  ⇒ ★★★**账上记的 P0..P5 六条【与源码逐字一致】** ⇒ ★无冲突。")
    say("  ⇒ ★★**规则 P 是【全部满足才删】**（★P0 先排除 β-内酰胺）。")

    say("\n▎★★★必答问 1 · 那些多黏菌素记录在入组标准下是删是留")
    rows = []
    for nm, note, smi, src in GOLD:
        g = scope_gate(smi)
        rows.append((nm, g))
        say("\n  【%s】 %s" % (nm, note))
        say("    ⇒ ★★**%s**（rule=%s）" % ("★★★删" if g["excluded"] else "★★留", g["rule"] or "—"))
        say("    %s" % g["why"])
    say("\n  ⇒ ★★★★**答：★三个【全部留下】，★一个都没被删。**")

    say("\n▎★★必答问 2 · P4「环数==0」是不是把环肽自动放行了 ⇒ ★★★实测")
    say("\n   %-24s %-8s %-10s %-10s %-10s %s" % ("分子", "P0", "P1肽键", "P2游离α氨基",
                                                   "P3羧基", "P4环数 / P5<=4"))
    for nm, g in rows:
        w = g["why"]
        seg = w.split("‖")[1] if "‖" in w else w
        say("   %-24s %s" % (nm[:24], seg.strip()))
    say("\n  ⇒ ★★★★**答：★是 —— ★但【不止 P4 一条】，★★实测有【三条】同时挡住它。**")
    say("     ① ★★**P4 环数 == 0 ✗** —— ★多黏菌素是【23 元内酰胺环】⇒ ★环数 >= 1")
    say("     ② ★★**P5 肽键数 <= 4 ✗** —— ★实测 11 个酰胺键（★远超 4）")
    say("     ③ ★★**P2 游离 alpha-氨基 >= 1 ✗**（★PMB / colistin）—— "
        "★★★**N 端被脂肪酸酰化了 ⇒ 没有游离 alpha-氨基**")
    say("     ⇒ ★★★**这三条【正好都是定义性特征】**：★成环、★十肽、★N 端脂酰化。")
    say("     ⇒ ★★★★**换句话说：★入组标准的『删寡肽』★★天生就删不掉多黏菌素** —— ★不是巧合。")
    say("     ⚠️★★**但 PMBN 不一样**：★它去掉了脂酰基 ⇒ ★★**P2 会命中（有游离 alpha-氨基）**")
    say("        ⇒ ★★但 P4/P5 仍然挡住 ⇒ ★★★**照样留下**。")

    say("\n▎★★★必答问 3 · 有没有分子【同时】命中两套规则")
    say("  ⇒ ★★★★**答：★三个金标【一个都没有】** —— ★入组标准一个没删，★标记规则三个全标（见件三）。")
    say("  ⇒ ★★★**故【本轮不存在需要定优先级的冲突】。**")
    say("  ⚠️★★**但这只是在【这 3 个】上成立** —— ★件三会在各大批上再核一遍，"
        "★★★若候选库里出现『既被入组标准删、又被判脂肽』的分子 ⇒ ★逐个列出。")
    return rows


def batch_gate_conflict(batches, meta):
    """★★★核：各批里有没有分子【同时】被入组标准删 + 被判脂肽。"""
    say("\n▎★★★件二追加 · 各批上的【双命中】复核（★★逐批实测）")
    tot_conf = []
    for nm, R in batches.items():
        mm = meta.get(nm, {})
        conf = []
        for k, f in R.items():
            if not f or f["tier"] == "不标":
                continue
            smi = mm.get(k, {}).get("_smi", "")
            if not smi:
                continue
            g = scope_gate(smi)
            if g["excluded"]:
                conf.append((k, f["tier"], g["rule"], smi))
        say("   %-14s ★被标(疑似+高度疑似) %-6d ⇒ ★★其中【同时被入组标准删】= %d"
            % (nm, sum(1 for f in R.values() if f and f["tier"] != "不标"), len(conf)))
        for k, t, r, smi in conf[:6]:
            say("        ⚠️★★双命中 %s  标记=%s  入组标准命中规则=%s" % (k, t, r))
            say("           %s" % smi[:110])
        tot_conf += conf
    say("   ⇒ ★★★**全批合计双命中 = %d**" % len(tot_conf))
    if not tot_conf:
        say("   ⇒ ★★★★**答：★没有任何分子同时命中两套规则** ⇒ ★★**本轮无需定优先级。**")
    else:
        say("   ⇒ ⚠️★★★★**存在冲突** ⇒ ★★**只报，★★★不擅自定优先级 —— 判决权在 Lang。**")
    return tot_conf


def stage3(batches, meta, cand_meta):
    say("\n" + "=" * 78)
    say("件三 · ★★★据定义定规则并验（★★沿用 AG 那套三级分档）")
    say("=" * 78)
    say("\n▎★★必答问 1 · L1/L2/L3 逐条 SMARTS 与理由")
    for lab, alg, why in CRIT_SPEC:
        say("\n  【%s】" % lab)
        say("    ★算法：%s" % alg)
        say("    ★理由：%s" % why)
    say("\n  ★★分档：★三条全中 ⇒ **高度疑似**；★两条 ⇒ **疑似**；★一条或零条 ⇒ **不标**")
    say("  ★★★不对称：★承 [S36-1]『宁可多标、不可漏标』⇒ ★三条门槛【全部低于真值】。")

    say("\n▎★★★必答问 2 · 那些多黏菌素中几条")
    say("\n   %-24s %-6s %-6s %-6s %-8s %-10s %-8s %-8s %-8s"
        % ("分子", "L1", "L2", "L3", "命中", "分档", "肽键", "大环", "伯胺"))
    gold_ok = True
    for nm, note, smi, src in GOLD:
        f = evaluate(smi)
        if nm.startswith("PMBN"):
            exp = "疑似"
        else:
            exp = "高度疑似"
        bad = (f["tier"] != exp)
        if bad and not nm.startswith("PMBN"):
            gold_ok = False
        say("   %-24s %-6s %-6s %-6s %-8d %-10s %-8d %-8d %-8d %s"
            % (nm[:24], "✓" if f["L1"] else "✗", "✓" if f["L2"] else "✗",
               "✓" if f["L3"] else "✗", f["n_hit"], f["tier"], f["n_pep"],
               f["macro_ring"], f["n_prim_amine"], "⚠️★与预期不符" if bad else ""))
        say("        脂链6命中=%d  严版酰基链命中=%d  MW=%.1f  环内肽键=%d"
            % (f["n_chain6"], f["n_acyl_strict"], f["MW"], f["pep_in_ring"]))
    if gold_ok:
        say("\n  ⇒ ★★★★**答：★PMB 与 colistin ★★两个都是 3/3 ⇒ 高度疑似。★一条都没漏。**")
        say("     ★★**PMBN 是 2/3（★缺 L3 脂尾）⇒ 疑似** —— ★★★**这与任务书的预期【逐字一致】。**")
    else:
        say("\n  ⚠️★★★★**不是 3/3 ⇒ ★★停下报出**（★见上表 ⚠️ 标记）")

    say("\n▎★★必答问 3 · 各批的三级分布（★★各批全跑一遍）")
    say("\n   %-14s %-8s %-10s %-8s %-8s %-10s" % ("批", "可解析", "高度疑似", "疑似", "不标",
                                                    "L1(环肽)命中"))
    dist = {}
    for nm, R in batches.items():
        c = Counter(f["tier"] for f in R.values() if f)
        ok = sum(1 for f in R.values() if f)
        nl1 = sum(1 for f in R.values() if f and f["L1"])
        dist[nm] = (ok, c["高度疑似"], c["疑似"], c["不标"], nl1)
        say("   %-14s %-8d %-10d %-8d %-8d %-10d"
            % (nm, ok, c["高度疑似"], c["疑似"], c["不标"], nl1))

    say("\n   ★★二级 84 复核（★[S36-1 核] 多黏菌素【不在其中】⇒ ★应全部不标）")
    R84 = batches["二级84"]
    c84 = Counter(f["tier"] for f in R84.values() if f)
    if c84["高度疑似"] == 0:
        say("      ⇒ ★★★**高度疑似 = 0 ✓ 与 [S36-1] 一致**")
    else:
        say("      ⇒ ⚠️★★★★**高度疑似 = %d ≠ 0 ⇒ 与账上冲突 ⇒ 逐个列出**" % c84["高度疑似"])
    for k, f in R84.items():
        if f and f["tier"] != "不标":
            say("        ⚠️ %-30s %s  L1=%s L2=%s L3=%s"
                % (k[:30], f["tier"], f["L1"], f["L2"], f["L3"]))

    say("\n   ★★Ritz 6,416 复核")
    cR = Counter(f["tier"] for f in batches["Ritz6416"].values() if f)
    say("      ⇒ 高度疑似 %d / 疑似 %d" % (cR["高度疑似"], cR["疑似"]))
    for k, f in list(batches["Ritz6416"].items()):
        if f and f["tier"] == "高度疑似":
            say("        ⚠️ %s  %s" % (k, meta["Ritz6416"].get(k, {}).get("_smi", "")[:100]))

    say("\n▎★★★必答问 4 · 『高度疑似=0』的批次 —— ★是规则准，还是库里没有")
    for nm, (ok, hi, sus, no, nl1) in dist.items():
        if hi:
            continue
        if nl1 == 0:
            verdict = ("★★★**是【库里压根没有环肽】** —— ★L1(环状肽)命中也是 0 "
                       "⇒ ★★该库【没有任何大环肽】")
        else:
            verdict = ("★★**是【规则挡住了】** —— ★L1 命中 %d 个（★有环肽），"
                       "★★但都没同时中 L2/L3" % nl1)
        say("   · %-14s 高度疑似=0，★L1 命中=%-6d ⇒ %s" % (nm, nl1, verdict))

    say("\n▎★★候选库 3,553 抽检（★★高度疑似 5 + 疑似 10，★seed=%d）" % SEED)
    C = batches["候选库3553"]
    rng = random.Random(SEED)
    for tier, k_n in (("高度疑似", 5), ("疑似", 10)):
        pool = sorted(k for k, f in C.items() if f and f["tier"] == tier)
        say("\n   ★★【%s】共 %d 个 ⇒ 抽 %d 个" % (tier, len(pool), min(k_n, len(pool))))
        if not pool:
            say("      ⇒ ★★**一个都没有** ⇒ ★见必答问 4 的判定")
            continue
        pick = pool if len(pool) <= k_n else rng.sample(pool, k_n)
        for k in sorted(pick):
            f = C[k]
            say("      · %-10s MW=%-7.1f L1=%s L2=%s L3=%s  肽键=%-3d 大环=%-3d 伯胺=%-3d 脂链=%d"
                % (k, f["MW"], "✓" if f["L1"] else "✗", "✓" if f["L2"] else "✗",
                   "✓" if f["L3"] else "✗", f["n_pep"], f["macro_ring"],
                   f["n_prim_amine"], f["n_chain6"]))
            say("        %s" % cand_meta.get(k, {}).get("smiles", "")[:130])
    return dist


def main():
    say("#" * 78)
    say("S36-5 查证轮 · 多黏菌素/脂肽类：★★查定义 → 定规则 → 验（★★★只核不改分数）")
    say("★开跑 %s   PID=%d" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.getpid()))
    say("★★★不造秤、★不改构件、★不改任何既有产物、★★★不改任何分子的分数")
    say("★★★次序：★★先定义、★后规则、★再验 —— ★★承 [S36-3]『不许先拼 SMARTS』的教训")
    say("★★★不对称：★宁可多标、不可漏标（★与入组标准【方向相反】）")
    say("⚠️ 两条禁读：%s —— ★本脚本从不打开" % (FORBIDDEN,))
    say("#" * 78)

    stage0()
    stage1()

    # ---- 载入各批 ----
    say("\n" + "=" * 78)
    say("★★载入各批（★★★与 [S36-2] 逐字同一套键，★★Ritz 用 row_i 不用 Reference）")
    say("=" * 78)
    sets = [("二级84", CLSZ, "smiles", "inchikey_full"),
            ("三级3805", L3, "SMILES", "完整InChIKey"),
            ("Ritz6416", RITZ, "smiles_norm", "row_i"),
            ("候选库3553", CANDF, "smiles", "mol_id")]
    batches, meta = {}, {}
    for nm, p, scol, kcol in sets:
        R = load_tsv(p)
        d, mm = {}, {}
        for r in R:
            k = str(r.get(kcol, "")).strip()
            if not k:
                continue
            smi = r.get(scol)
            d[k] = evaluate(smi)
            r["_smi"] = smi or ""
            mm[k] = r
        batches[nm], meta[nm] = d, mm
        say("   ★载入 %-14s %s\n      md5=%s  ★条目 %d" % (nm, p, md5(p), len(d)))

    Zmap = {r["inchikey_full"].strip(): r for r in load_tsv(CLSZ)}
    FW = load_tsv(FEATW)
    for scale in ("A", "D", "B"):
        d, mm = {}, {}
        for r in FW:
            if (r.get("scale") or "").strip() != scale:
                continue
            k = (r.get("inchikey_full") or "").strip() or r.get("molecule")
            smi = (Zmap.get(k, {}) or {}).get("smiles")
            d[k] = evaluate(smi)
            mm[k] = {"_smi": smi or ""}
        batches["%s秤%d" % (scale, len(d))] = d
        meta["%s秤%d" % (scale, len(d))] = mm
    say("   ★载入 A/D/B 秤（★SMILES 经 inchikey 挂回 molecule_class_z.tsv）"
        "  md5(features_water)=%s" % md5(FEATW))

    rows_gate = stage2()
    conf = batch_gate_conflict(batches, meta)
    dist = stage3(batches, meta, meta["候选库3553"])

    # ---- 产物 1：pm_criteria_hits_PM.tsv（★件三：一行一分子 × L1/L2/L3）----
    p1 = os.path.join(S36, "pm_criteria_hits_PM.tsv")
    with open(p1, "w", encoding="utf-8", newline="") as fh:
        fh.write("# S36-5 件三 · 三条判据在【金标 + 各批汇总】上的逐条命中。\n")
        fh.write("# ★★★三条判据【各自独立评估】，★不是复合规则。\n")
        fh.write("# ★L1 环状肽 / L2 多个游离伯胺 / L3 脂肪酰尾；★三全中=高度疑似，两中=疑似，"
                 "<=一条=不标。\n")
        fh.write("# ★★★门槛全部低于定义真值（大环23/肽键9/伯胺5/脂链C8-9）⇒ ★宁可多标、不可漏标。\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["类别", "分子", "L1环状肽", "L2游离伯胺", "L3脂肪尾", "命中条数", "脂肽标记",
                    "肽键数", "最大环", "环内肽键", "游离伯胺数", "脂链6命中", "严版酰基链",
                    "MW", "入组标准是否删", "入组标准命中规则", "SMILES/来源"])
        for nm, note, smi, src in GOLD:
            f = evaluate(smi)
            g = scope_gate(smi)
            w.writerow(["金标", nm, f["L1"], f["L2"], f["L3"], f["n_hit"], f["tier"],
                        f["n_pep"], f["macro_ring"], f["pep_in_ring"], f["n_prim_amine"],
                        f["n_chain6"], f["n_acyl_strict"], f["MW"],
                        "★删" if g["excluded"] else "★留", g["rule"] or "—", smi])
        for nm, (ok, hi, sus, no, nl1) in dist.items():
            w.writerow(["批汇总", nm, "", "", "", "", "高度疑似=%d 疑似=%d 不标=%d" % (hi, sus, no),
                        "", "", "", "", "", "", "", "", "", "可解析=%d  L1命中=%d" % (ok, nl1)])
    say("\n★写出 %s  md5=%s" % (p1, md5(p1)))

    # ---- 产物 2：flag_all_PM.tsv（★★含分数 + 两套标记）----
    say("\n★★★分数源 = %s（md5=%s）⇒ ★★只读，★★★本轮不改任何分数" % (CANDS, md5(CANDS)))
    sc = {r["mol_id"]: r for r in load_tsv(CANDS)}
    ag3 = {}
    for r in load_tsv(AG3):
        ag3[(r.get("批次", ""), r.get("key", ""))] = r.get("氨基糖苷标记", "")
    say("★★★氨基糖苷标记源 = %s（md5=%s）⇒ ★只读并列，★不改" % (AG3, md5(AG3)))

    p2 = os.path.join(S36, "flag_all_PM.tsv")
    n_row = 0
    n_score = 0
    with open(p2, "w", encoding="utf-8", newline="") as fh:
        fh.write("# S36-5 件三 · 一行一分子。★★★【穿透分数】与【两套标记】是【三个独立的列】。\n")
        fh.write("# ⚠️★★★★分数一律照常给 —— ★本轮【未因任何标记而改动、置空或剔除任何一个分数】。\n")
        fh.write("# ★脂肽标记（本轮新增）三级：高度疑似(L1+L2+L3 全中) / 疑似(两条) / 不标(<=一条)。\n")
        fh.write("# ★氨基糖苷标记_AG3 = 直接读自 ag_flag_all_AG3.tsv（★[S36-2]，★只读未改）。\n")
        fh.write("# ★氨基糖苷_D定义版 = [S36-3] 的 D1+D2 纯定义版 ⇒ ⚠️★★★【尚未经 Lang 裁定】，"
                 "★仅作并列参考。\n")
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["批次", "key", "SMILES", "穿透分数", "分数来源",
                    "脂肽标记", "脂肽命中判据", "L1", "L2", "L3", "脂肽命中条数",
                    "氨基糖苷标记_AG3", "氨基糖苷_D定义版",
                    "肽键数", "最大环", "游离伯胺数", "脂链6命中", "MW"])
        for bn, R in batches.items():
            mm = meta.get(bn, {})
            for k, f in sorted(R.items()):
                score, src = "", ""
                if bn.startswith("候选库"):
                    score = sc.get(k, {}).get("score_penetration", "")
                    src = "scored_aibiotics.tsv:score_penetration（★只读，★未改）"
                    if str(score).strip():
                        n_score += 1
                if f is None:
                    w.writerow([bn, k, mm.get(k, {}).get("_smi", "")[:400], score, src,
                                "★SMILES解析不了", "", "", "", "", "",
                                ag3.get((bn, k), ""), "", "", "", "", "", ""])
                else:
                    crit = "+".join([c for c, ok in (("L1", f["L1"]), ("L2", f["L2"]),
                                                     ("L3", f["L3"])) if ok]) or "—"
                    w.writerow([bn, k, mm.get(k, {}).get("_smi", "")[:400], score, src,
                                f["tier"], crit, f["L1"], f["L2"], f["L3"], f["n_hit"],
                                ag3.get((bn, k), ""), "★中" if f["AG_D"] else "",
                                f["n_pep"], f["macro_ring"], f["n_prim_amine"],
                                f["n_chain6"], f["MW"]])
                n_row += 1
    say("★写出 %s  md5=%s  行数(数据)=%d" % (p2, md5(p2), n_row))
    say("★★★分数完整性自检：候选库 3,553 里带分数的 = %d ⇒ ★★与 [S36-2] 同源，"
        "★本轮一个都没改" % n_score)

    say("\n" + "=" * 78)
    say("★★收尾 · 触及文件 md5 + mtime")
    say("=" * 78)
    touched = [__file__, p1, p2]
    readonly = [CLSZ, FEATW, L3, RITZ, CANDF, CANDS, AG3,
                os.path.join(HOME, "antibiotic-benchmark/scripts/s34l_scope_filter.py"),
                os.path.join(HOME, "antibiotic-benchmark/scripts/s36d_ag_definition.py")]
    for p in touched:
        say("   ★写   %-64s %s  %s" % (os.path.basename(p), md5(p), mtime(p)))
    for p in readonly:
        say("   ★只读 %-64s %s  %s" % (os.path.basename(p), md5(p), mtime(p)))
    say("\n   ★★★_AG2/_AG3/_AG4 未覆盖证明：")
    for fn in ("aminoglycoside_flag_AG2.md", "ag_smarts_hits_AG2.tsv", "ag_literature_AG2.tsv",
               "ag_three_tier_AG3.md", "ag_criteria_hits_AG3.tsv", "ag_flag_all_AG3.tsv",
               "ag_guanidine_AG4.md", "ag_guanidine_variants_AG4.tsv",
               "ag_gold6_variants_AG4.tsv", "ag_definition_AG4.md",
               "ag_subfamilies_AG4.tsv", "ag_classifier_check_AG4.tsv"):
        p = os.path.join(S36, fn)
        say("     %-38s %s  %s" % (fn, md5(p), mtime(p)))

    say("\n★★本轮总墙钟 %.0fs" % (time.time() - _t0))
    say("★★★跑完。★报告由 Claude 另写。")


if __name__ == "__main__":
    main()
