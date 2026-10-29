#!/usr/bin/env python
"""S36-3 查证轮 —— 先查【氨基糖苷的化学定义】，再定规则（★★只核不改）。

★★★★【Lang 2026-08-21】▎「要不要直接查一下氨基糖苷的化学性质算了，★与其搞这么多针对分子的筛选」
  ⇒ ★★★**助手认同并自认一处方法错**：★我们一直在【拼 SMARTS】，★★**却没先搞清这一类的化学定义**。
  ⇒ ★★[S36-2] 那条"放宽 A 收纳胍基"的裁定 ⇒ ★★★**本轮【暂停执行】**，★等定义查清再定。

★★★★【本轮查到的权威定义（★件一逐字）】
  ▎"Aminoglycosides are a large class of antibiotics that are characterised by
    **two or more amino sugars linked by glycosidic bonds to an aminocyclitol component**."
  ▎"The aminocyclitol is either **streptidine, as in streptomycin**, or **2-deoxystreptamine,
    which is found in all other available aminoglycosides**."
  ⇒ ★★★★**定义只有【两条】：① 氨基环醇核心 ② ≥2 个糖经糖苷键连上去。**
  ⇒ ★★★**"净正电 ≥+3"与"伯胺 ≥3"（★我们的判据 C）★★根本不在定义里** —— ★是我们自己加的。

⚠️ 两条禁读：cooper_features.csv / cooper_casecontrol_*.csv —— ★本脚本从不打开。
★产物 _AG4 后缀（★新文件名，★★_AG2/_AG3 与已有 _AG4 一个都不覆盖）。
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

DL = os.path.join(HOME, "downloads_raw")
S34, S35, S36 = (os.path.join(DL, f"section{n}") for n in (34, 35, 36))
CLSZ = os.path.join(S34, "molecule_class_z.tsv")
L3 = os.path.join(S35, "level3_labels_L3m.tsv")
RITZ = os.path.join(S35, "ritz_ourfeatures_C.tsv")
CANDF = os.path.join(HOME, "antibiotic-benchmark/candidates/features_aibiotics.tsv")
CANDS = os.path.join(HOME, "antibiotic-benchmark/candidates/scored_aibiotics.tsv")

FORBIDDEN = ("cooper_features.csv", "cooper_casecontrol_")
SEED = 20260822
_t0 = time.time()

GLY = Chem.MolFromSmarts("[CX4;R][OX2;!R][#6;R]")

# ═════════════ 件一 · 亚族表（★逐字来自文献，★出处见报告）═════════════
SUBFAMILIES = [
 dict(编号="F1", 亚族="链霉胺族（streptidine）", 母核="streptidine（链霉胺）= 1,3-二胍基-2,4,5,6-四羟基环己烷",
      环上取代基="★★**胍基**（★不是伯胺）", 代表药="streptomycin（链霉素）",
      我们盘上有几个="1", 拓扑="★★**线性**：氨基环醇 → 链霉糖 → N-甲基葡糖胺",
      逐字依据="「The aminocyclitol is either **streptidine, as in streptomycin**, or 2-deoxystreptamine」",
      出处="ScienceDirect Topics · Aminoglycoside（★教科书汇编）"),
 dict(编号="F2", 亚族="2-脱氧链霉胺族 · 4,5-二取代", 母核="2-deoxystreptamine（2-脱氧链霉胺）",
      环上取代基="伯胺", 代表药="neomycin（新霉素＝fradiomycin）／paromomycin／lividomycin",
      我们盘上有几个="3（fradiomycin / lividomycin / mannosylparomomycin）",
      拓扑="★★**分支**：糖—氨基环醇—糖（★4,5 位）",
      逐字依据="「the **4,5-disubstituted 2-deoxystreptamines**」；「4,5-disubstituted deoxystreptamine, e.g., **neomycin**」",
      出处="同上"),
 dict(编号="F3", 亚族="2-脱氧链霉胺族 · 4,6-二取代", 母核="2-deoxystreptamine（2-脱氧链霉胺）",
      环上取代基="伯胺", 代表药="gentamicin／kanamycin／tobramycin／amikacin",
      我们盘上有几个="2（gentamicin / kanamycin）",
      拓扑="★★**分支**：糖—氨基环醇—糖（★4,6 位）",
      逐字依据="「the **4,6-disubstituted 2-deoxystreptamines** containing **most of the clinically useful** aminoglycoside antibiotics」",
      出处="同上"),
 dict(编号="F4", 亚族="★其他（others）", 母核="★文献只写 others，★未逐一列",
      环上取代基="★未取到", 代表药="★未取到",
      我们盘上有几个="0", 拓扑="★未取到",
      逐字依据="「These structural types include the 4,6-disubstituted…, the 4,5-disubstituted…, and **others**.」",
      出处="同上（⚠️★★『others』这一档★本轮未核到具体内容）"),
 dict(编号="X1", 亚族="⚠️★★**氨基环醇类（aminocyclitol）—— ★★★不是氨基糖苷**",
      母核="氨基环醇（★无糖苷键连的氨基糖）", 环上取代基="—",
      代表药="★★**spectinomycin（壮观霉素）**", 我们盘上有几个="0", 拓扑="—",
      逐字依据="「Spectinomycin is an **aminocyclitol antibiotic**… it differs in that it **does not contain amino sugars or glycosidic bonds**」",
      出处="ScienceDirect Topics · Spectinomycin"),
]

# ═════════════ 件二 · 现成分类器核查表 ═════════════
CLASSIFIERS = [
 dict(编号="C1", 分类器="ChEBI `aminoglycoside` (CHEBI:47779)",
      有没有="★有", 可不可执行="★★★**不可** —— ★成员是【人工策展】，★无结构判据/SMARTS",
      成员数="★★**300+**（★本轮取回的页面所见）",
      在我们6个上准不准="★★★**未逐分子实测**（★无离线数据、★本轮不做批量下载）",
      实测到的错="★★★★**子条目里含 doxorubicin（蒽环类）与 erythromycin（大环内酯）** ⇒ ★★**两者都不是氨基糖苷**",
      成因="★该节点的本体关系是「is a **carbohydrate derivative**」「is a **glycoside**」 ⇒ ★★★**它是【含氨基糖的糖苷】这个宽泛化学类，★不是抗生素类**",
      可下载="★★可（OWL / OBO / JSON / SDF / SQL，★https://www.ebi.ac.uk/chebi/downloadsForward.do）"),
 dict(编号="C2", 分类器="ChEBI `aminoglycoside antibiotic` (CHEBI:22507)",
      有没有="★有", 可不可执行="★★★**不可** —— ★★**role-based/curated**（★按功能与人工策展），★无结构判据",
      成员数="★★**约 30 个**",
      在我们6个上准不准="★★**streptomycins 与 kanamycins 在成员表里** ⇒ ★这两个对得上；★其余未逐一核",
      实测到的错="★★★**成员表里含【蒽环类】daunorubicin / carminomycin** ⇒ ★★**不是氨基糖苷**",
      成因="★★该节点同时挂在「antimicrobial agent」这个【药理角色】下 ⇒ ★★★**化学与药理混用**",
      可下载="★同 C1"),
 dict(编号="C3", 分类器="ClassyFire / ChemOnt",
      有没有="★有（★4,825 个化学类）", 可不可执行="★★**需联网 API**（★web 或 Ruby API）；★★★**盘上没装**（★[S34-11] 已核）",
      成员数="—（★是分类器不是成员表）",
      在我们6个上准不准="★★★**本轮未实测**（★需逐分子联网调用）",
      实测到的错="—", 成因="★基于规则的层级分类（ChemOnt）",
      可下载="★★API 可用；★离线可得性本轮未核"),
 dict(编号="C4", 分类器="NPClassifier",
      有没有="★有", 可不可执行="⚠️★★★**它是【深度神经网络】** ⇒ ★★★**与本项目【不用机器学习】的约束冲突**",
      成员数="—", 在我们6个上准不准="★★**本轮未实测**",
      实测到的错="—", 成因="★天然产物专用的 DNN 分类器",
      可下载="★★本轮未核"),
 dict(编号="C5", 分类器="ATC `J01G`（氨基糖苷类抗菌药）",
      有没有="★有", 可不可执行="★★★**不可** —— ★★**这是【药理分类】不是化学分类**",
      成员数="★本轮未取到确切数",
      在我们6个上准不准="★★**不适用** —— ★ATC 只覆盖【上市药物】，★★我们候选库的 3,553 个生成分子**一个都没有 ATC 码**",
      实测到的错="⚠️★★**ATC 把 spectinomycin 归在 J01XX**（★其他抗菌药），★★**而不少教科书把它与氨基糖苷并列讲** ⇒ ★口径不一致",
      成因="★按治疗用途分类", 可下载="★可（WHO ATC）"),
]


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


def load_tsv(p, cmark="#"):
    with open(guard(p), newline="", encoding="utf-8", errors="replace") as fh:
        lines = [l for l in fh if not l.startswith(cmark)]
    return list(csv.DictReader(io.StringIO("".join(lines)), delimiter="\t"))


# ═════════════ 件三 · 据定义重写的判据 ═════════════
def aminocyclitol_rings(m):
    """★D1【氨基环醇核心】：★六元全碳非芳环；★环上带氧取代基 ≥2；★带氮取代基 ≥1。
    ★★★氮【不限类型】—— ★★链霉胺族是【胍基】、2-脱氧链霉胺族是【伯胺】，★两族都要覆盖。"""
    out = []
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
            out.append(rs)
    return out


def D2_any(m):
    """★D2【≥2 糖经糖苷键相连】· 宽版：★≥2 个含环内氧的环，★且有糖苷键跨【不同环】。"""
    rings = m.GetRingInfo().AtomRings()
    if sum(1 for r in rings if any(m.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in r)) < 2:
        return False
    for c1, o, c2 in m.GetSubstructMatches(GLY):
        r1 = {k for k, r in enumerate(rings) if c1 in r}
        r2 = {k for k, r in enumerate(rings) if c2 in r}
        if r1 and r2 and r1 != r2:
            return True
    return False


def D2_strict(m):
    """★D2′ 严版：★★氨基环醇【本身】带 ≥2 个糖苷键连到糖环。
    ⚠️★★★实测【漏判 streptomycin】—— ★它是【线性】拓扑（氨基环醇只连 1 个糖）。"""
    rings = m.GetRingInfo().AtomRings()
    sugars = [set(r) for r in rings if any(m.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in r)]
    for ac in aminocyclitol_rings(m):
        n = 0
        for c1, o, c2 in m.GetSubstructMatches(GLY):
            for a, b in ((c1, c2), (c2, c1)):
                if a in ac and any(b in s for s in sugars):
                    n += 1
                    break
        if n >= 2:
            return True
    return False


RULES = {
    "★★D 定义版(D1+D2宽)": lambda m: bool(aminocyclitol_rings(m)) and D2_any(m),
    "D′ 定义严版(D1+D2严)": lambda m: bool(aminocyclitol_rings(m)) and D2_strict(m),
}


def main():
    say("#" * 78)
    say("S36-3 查证轮 · 先查【氨基糖苷的化学定义】，再定规则（★★只核不改）")
    say(f"★开跑 {datetime.now():%Y-%m-%d %H:%M:%S}   PID={os.getpid()}")
    say("★★★[S36-2] 那条『放宽 A 收纳胍基』的裁定 ⇒ ★★本轮【暂停执行】")
    say("★★★标记绝不影响分数；★不造秤、★不改任何既有产物；★多黏菌素仍不标")
    say(f"⚠️ 两条禁读：{FORBIDDEN} —— ★本脚本从不打开")
    say("#" * 78)

    # ---- 件一 ----
    say("\n" + "=" * 78)
    say("件一 · 氨基糖苷的正式定义（★★逐字，★本轮地基）")
    say("=" * 78)
    say("\n★★★★**权威定义逐字**：")
    say('   「Aminoglycosides are a large class of antibiotics that are characterised by')
    say('     **two or more amino sugars linked by glycosidic bonds to an aminocyclitol component**.」')
    say('   「the descriptor aminoglycoside results from the glycosidic bonds between the aminocyclitol')
    say('     and **two or more amino-containing or non–amino-containing sugars**」')
    say('   「The aminocyclitol is either **streptidine, as in streptomycin**, or **2-deoxystreptamine,')
    say('     which is found in all other available aminoglycosides**.」')
    say("\n   ⇒ ★★★★**定义只有【两条】：★① 氨基环醇核心 ★② ≥2 个糖经糖苷键连上去。**")
    say("   ⇒ ★★★**我们判据 C 的「净正电≥+3」「伯胺≥3」★★根本不在定义里** —— ★是我们自己加的。")
    say("\n★★亚族（★逐个见 ag_subfamilies_AG4.tsv）：")
    for r in SUBFAMILIES:
        say(f"   {r['编号']:4s} {r['亚族'][:34]:36s} 母核={r['母核'][:30]:32s} 我们有 {r['我们盘上有几个']}")

    # ---- 件二 ----
    say("\n" + "=" * 78)
    say("件二 · 现成的权威分类能不能直接用")
    say("=" * 78)
    for r in CLASSIFIERS:
        say(f"\n   ★{r['编号']} · {r['分类器']}")
        say(f"      可不可执行 : {r['可不可执行']}")
        say(f"      成员数     : {r['成员数']}")
        if r["实测到的错"] != "—":
            say("      ⚠️实测到的错: " + r["实测到的错"])
            say(f"      成因       : {r['成因']}")

    # ---- 件三 ----
    say("\n" + "=" * 78)
    say("件三 · 据定义重定规则（★★只给方案，★★★不实施）")
    say("=" * 78)
    Z = load_tsv(CLSZ)
    gold = [r for r in Z if "氨基糖苷" in (r.get("自判大类") or "")]
    gold.sort(key=lambda r: r["molecule"])
    say(f"\n▎★★★必答问 2 · 新方案在那 6 个上漏判几个")
    for rn, fn in RULES.items():
        say(f"\n   ── {rn}")
        miss = []
        for r in gold:
            m = Chem.MolFromSmiles(r["smiles"])
            ok = fn(m)
            say(f"      {r['molecule'][:24]:26s} {'★★中' if ok else '✗★★漏'}")
            if not ok:
                miss.append(r["molecule"])
        say(f"      ⇒ ★★**漏判 {len(miss)}/6** "
            f"{'✓' if not miss else '❌★★★停下报出：' + '/'.join(miss)}")

    say(f"\n▎★★必答问 3 · 各批分布")
    sets = [("二级84", CLSZ, "smiles"), ("三级3805", L3, "SMILES"),
            ("Ritz6416", RITZ, "smiles_norm"), ("候选库3553", CANDF, "smiles")]
    hits = {rn: {} for rn in RULES}
    metas = {}
    say(f"\n   {'批':14s} {'可解析':>7s} " + " ".join(f"{rn[:16]:>18s}" for rn in RULES))
    for nm, p, scol in sets:
        R = load_tsv(p)
        metas[nm] = R
        tot = 0
        cnt = {rn: 0 for rn in RULES}
        for r in R:
            s = (r.get(scol) or "").strip()
            if not s:
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            tot += 1
            for rn, fn in RULES.items():
                if fn(m):
                    cnt[rn] += 1
                    hits[rn].setdefault(nm, []).append((r, scol))
        say(f"   {nm:14s} {tot:>7d} " + " ".join(f"{cnt[rn]:>18d}" for rn in RULES))

    say(f"\n   ⚠️★★**Ritz 复核**：★[S35-17] 记该库氨基糖苷 = 0")
    for rn in RULES:
        n = len(hits[rn].get("Ritz6416", []))
        say(f"      {rn[:22]:24s} 被标 {n} {'✓' if n == 0 else '❌逐个报'}")

    say(f"\n   ★★**与 [S36-1] 单级规则的对照**（★★账上记 141，★★★[S36-1] 报告与产物实测都是 **42**）")
    for rn in RULES:
        n = len(hits[rn].get("候选库3553", []))
        say(f"      {rn[:22]:24s} 候选库命中 **{n}** "
            f"⇒ ★相对 42 {'收紧' if n < 42 else '放宽'} {abs(42-n)} 个")

    # 抽检
    rng = random.Random(SEED)
    rn = "★★D 定义版(D1+D2宽)"
    cand = hits[rn].get("候选库3553", [])
    say(f"\n▎★★候选库抽检（★{rn}，★共 {len(cand)} 个，★seed={SEED}）⇒ ★★★请 Lang 肉眼核")
    for r, scol in rng.sample(cand, min(10, len(cand))):
        m = Chem.MolFromSmiles(r[scol])
        say(f"      {str(r.get('mol_id'))[:12]:12s} MW={Descriptors.MolWt(m):>6.0f}")
        say(f"         {str(r[scol])[:92]}")

    # ---- 落盘 ----
    p1 = os.path.join(S36, "ag_subfamilies_AG4.tsv")
    with open(p1, "w", newline="", encoding="utf-8") as fh:
        fh.write("# S36-3 件一 · 一行一亚族。★★逐字依据与出处逐行标明。\n")
        fh.write("# ⚠️★X1 是【氨基环醇类】，★★★不是氨基糖苷（★spectinomycin 的归属）。\n")
        fh.write(f"# 行数：{len(SUBFAMILIES)}\n")
        w = csv.DictWriter(fh, fieldnames=list(SUBFAMILIES[0].keys()), delimiter="\t")
        w.writeheader()
        for r in SUBFAMILIES:
            w.writerow(r)
    say(f"\nwrote {p1} ({len(SUBFAMILIES)} 行)")

    p2 = os.path.join(S36, "ag_classifier_check_AG4.tsv")
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        fh.write("# S36-3 件二 · 一行一分类器。★★『可不可执行』是本轮的关键列。\n")
        fh.write("# ⚠️★★★现成分类器【必须实测，不许因权威就信】（★fosfomycin 的先例，[S34-11]）。\n")
        fh.write(f"# 行数：{len(CLASSIFIERS)}\n")
        w = csv.DictWriter(fh, fieldnames=list(CLASSIFIERS[0].keys()), delimiter="\t")
        w.writeheader()
        for r in CLASSIFIERS:
            w.writerow(r)
    say(f"wrote {p2} ({len(CLASSIFIERS)} 行)")
    say(f"\n★★本轮总墙钟 {time.time()-_t0:.0f}s")
    say("★★★三件跑完。★报告由 Claude 另写 ag_definition_AG4.md。")


if __name__ == "__main__":
    main()
