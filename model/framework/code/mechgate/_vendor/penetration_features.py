"""
C1 · 关卡1（穿透 / porin）构件计算器
Paper 3 抗生素评估 benchmark —— 只算分子内在构件，不做加权。

设计边界（按项目地图 A1 裁决）：
  - 本模块只产出【构件向量】，不产出标量穿透分。
  - 权重在 D1 上对 CO-ADD fold change 拟合，不在此硬编码。
  - 因此 score() 的权重是【入参】，缺省不给则报错，防止误用未拟合的权重。

构件（关卡1 底盘）：
  min_proj_area   最小投影面积 U_steric   ← Ceccarelli   (A2 方法，RDKit)
  net_charge_74   净电荷 @ pH 7.4          ← Ceccarelli   (OpenBabel pH 质子化)
  b_rotR          可旋转键分数              ← Cooper       (E. coli ΔTolC 证据)
  n_prim_amine    伯胺个数                  ← eNTRy
  mol_size        分子大小 (MW)             ← Krishnamoorthy 2017  (2026-08-06 由外排轴挪入)
  dxy             横向偶极                  ← Ceccarelli   (惰性构件，缺省不算)

C5-2 候选补构件（缺省关闭，用参数开启，不进缺省构件表）：
  a_don           H 键供体数                ← Cooper Table 2 (RDKit NumHDonors)
  logp_ow         logP(o/w)                 ← Cooper Table 2 (Crippen MolLogP)
  amphiphilic moment —— eNTRy，需 3D 构象，本模块【未实现】，见文末说明。

构件与前人原始定义/资产的对齐情况：
  min_proj_area = a2 正版，对 Acosta-Gutiérrez 2018 SI Table 2 的 9 个 β-内酰胺
                  Spearman 0.87–0.97（随构象种子波动）—— 详见函数注释；
  b_rotR = Cooper 2018 p.1229 定义 + 表3 核实；
  net_charge = OpenBabel pH 7.4；伯胺 = eNTRy SMARTS；
  mol_size = MW（Krishnamoorthy 2017 Fig 5 图注 "Sizes [log(MW)]"）。
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# 孔常数 —— 现成，不跑 porin MD。
# 【出处更正 2026-08-06】这四个数在 Acosta-Gutiérrez 2018 的【正文 Table 1】，
#   不在 SI Table 1（SI Table 1 是同源物的 RMSD / 序列一致性矩阵）。
#   数值本身逐个正确，本次只更正出处，不改数值。
# 【2026-07-22 定：走甲】穿透分不接入孔项，纯分子描述符加权和。
#   乙（q·V_p / D·E_p 耦合）拿表型标准答案校无先例、偶极项已被 A3 判死；
#   甲保留 Ceccarelli 的描述符贡献、只弃其权重形式(A1已判不可搬)；
#   且甲让 Y 与 X₃(Ceccarelli 单独) 在穿透轴上不撞脸，消融读得出增量。
#   ⇒ 以下常数仅留档备查，不接入 score()。
# ---------------------------------------------------------------------------
PORE_CONSTANTS = {  # reference-only, not wired into score()
    "OmpF": {"V_p": -1.3, "E_p": 25.2},
    "OmpC": {"V_p": -0.5, "E_p": 15.6},
}

# 伯胺 SMARTS：三价 N、两个 H、非酰胺 N、非亚胺 —— eNTRy 的伯胺规则
_PRIMARY_AMINE = Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(N=*)]")


def _prep_mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


# ---------- 构件 1：最小投影面积 U_steric（A2 口径重建） ----------
def min_projection_area(smiles: str, nconf: int = 20, seed: int = 42) -> float | None:
    """
    最小投影面积（Å²）。正版：来自 a2_min_projection_area.py。算法原样保留：
      AddHs → EmbedMultipleConfs → MMFF 优化 → 投影到【垂直于最长惯性轴】的
      平面（vt[1:]）→ 凸包面积 → 【对构象取平均】。

    与前人的对齐（2026-08-06 更正，旧注释宣称的 0.988 / 0.996 与"排序 100%
    保留"均已作废 —— 10 个分子里有 4 处换位）：
      在 Acosta-Gutiérrez 2018 SI Table 2 的 9 个 β-内酰胺上 Spearman 0.87–0.97
      （随构象种子波动），比值 ours/SI 稳定 0.49。定点值 0.933 / 0.929，
      配置 = nconf=20, seed=42（脚本默认，非挑选）。
      对照表：results/c9_citation_audit/ceccarelli/si_table2_vs_ours.csv
    绝对值系统偏小（未加 vdW 半径），比例因子进秤时被拟合权重吸收。

    ⚠️ 2026-08-12（未定 29）：本函数已查实三处缺陷，补正版见
       min_projection_area_v2() / scripts/a2_min_projection_area_v2.py。
       本函数【保留不动】，因既有产物依赖它，删了会断出处。
         ① 不加 vdW 半径 —— 算的是原子中心点集的凸包，不是截面
            （极端例：benzoate 本函数 0.00005 Å² vs ChemAxon 19 Å²）
         ② 只投影到垂直于第一主轴的那一个平面，而 MinPA 的定义是对所有方向求最小
         ③ 在输入 SMILES 原态上算，未按 pH 7.4 质子化
       Boi 42 个分子上：41/41 全部偏小，比值均值 0.392，
       Spearman(比值, MW) = +0.791（偏差随分子变大而变大，是系统性的）。
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    cids = AllChem.EmbedMultipleConfs(m, numConfs=nconf, randomSeed=seed)
    if len(cids) == 0:
        return None
    AllChem.MMFFOptimizeMoleculeConfs(m)
    areas = []
    for c in cids:
        conf = m.GetConformer(c)
        pts = np.array(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z]
             for i in range(m.GetNumAtoms())]
        )
        pts = pts - pts.mean(0)
        # PCA：投影到垂直于最长惯性轴的平面（最小投影方向的近似）
        _, _, vt = np.linalg.svd(pts)
        proj = pts @ vt[1:].T
        try:
            hull = ConvexHull(proj)
            areas.append(hull.volume)  # 2D ConvexHull.volume = 面积
        except Exception:
            pass
    return float(np.mean(areas)) if areas else None


def min_projection_area_v2(smiles: str, nconf: int = 20, seed: int = 42,
                           n_dir: int = 500, protonate: bool = True):
    """最小投影面积 v2（未定 29 补正版）。★与 v1 并存，v1 未改一行。

    三处补正：① 投影后按 vdW 半径成圆取并集面积（解析法，非点集凸包）
              ② 半球 Fibonacci 采样 n_dir 个方向取最小（非单一主轴平面）
              ③ 先经 OpenBabel 生成 pH 7.4 微种再嵌构象

    返回 dict：{'lowest', 'min', 'mean', 'n_conf', 'n_dir', 'smiles_used'}
    —— 三个构象口径并排给出，★本轮不选，由 Lang 判（未定 29 的"丁"）。
    实现与自检见 scripts/a2_min_projection_area_v2.py。
    """
    from a2_min_projection_area_v2 import min_projection_area_v2 as _v2
    return _v2(smiles, nconf=nconf, seed=seed, n_dir=n_dir, protonate=protonate)


# ---------- 构件 2：净电荷 @ pH 7.4 ----------
def net_charge_pH74(smiles: str) -> int | None:
    """
    OpenBabel 在 pH 7.4 做确定性质子化（内置 pKa 规则模型 phmodel），
    返回总形式电荷。这是 Marvin pH-微种电荷的开源确定性近似（免 license）。

    注：先前 Dimorphite-DL 取首个微种会把碱性胺（pKa≈10）漏质子化 →
    环丙沙星/甘氨酸误判为 -1。OpenBabel 对这些两性离子给出正确的净 0。
    ⚠️ 仍是规则法近似，非严格 Henderson-Hasselbalch 分数电荷；近 pKa 位点
       仍有系统性取整误差，但对占多数的强酸/强碱基团结果稳健。

    ⚠️ 2026-08-12（未定 34）：本函数已查实 OpenBabel phmodel 的两处规则缺口，
       补正版见 net_charge_pH74_v2() / scripts/_charge_rules.py。
       本函数【保留不动】，因既有产物依赖它。
         ① 磺酰胺酸性整类缺失 —— asulam(pKa~4)、磺胺甲噁唑(5.7)、★糖精(1.6)
            全部不去质子（对照：羧酸 −1 ✅、四氮唑 −1 ✅，说明不是边界情况）
         ② 脒规则只对"脒碳带碳取代"生效 —— 乙脒/胍/N-甲基乙脒 +1 ✅，
            但甲脒、N-甲基甲脒、imipenem 片段 CSCCNC=N 全部 0 ❌
    """
    from openbabel import pybel
    m = pybel.readstring("smi", smiles)
    m.OBMol.AddHydrogens(False, True, 7.4)  # (polaronly=F, correctForPH=T, pH=7.4)
    return m.OBMol.GetTotalCharge()


def net_charge_pH74_v2(smiles: str, rules=("sulfonamide", "formamidine"),
                       with_detail: bool = False):
    """净电荷 @ pH 7.4 v2（未定 34 补正版）。★与 v1 并存，v1 未改一行。

    v2 = OpenBabel 的结果 + SMARTS 后处理层，两条规则：
      sulfonamide  N 上带吸电子取代基的磺酰胺 -> 每处 −1
      formamidine  N-取代甲脒（脒碳带 H）     -> 每处 +1
    规则在 OB 的【输出分子】上匹配且全部要求相关原子中性，故不与 OB 重复计数。
    rules 可逐条关掉；rules=() 等价于 v1。

    实现、对照表与一处「与任务书冲突、未自行放宽」的说明见 scripts/_charge_rules.py。
    """
    from _charge_rules import net_charge_pH74_v2 as _v2
    return _v2(smiles, rules=rules, with_detail=with_detail)


# ---------- 构件 3：可旋转键分数 b_rotR（Cooper/MOE 口径） ----------
def b_rotR(smiles: str) -> float | None:
    """
    Cooper 2018 第1229页：b_rotR = b_rotatable / b_total（可旋转键 / 总键）。
    经对 Cooper 表3 反推核实（ampicillin/amoxicillin = 0.19）：
      - 分母 b_total = 重原子总键数（不含 C-H）→ RDKit GetNumBonds()。
      - 分子 b_rotatable = MOE 口径：非环单键、两端均为非末端重原子，
        【不排除酰胺】。RDKit 默认 CalcNumRotatableBonds 会排酰胺 → 少数一根，
        故这里自算，不用 CalcNumRotatableBonds。
    """
    mol = _prep_mol(smiles)
    if mol is None:
        return None
    denom = mol.GetNumBonds()
    if denom == 0:
        return 0.0
    n_rot = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
            if b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1:
                n_rot += 1
    return n_rot / denom


# ---------- 构件 4：伯胺个数 ----------
def n_primary_amine(smiles: str) -> int | None:
    mol = _prep_mol(smiles)
    if mol is None:
        return None
    return len(mol.GetSubstructMatches(_PRIMARY_AMINE))


# ---------- 构件 5：分子大小 mol_size ----------
def mol_size(smiles: str) -> float | None:
    """
    分子量（Da）。2026-08-06 由外排轴挪入穿透轴 —— Krishnamoorthy 2017 的 size
    规则属【外膜扩散侧】：其 Intro 直述 ">600 Da … most restricted by OM"，
    且原文称外膜屏障与外排两者 "mechanistically unrelated"。原先放外排轴是轴归错。

    度量【不变】：仍是 MW（原文 Fig 5 图注 "Sizes [log(MW)]"；ln(MW) vs 线性 MW
    实测 ρ=1.000000，故用线性 MW）。不改成体积 / 回转半径 / 截面积。
    列名 mol_size 保持不变，函数体与外排轴版本逐字相同。

    ⚠️ 无预期方向：原文 Discussion 直述 "the size of a compound is not a
       characteristic predictive of an efflux substrate"，故 expect 一律为 None，
       符号由 D1 自己拟合，不预设 +1/−1。
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Descriptors.MolWt(m)


# ---------- 构件 6：横向偶极 D_xy —— 惰性构件（缺省不算） ----------
def dxy(smiles: str, enable: bool = False):
    """
    A3 判死单构象近似（~50% 分子 cv>20%，与柔性无关，换电荷无救）。
    缺省返回 None；漏斗式：仅对过前几关的分子、需要时才置 enable=True 走多构象平均。
    此处留桩，不实现多构象偶极（阶段一无需）。
    """
    if not enable:
        return None
    raise NotImplementedError("D_xy 多构象偶极为惰性构件，需要时另接多构象实现")


# ---------------------------------------------------------------------------
# C5-2 候选补构件（缺省关闭）
# 判据锁在 docs/c5_2_penetration_criteria.md（2026-07-31 锁定，只读）。
# 本模块只负责【算】，不判 (a)(b)(c)，也不判否决条 ③。
# ---------------------------------------------------------------------------

# ---------- 候选 1：a_don —— H 键供体数（Cooper Table 2） ----------
def a_don(smiles: str, enable: bool = False) -> int | None:
    """
    H 键供体数。RDKit `Lipinski.NumHDonors`：按 SMARTS 数【带 H 的 N/O 原子个数】
    （原子计数，不是 H 计数；后者是 CalcNumLipinskiHBD）。

    ⚠️ 口径说明 —— Lang 2026-07-31 定，跑之前写在这里，不事后调：
      Cooper Table 2 的 a_don 是 MOE 描述符。MOE a_don 同为供体【原子】计数，
      与 RDKit NumHDonors 同类但 SMARTS 定义未必逐字相同（MOE 对碱性氮/
      酸性氧的归类有自己的规则）。本次直接用 RDKit NumHDonors。
      【未对 Cooper 表 3 做反推核实】—— b_rotR 当初做过那种核实（对
      ampicillin/amoxicillin = 0.19 反推分子分母口径），此处【没做】。
      ⇒ 论文中【不得写"与 Cooper 同口径"】，只能写「H 键供体原子数，
        RDKit NumHDonors 实现」。要改这个说法，须先补做表 3 核验。

    缺省 enable=False 返回 None，与 dxy 同样式：不开参数就不进构件表。
    """
    if not enable:
        return None
    mol = _prep_mol(smiles)
    if mol is None:
        return None
    return int(Lipinski.NumHDonors(mol))


# ---------- 候选 2：logP(o/w) —— Wildman-Crippen（Cooper Table 2） ----------
def logp_ow(smiles: str, enable: bool = False) -> float | None:
    """
    辛醇/水分配系数，Wildman-Crippen 原子贡献法：RDKit `Crippen.MolLogP`。
    列名用 logp_ow（判据文件写作 logP(o/w)；括号与斜杠不适合做 csv 列名/
    pandas 列名，此处只改书写形式，不改定义）。

    ⚠️ 与外排轴 slogp 同为 Crippen 实现，判据文件否决条 ③ 已预先声明
       「撞线属预期；撞了就是撞了，按原文读」。相关性由 D1 侧计算与判读，
       本函数不做任何回避性变换（不换实现、不做残差化）。

    【口径已定 —— Lang 2026-07-31：走原态版】
       本函数在【输入 SMILES 原态】上算 Crippen，不质子化到 pH 7.4。
       理由（原话记录）：pH 7.4 版会与 slogp 逐值相同、Spearman 恒为 1.0，
       使判据文件的否决条 ③ 退化为定义上必撞，失去实测内容。
       对照：efflux_features.slogp 是先 OpenBabel 质子化到 pH 7.4 再算 Crippen
       （忠于 Cooper「全部描述符在 pH 7.4 微种上算」的流程）。两者对中性分子
       给出相同的数，仅对可电离分子有差 —— 这个差正是 ③ 要实测的东西。
       此决定为判据外的实现选择，跑之前定，跑完不改。

    ⚠️ 因此【不得声称与 Cooper Table 2 的 logP(o/w) 同口径】：
       Cooper 用 MOE 且在 pH 7.4 微种上算，本实现是 RDKit Crippen + 原态。
       论文中只能写「Wildman-Crippen logP，中性态，RDKit 实现」，
       不能写「Cooper 的 logP(o/w)」。
    """
    if not enable:
        return None
    mol = _prep_mol(smiles)
    if mol is None:
        return None
    return float(Crippen.MolLogP(mol))


# ---------- 候选 3：amphiphilic moment —— 本轮不实现 ----------
# eNTRy 的 amphiphilic moment 是矢量量，需 3D 构象，与 a_don / logp_ow 的风险
# 性质不同（后两者是 2D 拓扑量，构象无关、确定性）。判据文件已记录：A3 裁定
# 单构象偶极不稳（约 50% 分子 cv>20%，换 MMFF 电荷无救），本构件大概率重演。
# 故本轮【不留桩、不实现】，避免留下一个可被误调用的半成品；单独处理。


# ---------- 候选构件注册表 ----------
# 列名 -> 计算函数。下游脚本按名取用，不各自硬编码构件名（构件名只在这里定义一次）。
# 这些函数都是【单分子、无构象、确定性】的 2D 量，可脱离 penetration_components
# 单独调用，不触发 min_proj_area 的构象嵌入 —— 补列脚本正是靠这点避免全表重算。
# amphiphilic moment 未实现，故不在表中；实现后在此登记即自动被下游看见。
C52_CANDIDATES = {
    "a_don": a_don,
    "logp_ow": logp_ow,
}


# ---------- 汇总 ----------
def penetration_components(smiles: str, with_dxy: bool = False,
                           with_a_don: bool = False,
                           with_logp_ow: bool = False) -> dict:
    """
    返回 5 个底盘构件 + smiles + dxy(None)。
    （2026-08-06：底盘由 4 件变 5 件，mol_size 由外排轴挪入，见 mol_size 注释。）
    C5-2 候选构件的键【仅在对应开关打开时才出现】，缺省连键都不加，
    以免下游按 keys() 建表时凭空多出两列空值、改变既有特征表列集。
    """
    out = {
        "smiles": smiles,
        "min_proj_area": min_projection_area(smiles),
        "net_charge_74": net_charge_pH74(smiles),
        "b_rotR": b_rotR(smiles),
        "n_prim_amine": n_primary_amine(smiles),
        "mol_size": mol_size(smiles),
        "dxy": dxy(smiles, enable=with_dxy),
    }
    if with_a_don:
        out["a_don"] = a_don(smiles, enable=True)
    if with_logp_ow:
        out["logp_ow"] = logp_ow(smiles, enable=True)
    return out


# ---------- 打分骨架：权重由 D1 提供，此处不硬编码 ----------
def score(components: dict, weights: dict) -> float:
    """
    穿透分 = Σ w_i * 构件_i（走甲：纯分子描述符加权和，不含孔项）。
    weights 必须由 D1 在 CO-ADD 上拟合后传入，不在此硬编码。
    留待 D1 决定：各构件拟合前是否先标准化 / 变换（量纲差异，属拟合层）。
    """
    missing = [k for k in weights if components.get(k) is None]
    if missing:
        raise ValueError(f"构件缺失，无法打分: {missing}")
    return sum(weights[k] * components[k] for k in weights)


# ---------------------------------------------------------------------------
# CLI：批量算构件表。列名 / 输入输出路径全走 argparse，不硬编码。
# ---------------------------------------------------------------------------
def _batch(in_path: str, smiles_col: str, out_path: str, with_dxy: bool = False,
           with_a_don: bool = False, with_logp_ow: bool = False) -> None:
    import csv
    sep = "\t" if in_path.endswith((".tsv", ".txt")) else ","
    rows_out = []
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if smiles_col not in reader.fieldnames:
            raise SystemExit(f"输入无列 '{smiles_col}'，现有列: {reader.fieldnames}")
        for row in reader:
            comp = penetration_components(row[smiles_col], with_dxy=with_dxy,
                                          with_a_don=with_a_don,
                                          with_logp_ow=with_logp_ow)
            rows_out.append(comp)
    out_sep = "\t" if out_path.endswith((".tsv", ".txt")) else ","
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()),
                                delimiter=out_sep, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"写出 {len(rows_out)} 行 → {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="C1 关卡1 穿透构件计算器")
    ap.add_argument("--in", dest="in_path", required=True, help="输入 csv/tsv（相对路径）")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES 列名")
    ap.add_argument("--out", dest="out_path", required=True, help="输出 csv/tsv（相对路径）")
    ap.add_argument("--with-dxy", action="store_true", help="启用偶极惰性构件（默认关）")
    ap.add_argument("--with-a-don", action="store_true",
                    help="C5-2 候选：H 键供体数 a_don（默认关）")
    ap.add_argument("--with-logp-ow", action="store_true",
                    help="C5-2 候选：logP(o/w) Crippen（默认关）")
    args = ap.parse_args()
    _batch(args.in_path, args.smiles_col, args.out_path, with_dxy=args.with_dxy,
           with_a_don=args.with_a_don, with_logp_ow=args.with_logp_ow)
