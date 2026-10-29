#!/usr/bin/env python
"""候选构件的暂存层 —— 水路标尺（Boi 41 条）本轮新实现的四个构件（2026-08-13）。

★命名与定位：与 _smiles_syntax.py / _charge_rules.py 同族 —— 下划线开头的“层”模块，
  ★只提供函数，不改任何既有构件模块（penetration_features.py / efflux_features.py /
  a2_min_projection_area_v2.py / _charge_rules.py 一行未动）。

【本模块实现四件（任务书编号）】
  #7  D_xy       横向偶极（★与 08-12 探针同口径，但★先按 pH 7.4 质子化再嵌构象）
  #8  D_total    总偶极（同一次构象/电荷计算顺带得到，成本近零）
  #12 globularity  Richter 2017 Nat. Methods 逐字定义：原子坐标协方差矩阵的
                   inverse condition number（= λ_min / λ_max）
  #13 rot_bonds  eNTRy 刚性规则的可旋转键【绝对数】（★不是 b_rotR 那个比值）
  #14 zwitterion 两性离子性 —— ★★文献无算法，本模块两版并给，【不选】

【★三处必须先声明的口径（原文没说清 -> 一律标 UNCLEAR，不按常理补）】
  见各函数 docstring 中以 “UNCLEAR” 起头的段落。本模块把每一处未定口径的
  ★全部候选读法都算出来并返回，由 Lang 判，模块自身不做取舍。

【本模块不做的事】
  不改任何既有实现、不重跑任何下游产物、不下“该不该进秤”的结论。
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import rdPartialCharges

__all__ = [
    "conformer_ensemble",
    "dipole_and_globularity",
    "rot_bond_counts",
    "zwitterion_flags",
    "EANG_TO_DEBYE",
    "ACID_SMARTS",
    "BASE_SMARTS",
    "self_check",
]

# 1 e·Å = 4.80320471257 D —— ★与 08-12 探针 dxy_conformer_stability.tsv 表头逐位相同。
EANG_TO_DEBYE = 4.80320471257

_PT = Chem.GetPeriodicTable()


# ===========================================================================
# 构象系综 —— #7 / #8 / #12 共用同一批构象，只嵌一次
# ===========================================================================
def conformer_ensemble(smiles_ph74: str, nconf: int = 20, seed: int = 42):
    """按 08-12 探针的逐条协议生成构象系综。

    ★协议逐字复自 downloads_raw/boi2025/dxy_conformer_stability.tsv 的表头
      （那张表是本项目 08-12 D_xy 探针的唯一留档；生成脚本未在盘上留存，
        故此处按其表头记录的协议重建，逐条对照列在下面）：
        hydrogens     : Chem.AddHs() 在嵌构象【之前】施加，显式 H 计入质量/电荷/惯量张量
        conformers    : ETKDGv3, randomSeed=42, pruneRmsThresh=-1（★不做 RMS 剪枝）
        optimisation  : MMFFOptimizeMoleculeConfs, variant='MMFF94s', maxIters=2000
                        ★未收敛的构象【保留】，不丢弃，计数另报（n_notconverged）
        partial charge: RDKit Gasteiger, nIter=12, throwOnParamFailure=False

    ⚠️★与 08-12 探针的【两处已知偏离】，跑前写在这里，跑完不改：
      甲 ★入参是 pH 7.4 质子化后的 SMILES（08-12 用的是源串原态）。
         这是任务书指定的唯一一处改动；理由是 08-12 已实证 D_xy 的瓶颈是
         质子化态而非构象采样（同一分子换 SMILES 来源差 8 倍，
         见 downloads_raw/section31/dxy_vs_acosta_comparison.tsv）。
      乙 ★RDKit 版本不同：08-12 探针跑在 conda env 'pka'（RDKit 2025.09.6，
         ★该环境【没有】OpenBabel）；本轮必须同时用 OpenBabel 质子化 + RDKit 嵌构象，
         盘上唯一同时具备两者的环境是 'gldm'（RDKit 2022.09.5 + OpenBabel 3.1.0）。
         ⇒ 版本差异【无法在本轮消除】，只能如实报出；其量级由本轮
           “原态复算”对照列（dxy_neutral_*）与 08-12 表逐分子比对给出。

    返回 dict 或 None（嵌构象失败时返回 None，★不推断填空）：
      {'mol'(含 H，带构象), 'energies', 'n_conf', 'n_notconverged', 'charges', 'masses'}
    """
    mol = Chem.MolFromSmiles(smiles_ph74)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, params=params)
    if len(cids) == 0:
        return None

    res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s",
                                            maxIters=2000, numThreads=1)
    # res: [(converged_flag, energy)]；flag==0 表示收敛。★未收敛的一律保留。
    energies = np.array([e for _c, e in res], dtype=float)
    n_notconv = int(sum(1 for c, _e in res if c != 0))

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol, nIter=12,
                                                 throwOnParamFailure=False)
        q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()],
                     dtype=float)
    except Exception:
        return None
    if not np.all(np.isfinite(q)):
        return None

    masses = np.array([a.GetMass() for a in mol.GetAtoms()], dtype=float)

    return {"mol": mol, "cids": list(cids), "energies": energies,
            "n_conf": len(cids), "n_notconverged": n_notconv,
            "charges": q, "masses": masses,
            "q_sum": float(q.sum()),
            "formal_charge": int(Chem.GetFormalCharge(mol))}


def _coords(mol, cid):
    conf = mol.GetConformer(cid)
    return np.array(conf.GetPositions(), dtype=float)


def _aggregate(vals, energies):
    """三个构象口径 —— ★与 min_proj_area v2 同（未定 29 的“丁”），本模块不选。

    lowest = MMFF 能量最低的那个构象上的值
    mean   = 系综算术平均
    min    = 系综最小值
    """
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals)
    if not ok.any():
        return {"lowest": None, "mean": None, "min": None, "std": None, "cv": None}
    e = np.asarray(energies, dtype=float)
    e_ok = np.where(ok & np.isfinite(e), e, np.inf)
    i_low = int(np.argmin(e_ok)) if np.isfinite(e_ok).any() else int(np.argmax(ok))
    v = vals[ok]
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {"lowest": float(vals[i_low]), "mean": mean, "min": float(v.min()),
            "std": std, "cv": (std / mean if mean else None)}


def dipole_and_globularity(ens: dict) -> dict:
    """在一个构象系综上一次算完 D_xy / D_total / globularity。

    ---- #7 D_xy 与 #8 D_total（★口径与 08-12 探针完全相同）----
      dipole origin : 质心 COM（质量加权，RDKit 原子质量）
                      ⚠️ 净电荷不为零时偶极依赖原点；COM 是【约定】不是事实
                      —— 08-12 表头已如此声明，本轮原样承接。
      mu            : mu = Σ_i q_i (r_i − COM)，单位 e·Å，乘 4.80320471257 转 Debye
      主惯量轴 â    : 质量加权惯量张量（对 COM），numpy.linalg.eigh，
                      ★取【最小本征值】对应的本征向量（= 分子长轴）
                      —— 任务书与 08-12 表头一致。
      D_xy          : |mu − (mu·â)â|（逐构象）
      D_total       : |mu|（逐构象，同原点同电荷）
      ★顺带留档 dxy_maxaxis（取最大本征值那一读法），与 08-12 探针的 side file 同。

    ---- #12 globularity（Richter 2017 Nature METHODS 逐字定义）----
      原文：“inverse condition number of the covariance matrix of atomic coordinates
             (a value of 1 indicates a perfect sphere while a value of 0 indicates a
             two- or one-dimensional object)”
      ⇒ 实现 = 原子坐标协方差矩阵的 λ_min / λ_max。

      ★★UNCLEAR（原文没说，以下三处一律标出，并把每一种读法都算出来）：
        UNCLEAR-1 用哪个构象？原文未说。
                  ⇒ 本模块返回 lowest / mean / min 三个口径，★与 D_xy 同，不选。
        UNCLEAR-2 加不加氢？原文只说 “atomic coordinates”，未说是否含 H。
                  ⇒ 主口径【含 H】（与 D_xy 的 AddHs 一致，同一批坐标，无额外成本）；
                    另返回 glob_noH_* 作为不含 H 的读法。
        UNCLEAR-3 加不加权？原文只说 “covariance matrix of atomic coordinates”，
                  未提任何权重。
                  ⇒ 主口径【不加权】（每个原子权重相同 —— 这是 “coordinates 的协方差”
                    的字面读法）；另返回 glob_massw_* 作为质量加权的读法。
        ★另有一处未标为 UNCLEAR 但须记：condition number 是对【矩阵】而言，
          协方差矩阵的本征值量纲是 Å²（方差），故 λ_min/λ_max 是方差之比，
          不是标准差之比。本模块按矩阵的字面定义走方差之比；
          若要标准差之比取平方根即可，本模块不另出列。

    返回扁平 dict；算不出的键为 None。
    """
    mol, cids = ens["mol"], ens["cids"]
    q, m = ens["charges"], ens["masses"]
    energies = ens["energies"]

    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])

    dxy, dxy_max, dtot = [], [], []
    g_h, g_noh, g_mass = [], [], []

    for cid in cids:
        X = _coords(mol, cid)

        # --- 偶极 ---
        com = (m[:, None] * X).sum(0) / m.sum()
        R = X - com
        mu = (q[:, None] * R).sum(0) * EANG_TO_DEBYE

        # 质量加权惯量张量（对 COM）
        r2 = (R ** 2).sum(1)
        I = np.zeros((3, 3))
        for k in range(3):
            for l in range(3):
                I[k, l] = (m * ((r2 if k == l else 0.0) - R[:, k] * R[:, l])).sum()
        w, V = np.linalg.eigh(I)
        a_min = V[:, int(np.argmin(w))]     # 最小本征值 -> 长轴（主口径）
        a_max = V[:, int(np.argmax(w))]     # 最大本征值 -> 另一读法（留档）

        dtot.append(float(np.linalg.norm(mu)))
        dxy.append(float(np.linalg.norm(mu - np.dot(mu, a_min) * a_min)))
        dxy_max.append(float(np.linalg.norm(mu - np.dot(mu, a_max) * a_max)))

        # --- globularity ---
        g_h.append(_inv_cond(X))
        g_noh.append(_inv_cond(X[heavy]) if heavy.sum() >= 3 else np.nan)
        g_mass.append(_inv_cond(X, w=m))

    out = {}
    for tag, vals in (("dxy", dxy), ("dtotal", dtot), ("dxy_maxaxis", dxy_max),
                      ("glob", g_h), ("glob_noH", g_noh), ("glob_massw", g_mass)):
        agg = _aggregate(vals, energies)
        for k, v in agg.items():
            out["%s_%s" % (tag, k)] = v
    return out


def _inv_cond(X: np.ndarray, w: "np.ndarray | None" = None) -> float:
    """坐标协方差矩阵的 inverse condition number = λ_min / λ_max。

    协方差用【总体协方差】（除以 N，不是 N−1）：inverse condition number 是比值，
    对 N vs N−1 的常数因子完全不敏感 —— 两者给出逐位相同的结果，故此处不另设开关。
    w 给出时为加权协方差（权重归一化后计算加权均值与加权二阶矩）。
    """
    if X.shape[0] < 3:
        return float("nan")
    if w is None:
        C = np.cov(X.T, bias=True)
    else:
        ww = np.asarray(w, dtype=float)
        ww = ww / ww.sum()
        mu = (ww[:, None] * X).sum(0)
        D = X - mu
        C = (ww[:, None, None] * (D[:, :, None] * D[:, None, :])).sum(0)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0.0, None)
    if ev[-1] <= 0:
        return float("nan")
    return float(ev[0] / ev[-1])


# ===========================================================================
# #13 eNTRy 刚性：可旋转键【绝对数】
# ===========================================================================
def rot_bond_counts(smiles: str) -> "dict | None":
    """可旋转键的绝对数（★不是 b_rotR 那个比值）。

    Richter 的 eNTRy 规则原文是 “可旋转键数 ≤ 5”。
    ★★UNCLEAR：原文没有指明用哪一套可旋转键定义（MOE / RDKit / ChemDraw 各不相同，
      最常见的分歧是【酰胺 C–N 键算不算】）。⇒ 本函数两套都算，两套都报，不选：

      rot_n_moe    ★与本仓库 penetration_features.b_rotR 的【分子】逐字同口径：
                   非环单键、两端均为非末端重原子，★不排除酰胺。
                   （代码为对 b_rotR 分子部分的【复制】—— b_rotR 只返回比值，
                     取不到分子；★既有实现未改一行。）
      rot_n_rdkit  RDKit 默认 rdMolDescriptors.CalcNumRotatableBonds（Strict 口径，
                   ★排除酰胺 C–N）。

    每套同时给出连续值与按 ≤5 二值化的结果（任务书要求两种都报）：
      rot_n_moe / rot_rigid_moe(1=可旋转键 ≤5) / rot_n_rdkit / rot_rigid_rdkit
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_moe = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
            if b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1:
                n_moe += 1
    n_rd = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    return {"rot_n_moe": n_moe, "rot_rigid_moe": int(n_moe <= 5),
            "rot_n_rdkit": n_rd, "rot_rigid_rdkit": int(n_rd <= 5)}


# ===========================================================================
# #14 两性离子性 —— ★★文献无算法，本模块的判定细则为【自定】
# ===========================================================================
# ⚠️★★必须与结果一同报出：08-11 已查实【文献无算法】。最接近的定义只有
#   Prajapati/Kleinekathöfer/Winterhalter 的一句图注 “zwitterionic having zero
#   net charge”。以下两版判定细则【全部是本项目自定的】，不是任何文献的口径，
#   ★不得写成“按 Prajapati 的定义算”。两版并给，本模块不选。
#
# ★酸/碱基团 SMARTS 的来源：从 downloads_raw/section31/s31c_pka_rules.py（08-13
#   已留档、已自检的可解离基团表）中挑出【在 pH 7.4 附近或更极端一侧会带电】的类别，
#   由本模块按酸/碱重新分组并逐条抄录。★分组是本模块做的，s31c 那张表本身
#   只标 pKa 窗口、不标酸碱 —— 这一层归类同属自定。
#   每条同时收【中性写法】与【已电离写法】，因为源 SMILES 里两种都有。

ACID_SMARTS = [
    ("羧酸", "[CX3](=[OX1])[OX2;H1]"),
    ("羧酸(已解离)", "[CX3](=[OX1])[OX1;-]"),
    ("磷酸/磷酸酯", "[PX4](=[OX1])([OX2H1,OX1-])[OX2][#6,#15]"),
    ("膦酸", "[PX4](=[OX1])([OX2H1,OX1-])([OX2H1,OX1-])[#6]"),
    ("硫酸酯/磺酸", "[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]"),
    ("四氮唑", "[nX3;H1;r5;$(n1nnnc1)]"),
    ("四氮唑(已解离)", "[nX2;-;r5;$(n1nnnc1)]"),
    ("磺酰胺_N-杂芳基", "[NX3;H1]([SX4](=[OX1])(=[OX1]))[c;$(c~[n,o,s;a])]"),
    ("磺酰胺_N-杂芳基(已解离)", "[NX2;H0;-]([SX4](=[OX1])(=[OX1]))[c;$(c~[n,o,s;a])]"),
    ("磺酰胺_N-酰基", "[NX3;H1]([CX3]=[OX1,SX1])[SX4](=[OX1])(=[OX1])"),
    ("磺酰胺_N-酰基(已解离)", "[NX2;H0;-]([CX3]=[OX1,SX1])[SX4](=[OX1])(=[OX1])"),
]

BASE_SMARTS = [
    ("脂肪胺", "[NX3;H2,H1,H0;!$(N[!#6]);!$(N=*);!$(NC=[O,N,S]);!$(n);!$(N[a])]"),
    ("脂肪胺(已质子化)", "[NX4;+;H1,H2,H3;$(N[CX4])]"),
    ("脒/胍", "[NX3][CX3]=[NX2]"),
    ("胍", "[NX3][CX3](=[NX2])[NX3]"),
    ("脒/胍(已质子化)", "[NX3][CX3]=[NX3;+]"),
    ("咪唑型芳香氮", "[nX2;r5;$([nX2]1:[c]:[nX3]:[c]:[c]1)]"),
    ("季铵/杂环鎓", "[N;+;!$([N;+]~[O;-])]"),
]

_ACID_P = [(k, Chem.MolFromSmarts(s)) for k, s in ACID_SMARTS]
_BASE_P = [(k, Chem.MolFromSmarts(s)) for k, s in BASE_SMARTS]


def zwitterion_flags(smiles: str) -> "dict | None":
    """两性离子性，★两版并给，本模块【不选】。

    (a) 净电荷版 zwit_charge
        判定 = net_charge_v2(smiles) == 0 【且】分子内同时存在正电位点与负电位点。
        ★“位点”的数法（自定）：在 _charge_rules 里 OpenBabel pH 7.4 的【输出串】上
          数形式电荷不为 0 的原子；再把 v2 的两条 SMARTS 后处理规则的命中数并入
          —— sulfonamide 每处记 1 个负位点，formamidine 每处记 1 个正位点。
          理由：v2 的修正正是“OB 没做出来的那个电荷”，若只看 OB 输出会漏掉。
        ⇒ zwit_charge = 1 当且仅当 q_v2 == 0 且 n_pos ≥ 1 且 n_neg ≥ 1。

    (b) 基团版 zwit_group
        判定 = SMARTS 同时命中 ≥1 个可解离酸性基团 与 ≥1 个可质子化碱性基团。
        ★不看净电荷、不看质子化态；在【输入串】（已过 normalize_smiles_syntax
          的原态串）上匹配。

    ⚠️ 两版会不一致，这是预期内的：(a) 要求净电荷恰为 0，(b) 不要求。
       两列都写进产物，★由 Lang 判，本模块不合并、不取交并集。

    返回 dict，含两个 0/1 标记与全部中间量（便于逐分子核对）。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    out = {}

    # ---- (b) 基团版 ----
    acid_hits = [k for k, p in _ACID_P if p is not None and mol.HasSubstructMatch(p)]
    base_hits = [k for k, p in _BASE_P if p is not None and mol.HasSubstructMatch(p)]
    out["zwit_acid_groups"] = ";".join(acid_hits)
    out["zwit_base_groups"] = ";".join(base_hits)
    out["zwit_group"] = int(bool(acid_hits) and bool(base_hits))

    # ---- (a) 净电荷版 ----
    from _charge_rules import net_charge_pH74_v2
    det = net_charge_pH74_v2(smiles, with_detail=True)
    if det is None:
        out["zwit_charge"] = None
        out["zwit_n_pos"] = None
        out["zwit_n_neg"] = None
        out["zwit_q_v2"] = None
        return out
    q_v2, fired, _q_ob, ob_smi = det
    obm = Chem.MolFromSmiles(ob_smi) if ob_smi else None
    n_pos = n_neg = 0
    if obm is not None:
        for a in obm.GetAtoms():
            fc = a.GetFormalCharge()
            if fc > 0:
                n_pos += 1
            elif fc < 0:
                n_neg += 1
    # v2 后处理规则命中的位点并入（fired 形如 ['R1'] 或 ['R1×2']）
    for lab in fired:
        n = int(lab.split("×")[1]) if "×" in lab else 1
        if lab.split("×")[0].startswith("R1"):      # sulfonamide -> 负位点
            n_neg += n
        elif lab.split("×")[0].startswith("R2"):    # formamidine -> 正位点
            n_pos += n

    out["zwit_q_v2"] = q_v2
    out["zwit_n_pos"] = n_pos
    out["zwit_n_neg"] = n_neg
    out["zwit_charge"] = int(q_v2 == 0 and n_pos >= 1 and n_neg >= 1)
    return out


# ===========================================================================
# 自检
# ===========================================================================
def self_check(verbose: bool = True) -> bool:
    """三组自检。★任何一条不符即口径漂移，须先查清再跑正式表。"""
    ok = True

    # --- 1. globularity 的两个解析极限：球 vs 平面/直线 ---
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20000, 3))
    sphere = v / np.linalg.norm(v, axis=1, keepdims=True)
    g_sphere = _inv_cond(sphere)
    flat = np.column_stack([rng.normal(size=200), rng.normal(size=200),
                            np.zeros(200)])
    g_flat = _inv_cond(flat)
    line = np.column_stack([np.linspace(-5, 5, 200), np.zeros(200), np.zeros(200)])
    g_line = _inv_cond(line)
    c1 = abs(g_sphere - 1.0) < 0.05 and g_flat < 1e-12 and g_line < 1e-12
    ok &= c1
    if verbose:
        print("  %s globularity 极限：球壳 %.4f (应≈1) / 平面 %.2e (应=0) / 直线 %.2e (应=0)"
              % ("✅" if c1 else "❌", g_sphere, g_flat, g_line))

    # --- 2. 可旋转键：酰胺口径分歧必须真的出现 ---
    cases = [
        # 乙酰苯胺：非环单键中 C(=O)–N 与 N–c 两根两端均非末端（甲基碳是末端，不计）。
        # ★MOE 口径两根都算 = 2；RDKit Strict 口径排掉酰胺 C–N，只剩 N–c = 1。
        # 这一条正是用来把“酰胺算不算”的口径分歧钉在自检里的。
        ("乙酰苯胺 CC(=O)Nc1ccccc1", "CC(=O)Nc1ccccc1", 2, 1),
        ("正己烷 CCCCCC", "CCCCCC", 3, 3),
        ("苯 c1ccccc1", "c1ccccc1", 0, 0),
    ]
    for lab, smi, want_moe, want_rd in cases:
        r = rot_bond_counts(smi)
        good = r["rot_n_moe"] == want_moe and r["rot_n_rdkit"] == want_rd
        ok &= good
        if verbose:
            print("  %s 可旋转键 %-28s MOE=%d(期望%d) RDKit=%d(期望%d)"
                  % ("✅" if good else "❌", lab, r["rot_n_moe"], want_moe,
                     r["rot_n_rdkit"], want_rd))

    # --- 3. 两性离子：三个手算得出的分子 ---
    zcases = [
        # (标签, SMILES, 期望 zwit_charge, 期望 zwit_group)
        ("甘氨酸（教科书两性离子）", "NCC(=O)O", 1, 1),
        ("苯甲酸（只有酸）", "OC(=O)c1ccccc1", 0, 0),
        ("正丁胺（只有碱）", "CCCCN", 0, 0),
        ("谷氨酸（两酸一碱，净 −1）", "N[C@@H](CCC(=O)O)C(=O)O", 0, 1),
    ]
    for lab, smi, want_c, want_g in zcases:
        r = zwitterion_flags(smi)
        good = r["zwit_charge"] == want_c and r["zwit_group"] == want_g
        ok &= good
        if verbose:
            print("  %s 两性离子 %-26s 净电荷版=%s(期望%s) 基团版=%s(期望%s) q_v2=%s"
                  % ("✅" if good else "❌", lab, r["zwit_charge"], want_c,
                     r["zwit_group"], want_g, r["zwit_q_v2"]))

    if verbose:
        print("\n  自检总判：%s" % ("全过" if ok else "★有不符，先查清再跑正式表"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
