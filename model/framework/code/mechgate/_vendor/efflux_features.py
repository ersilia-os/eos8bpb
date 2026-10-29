"""
C2 · 关卡2（外排 / AcrAB-TolC）构件计算器
Paper 3 抗生素评估 benchmark —— 只算分子内在构件，不做加权（走甲，与 C1 一致）。

⚠️ 地图既定事实：关卡2 分量比关卡1 少一半，且【没有物理模型】
   （无 Ceccarelli 那样的对应物）。这是范围声明，不是缺陷；写作时要承认。

构件（关卡2 底盘）：
  slogp          亲脂性 SlogP @ pH7.4     ← Cooper 2018   (Crippen=Wildman-Crippen, ref38)

⚠️ 2026-08-06 起，外排轴【只剩 slogp 一个构件】。两处变动：
   - mol_size 移入 penetration_features.py（Krishnamoorthy 2017 的 size 规则属
     外膜扩散侧，原文 Intro 直述 ">600 Da … most restricted by OM"，并称两道屏障
     "mechanistically unrelated"）—— 度量不变，仍是 MW。
   - gurvic_group 已删（见下方说明）。
   「外排轴这个说法在论文里怎么写」是未定项，此处不预设说法。

产出边界（与 C1 相同）：只输出【构件向量】，权重在 D1 上对 CO-ADD tolC fold
change 拟合，不在此硬编码。方向（高亲脂 → 更易被泵出）由拟合权重自然吸收。
"""

from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen

# ---------------------------------------------------------------------------
# gurvic_group 及其 SMARTS 词表已于 2026-08-06 删除（历史见 git）。
# 删除理由是【适用性】，不是保真度：Gurvic 的子结构富集来自 MMPA，
# 只在"同一骨架上增减基团"的框架内成立，不能搬到任意分子间的横向比较。
# 实现本身抄得准（9 条 SMARTS 逐字命中 9/9、方向 9/9）—— 不得写成"实现有误"。
# ---------------------------------------------------------------------------


# ---------- 构件 1：亲脂性 SlogP @ pH 7.4 ----------
def slogp(smiles: str) -> float | None:
    """
    slogp = Wildman-Crippen（Cooper 2018 ref38 = RDKit Crippen.MolLogP）。
    两版并行：中性态与 pH 7.4 质子化态。保真度选中性（vs Cooper 62 分子
    ρ=0.972 优于 pH7.4 的 0.907），预测力选 pH 7.4（C8 逐折 5/5 优于中性态，
    +0.0162）。两版并报，见敏感性分析。
    不得宣称"复现了 Cooper"，不得直搬 −0.7 阈值。

    本函数为 pH 7.4 版（当前默认，不改）：先 OpenBabel 质子化到 pH 7.4 再算 Crippen。
    """
    from openbabel import pybel
    try:
        m = pybel.readstring("smi", smiles)
        m.OBMol.AddHydrogens(False, True, 7.4)
        smi74 = m.write("smi").strip().split("\t")[0]
    except (IOError, OSError, Exception):
        return None
    rm = Chem.MolFromSmiles(smi74)
    if rm is None:
        return None
    return Crippen.MolLogP(rm)


# ---------- 构件 2（原）：分子大小 —— 已移入 penetration_features.mol_size ----------
# 2026-08-06 挪轴：外排轴 → 穿透轴。度量不变（MW），列名不变（mol_size）。


# ---------- 汇总 ----------
def efflux_components(smiles: str) -> dict:
    """外排轴构件向量。现只有 slogp 一个构件（mol_size 已挪至穿透轴，
    gurvic_group 已删）。"""
    return {
        "smiles": smiles,
        "slogp": slogp(smiles),
    }


# ---------- 打分骨架：走甲，权重由 D1 提供 ----------
def score(components: dict, weights: dict) -> float:
    """
    外排分 = Σ w_i * 构件_i（走甲，纯分子描述符加权和）。
    weights 由 D1 在 CO-ADD tolC fold change 上拟合后传入，不在此硬编码。
    现只有一个构件（slogp）；仅解析失败时为 None。
    """
    missing = [k for k in weights if components.get(k) is None]
    if missing:
        raise ValueError(f"构件缺失，无法打分: {missing}")
    return sum(weights[k] * components[k] for k in weights)


# ---------------------------------------------------------------------------
# CLI：批量算外排构件表。列名 / 路径全走 argparse，LF 行终止。
# ---------------------------------------------------------------------------
def _batch(in_path: str, smiles_col: str, out_path: str) -> None:
    import csv
    sep = "\t" if in_path.endswith((".tsv", ".txt")) else ","
    rows = []
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if smiles_col not in reader.fieldnames:
            raise SystemExit(f"输入无列 '{smiles_col}'，现有列: {reader.fieldnames}")
        for row in reader:
            rows.append(efflux_components(row[smiles_col]))
    out_sep = "\t" if out_path.endswith((".tsv", ".txt")) else ","
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           delimiter=out_sep, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {len(rows)} 行 → {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="C2 关卡2 外排构件计算器")
    ap.add_argument("--in", dest="in_path", required=True, help="输入 csv/tsv（相对路径）")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES 列名")
    ap.add_argument("--out", dest="out_path", required=True, help="输出 csv/tsv（相对路径）")
    args = ap.parse_args()
    _batch(args.in_path, args.smiles_col, args.out_path)
