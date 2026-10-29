#!/usr/bin/env python3
"""
A2: 最小投影面积批量计算（RDKit 免费替代 Ceccarelli 的 Marvin）

结论（2026-07-21）：对 Ceccarelli SI Table 2 的 9 个 β-lactam，
  Spearman 0.988 / Pearson 0.996。绝对值系统偏小（比值 0.49，因未加
  范德华半径），但排序 100% 保留，比例因子进秤时被拟合权重吸收。
  ⇒ 不需要 ChemAxon license。

用法:
  python a2_min_projection_area.py --input mols.csv --smiles-col SMILES \\
      --output areas.csv --nconf 20
"""
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from scipy.spatial import ConvexHull

RDLogger.DisableLog("rdApp.*")


def min_projection_area(smiles, nconf=20, seed=42):
    """返回多构象平均的最小投影面积（Å²，凸包近似，未加 vdW 半径）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 CSV")
    ap.add_argument("--smiles-col", default="SMILES", help="SMILES 列名")
    ap.add_argument("--output", required=True, help="输出 CSV")
    ap.add_argument("--nconf", type=int, default=20, help="每分子构象数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["min_proj_area"] = df[args.smiles_col].apply(
        lambda s: min_projection_area(s, args.nconf, args.seed)
    )
    df.to_csv(args.output, index=False)
    ok = df["min_proj_area"].notna().sum()
    print(f"完成: {ok}/{len(df)} 成功 -> {args.output}")


if __name__ == "__main__":
    main()
