#!/usr/bin/env python3
"""C7 运行 G —— Hyper wiener index 与 Platt index 的 ChemAxon 口径与开源复现。

本模块只放实现，不跑数据。三个脚本共用：
  c7_run_g_step0_libsearch.py  库直出接口普查（决定能不能手写）
  c7_run_g_step2_probes.py     判别分子探针（先钉口径）
  c7_run_g_step3_reproduce.py  5,000 分子逐分子对真值（同 seed 42、同 mol_id）

================================================================ 查证来源
  [1] Topological Analysis Plugin, docs.chemaxon.com/latest/
      calculators_topological-analysis-plugin.html
  [2] cxcalc calculator functions, docs.chemaxon.com/latest/
      calculators_cxcalc-calculator-functions.html
  [3] M. Randic, Chem. Phys. Lett. 211, 478-483 (1993)
      —— hyper-Wiener 的**原始**定义，只对树成立
  [4] D. J. Klein, I. Lukovits, I. Gutman, J. Chem. Inf. Comput. Sci. 35, 50-52
      (1995), DOI 10.1021/ci00023a007
      —— 把 hyper-Wiener 推广到含环图的距离矩阵定义；在树上与 [3] 恒等
  [5] J. R. Platt, J. Chem. Phys. 15, 419-420 (1947) —— Platt 数
  完整查证记录见 results/c7_run_g/run_g_step1_defs.md

=================================================== 第 0 步：库有没有直出接口
六个库实证查过（脚本 c7_run_g_step0_libsearch.py，明细 run_g_step0_libsearch.tsv）：

  RDKit 2022.09.5 / Mordred 1.2.0 / CDK 2.12 / PaDEL 2.21 —— 两个都★无
      （scyjava、cdk-pywrapper 只是调 CDK 的桥，不新增描述符；
        PaDEL 内嵌 cdk-1.4.15，Wiener 一族同样只有 WienerNumbers = WPATH/WPOL）
  PyBioMed 1.0    —— **有 Platt**（CalculatePlatt），无 hyper-Wiener
  chemopy2 1.0.11 —— **有 Platt**（calculate_platt，与上式逐字相同），无 hyper-Wiener

⇒ **Platt index 用库**（PyBioMed 的 CalculatePlatt），不手写。
⇒ **Hyper wiener index 无库**，手写，且只实现 [4] 的公开定义。

============================================================ Hyper wiener index

[1] 原文（唯一一句）：「Hyper Wiener index: a variant of the Wiener index.」
[2] 只给回显名与 `-s, --single`，**没有 `-p/--precision`**（对比 balabanindex 有）。

  ⇒ **哪一版、含不含氢、距离矩阵是什么、多片段怎么办，ChemAxon 文档一概未载。**
     下面每一条都是探针在真值上判出来的。

  文献口径 [4]（本模块实现的那个）：
      WW(G) = 1/2 * SUM_{i<j} ( d_ij + d_ij^2 )
      在树上恒等于 [3] 的路径求和定义；[3] 对含环图根本没定义，
      所以含环分子能分辨两版 —— 由探针 V1 裁决，不由本模块断言。
      d_ij + d_ij^2 = d_ij(d_ij+1) 恒为偶数 ⇒ WW 恒为整数，与真值该列全整数相符。

  探针判定（逐条见 step2 的 V1/H1/D1/F1）：
    V1 线性项 d 在不在里面（½Σ(d+d²) vs ½Σd²）
    H1 含不含氢
    D1 距离矩阵是纯拓扑还是键级加权（沿用运行 F 的 B1 做法）
    F1 多片段怎么处理

================================================================= Platt index

[1] 原文（唯一一句）：「Platt index: sum of the edge degrees of a molecular graph.」
[2] 只给回显名与 `-s, --single`。

  文献口径 [5]：
      F(G) = SUM_{(i,j) in E} ( delta_i + delta_j - 2 )
  括号里就是边 (i,j) 的 edge degree（与该边相邻的边数），与 [1] 的
  「sum of the edge degrees」逐字对上。等价写法 F = M1 - 2q（M1 = 第一 Zagreb 指数）
  ⇒ Zagreb1 与 Platt 差 2q，不是同一个量。

  ★但 [1] 没有定义「edge degree」本身，也没提氢、没提片段：
    P1 edge degree 是 delta_i+delta_j-2 还是 delta_i+delta_j
    H2 度数含不含氢       ★两列的氢口径不一定相同，各判各的
    F2 多片段怎么处理

  实现一律调 PyBioMed 的 CalculatePlatt；「含不含氢」「取哪个片段」不是库的事，
  是**喂给库的图**的事，由本模块在调用前决定。

================================================================ 编码
只用 RDKit 的库原语（GetDistanceMatrix / GetMolFrags / AddHs）与 PyBioMed 的
CalculatePlatt，不手写图算法。
"""

import os
import sys

import numpy as np
from rdkit import Chem

# --------------------------------------------------------- PyBioMed 延迟装载
#
# PyBioMed 用 `pip install --no-deps --target` 装在 results/c7_run_g/_lib/pybiomed，
# 路径由调用方经 argparse 传入（编码规范：无默认绝对路径）。

_PB_PLATT = None


def load_pybiomed(path):
    """把 PyBioMed 的 --target 目录挂进 sys.path 并取出 CalculatePlatt。"""
    global _PB_PLATT
    if not os.path.isdir(path):
        sys.exit("PyBioMed 路径不存在或不是目录: %s" % path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from PyBioMed.PyMolecule.topology import CalculatePlatt   # noqa: PLC0415
    _PB_PLATT = CalculatePlatt
    return _PB_PLATT


def _platt_lib(mol):
    if _PB_PLATT is None:
        sys.exit("PyBioMed 未装载 —— 先调 load_pybiomed(path)。拒绝用手写公式顶替库。")
    return int(_PB_PLATT(mol))


# ------------------------------------------------------------------ 图的准备

def largest_fragment(mol):
    """多片段只取原子数最大的片段（沿用运行 F 对 Balaban 的 B5 口径记法）。

    原子数相同时取 RDKit 片段序里的第一个。
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, key=lambda m: m.GetNumAtoms())


def _prep(mol, addH, frag):
    """按 (含不含氢, 片段口径) 造出要算的那张图，返回图的列表。

    frag="largest" —— 只留最大片段，返回 1 张图
    frag="all"     —— 每个片段一张图，返回 n 张（调用方求和 = 「逐片段相加」读法）
    frag="whole"   —— 整图一张，不做片段处理（含多片段时距离矩阵会出 1e8）
    """
    if frag == "largest":
        graphs = [largest_fragment(mol)]
    elif frag == "all":
        graphs = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    elif frag == "whole":
        graphs = [mol]
    else:
        raise ValueError("未知片段口径: %r" % frag)
    return [Chem.AddHs(g) for g in graphs] if addH else graphs


# ========================================================== Hyper wiener index

def _ww_one(m, useBO):
    """[4] 的 WW = ½ Σ_{i<j} (d_ij + d_ij²)，距离矩阵调 RDKit 库原语。"""
    n = m.GetNumAtoms()
    if n < 2:
        return 0
    D = Chem.GetDistanceMatrix(m, useBO=1 if useBO else 0, useAtomWts=0, force=1)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    return float(0.5 * (d.sum() + (d * d).sum()))


def _ww_sq_one(m, useBO):
    """对照读法：去掉线性项，只留 ½Σd²（V1 的另一读法）。"""
    n = m.GetNumAtoms()
    if n < 2:
        return 0
    D = Chem.GetDistanceMatrix(m, useBO=1 if useBO else 0, useAtomWts=0, force=1)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    return float(0.5 * (d * d).sum())


def hyper_wiener(mol, addH=False, frag="largest", useBO=False, square_only=False):
    """Hyper-Wiener，默认参数 = 探针全过的那套口径。

    其余参数是探针用的对照读法开关，第 3 步一律用默认值。
    """
    fn = _ww_sq_one if square_only else _ww_one
    return float(sum(fn(g, useBO) for g in _prep(mol, addH, frag)))


# =============================================================== Platt index

def platt(mol, addH=True, frag="largest", minus2=True):
    """Platt，默认参数 = 探针全过的那套口径。

    minus2=True  —— 调 PyBioMed 的 CalculatePlatt（= [5] 的 Σ(δᵢ+δⱼ−2)）
    minus2=False —— 对照读法 Σ(δᵢ+δⱼ)，即把「edge degree」读成不减 2（探针 P1）。
                    这一支不是库口径，只在探针里用，不进第 3 步。
    """
    tot = 0
    for g in _prep(mol, addH, frag):
        if minus2:
            tot += _platt_lib(g)
        else:
            tot += sum(b.GetBeginAtom().GetDegree() + b.GetEndAtom().GetDegree()
                       for b in g.GetBonds())
    return float(tot)


# ---------------------------------------------------------------- 特征表

FEATURES = [
    ("Hyper wiener index", "",
     "手写 [4] Klein 1995：½Σ_{i<j}(d+d²)，重原子图、纯拓扑距离、最大片段",
     lambda m: hyper_wiener(m)),
    ("Hyper wiener index", "addH",
     "同上但加显式氢（对照）",
     lambda m: hyper_wiener(m, addH=True)),
    ("Platt index", "",
     "PyBioMed CalculatePlatt（= [5] Σ(δᵢ+δⱼ−2)），加显式氢、最大片段",
     lambda m: platt(m)),
    ("Platt index", "noH",
     "同上但重原子图、不加氢（对照）",
     lambda m: platt(m, addH=False)),
]

TARGET_COLS = ["Hyper wiener index", "Platt index"]


def compute_all(mol):
    out = {}
    for col, variant, _note, fn in FEATURES:
        key = col if not variant else "%s [%s]" % (col, variant)
        out[key] = float(fn(mol))
    return out
