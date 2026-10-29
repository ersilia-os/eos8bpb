#!/usr/bin/env python
"""最小投影面积 v2 —— 补正 v1 的三处缺陷（未定 29，2026-08-12）。

★v1（a2_min_projection_area.py / penetration_features.min_projection_area）保留不动，
  本模块与其【并存】。v1 被既有产物依赖，删了会断出处。

【v1 的三处缺陷（08-11 逐分子审计，未定第 16 条加注）】
  ① 不加 vdW 半径 —— 算的是【原子中心点集】的凸包，不是截面。
     极端例：benzoate v1 得 0.00005 Å²（六个环原子近共面 → 点集凸包退化），
     ChemAxon 19 Å²。
  ② 只投影到垂直于第一主轴的【那一个】平面 —— 而 MinPA 的定义是
     【对所有方向求最小】。v1 自己的注释就写着"最小投影方向的近似"。
  ③ 在输入 SMILES 的【原态】上算，未按 pH 7.4 质子化。
  实测后果：Boi 42 个分子上 41/41 全部偏小，比值均值 0.392，
  Spearman(比值, MW) = +0.791（★偏差随分子变大而变大，是系统性的）。

【v2 的三处补正】
  ① 投影后每个原子按其 vdW 半径成圆，取【所有圆的并集面积】。
     半径 = Chem.GetPeriodicTable().GetRvdw(原子序数)。
     并集面积用【解析法】（Green 定理沿并集边界积分），不是格点近似：
       Area = 1/2 · Σ_i ∫_{圆 i 未被覆盖的弧} (x dy − y dx)
       其中对圆心 (cx,cy)、半径 r 的一段弧 θ:a→b
       ∫ (x dy − y dx) = cx·r·(sin b − sin a) − cy·r·(cos b − cos a) + r²·(b − a)
     每个圆被其他圆遮住的角区间用「排序 + 前缀最大值」合并（见 _union_disk_area）。
     ⚠️ 该式假定并集无内部空洞（分子投影里空洞需要环足够大且原子足够小；
        vdW 半径 1.2–1.9 Å 下未观察到）。自检见 self_check()。
  ② 方向在【半球】上做 Fibonacci 均匀采样，对所有方向取最小。
     投影方向与其反向给出同一张投影图（点集只是镜像，面积不变），
     故只取半球即可，不做全球采样 —— 全球采样等于把每个方向算两遍。
  ③ 先用【与 net_charge_pH74 同一套 OpenBabel 调用】生成 pH 7.4 微种，再嵌构象。
     衔接方式见 protonate_ph74()：OB 返回的是 OBMol 对象，不是 SMILES，
     故经 pybel 的 write("smi") 导出 SMILES 再交回 RDKit —— 这是唯一的衔接点。

【构象口径：三种都算，全部返回（未定 29 的"丁"）】
  lowest  MMFF 能量最低的那个构象的最小投影面积
  min     nconf 个构象里最小的那个最小投影面积
  mean    nconf 个构象的算术平均（★v1 用的就是这个口径）
  ★本模块不选，三个并排返回，由 Lang 判。

【不做的事】
  不改任何构件语义、不碰 v1、不重跑任何下游产物。
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

__all__ = [
    "min_projection_area_v2",
    "protonate_ph74",
    "fibonacci_hemisphere",
    "union_disk_area",
    "self_check",
]

_PT = Chem.GetPeriodicTable()


# ---------------------------------------------------------------- pH 7.4 微种
def protonate_ph74(smiles: str) -> "str | None":
    """用 net_charge_pH74 的同一套 OpenBabel 调用生成 pH 7.4 微种，回传 SMILES。

    ⚠️ 衔接方式：OBMol.AddHydrogens(False, True, 7.4) 就地修改的是【OBMol 对象】，
       不返回 SMILES。本函数用 pybel 的 write("smi") 把该对象导出成 SMILES 字符串，
       再由调用方交给 RDKit 嵌构象。这是 OB 与 RDKit 之间唯一的衔接点，
       中间不做任何中性化/去盐/规范化。
    """
    from openbabel import pybel
    m = pybel.readstring("smi", smiles)
    m.OBMol.AddHydrogens(False, True, 7.4)   # 与 penetration_features.net_charge_pH74 逐参数相同
    out = m.write("smi").strip().split()[0] if m.write("smi").strip() else ""
    return out or None


# ---------------------------------------------------------------- 方向采样
def fibonacci_hemisphere(n_dir: int) -> np.ndarray:
    """半球上的 Fibonacci 均匀采样，返回 (n_dir, 3) 单位向量，z >= 0。

    只取半球：投影方向 u 与 −u 给出互为镜像的同一张二维图，面积相同，
    所以全球采样是纯粹的重复劳动。
    """
    k = np.arange(n_dir, dtype=float) + 0.5
    z = k / n_dir                      # (0, 1) -> 只覆盖上半球
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _basis_from_normal(u: np.ndarray) -> np.ndarray:
    """给每个法向量 u 配一组正交基 (e1, e2)，返回 (n_dir, 2, 3)。"""
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(u), 1))
    flip = np.abs(u[:, 2]) > 0.9
    helper[flip] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    return np.stack([e1, e2], axis=1)


# ---------------------------------------------------------------- 圆并集面积
def _union_disk_area_batch(P: np.ndarray, r: np.ndarray) -> np.ndarray:
    """一批投影的圆并集面积。P (D, n, 2) 圆心，r (n,) 半径 -> (D,) 面积。

    解析法。对每个圆 i：
      · 若被某个圆 j 完全包含 -> 不贡献边界。
      · 否则把所有「被 j 遮住的角区间」合并，未被遮住的弧按 Green 定理积分。
    区间合并用排序 + 前缀最大值：按起点排序后，
      piece_k = [max(s_k, m_k), max(e_k, m_k)]（m_k = 前 k 个终点的最大值）
    这些 piece 互不相交且并集等于原并集，而 Green 积分对区间可加。
    """
    D, n, _ = P.shape
    if n == 0:
        return np.zeros(D)
    if n == 1:
        return np.full(D, np.pi * r[0] ** 2)

    TWO_PI = 2.0 * np.pi
    eps = 1e-9

    delta = P[:, None, :, :] - P[:, :, None, :]          # (D,n,n,2)  = P_j - P_i
    dist = np.linalg.norm(delta, axis=-1)                # (D,n,n)
    ri = r[None, :, None]
    rj = r[None, None, :]
    idx = np.arange(n)
    eye = (idx[:, None] == idx[None, :])[None, :, :]

    # 圆 i 被圆 j 完全包含。等半径且重合时按序号打破平局，避免互相判死。
    bigger = (rj > ri + eps) | ((np.abs(rj - ri) <= eps) & (idx[None, :] < idx[:, None])[None, :, :])
    contained = (dist + ri <= rj + eps) & (~eye) & bigger
    dead = contained.any(axis=2)                          # (D,n)

    # 真相交
    proper = (dist < ri + rj - eps) & (dist > np.abs(ri - rj) + eps) & (~eye)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = (dist ** 2 + ri ** 2 - rj ** 2) / (2.0 * dist * ri)
    cos_a = np.clip(np.nan_to_num(cos_a, nan=1.0), -1.0, 1.0)
    alpha = np.arccos(cos_a)                              # 半角宽
    phi = np.arctan2(delta[..., 1], delta[..., 0])        # i 指向 j 的方位角

    s = np.mod(phi - alpha, TWO_PI)
    e = s + 2.0 * alpha

    # 跨 2π 的区间：原区间截到 [0,2π]，再补一份平移 −2π 后截到 [0,2π]
    sA = np.where(proper, s, 0.0)
    eA = np.where(proper, np.minimum(e, TWO_PI), 0.0)
    sB = np.zeros_like(s)
    eB = np.where(proper, np.maximum(e - TWO_PI, 0.0), 0.0)

    starts = np.concatenate([sA, sB], axis=2)             # (D,n,2n)
    ends = np.concatenate([eA, eB], axis=2)
    ends = np.maximum(ends, starts)                       # 保证非负长度

    order = np.argsort(starts, axis=2)
    starts = np.take_along_axis(starts, order, axis=2)
    ends = np.take_along_axis(ends, order, axis=2)

    run = np.maximum.accumulate(ends, axis=2)
    prev = np.concatenate([np.zeros_like(run[:, :, :1]), run[:, :, :-1]], axis=2)
    a = np.maximum(starts, prev)
    b = np.maximum(ends, prev)

    cx = P[:, :, 0:1]
    cy = P[:, :, 1:2]
    rr = r[None, :, None]

    def _green(t0, t1):
        return (cx * rr * (np.sin(t1) - np.sin(t0))
                - cy * rr * (np.cos(t1) - np.cos(t0))
                + rr ** 2 * (t1 - t0))

    covered = _green(a, b).sum(axis=2)                    # (D,n)
    full = 2.0 * np.pi * r[None, :] ** 2                  # G_i(0, 2π)
    contrib = np.where(dead, 0.0, full - covered)
    return 0.5 * contrib.sum(axis=1)


def union_disk_area(centers: np.ndarray, radii: np.ndarray) -> float:
    """单张投影的圆并集面积（解析法）。"""
    P = np.asarray(centers, dtype=float)[None, :, :]
    return float(_union_disk_area_batch(P, np.asarray(radii, dtype=float))[0])


def _union_disk_area_grid(centers, radii, step=0.01):
    """数值格点参照实现，只用于 self_check，不进生产路径。"""
    c = np.asarray(centers, float)
    r = np.asarray(radii, float)
    lo = (c - r[:, None]).min(0) - step
    hi = (c + r[:, None]).max(0) + step
    xs = np.arange(lo[0], hi[0], step)
    ys = np.arange(lo[1], hi[1], step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    inside = np.zeros(gx.shape, bool)
    for (px, py), rad in zip(c, r):
        inside |= ((gx - px) ** 2 + (gy - py) ** 2) <= rad ** 2
    return inside.sum() * step * step


# ---------------------------------------------------------------- 主函数
def min_projection_area_v2(smiles: str, nconf: int = 20, seed: int = 42,
                           n_dir: int = 500, protonate: bool = True,
                           dir_chunk: int = 64) -> "dict | None":
    """最小投影面积 v2。返回 {'lowest':…, 'min':…, 'mean':…, …} 或 None。

    参数
      nconf      构象数（默认 20，与 v1 同）
      seed       ETKDG 随机种子（默认 42，与 v1 同）
      n_dir      半球方向采样数 —— ★本轮跑 200 / 500 / 2000 三档比收敛
      protonate  是否先做 pH 7.4 质子化（缺省 True，即 v2 口径）
    """
    smi_used = smiles
    if protonate:
        p = protonate_ph74(smiles)
        if p:
            smi_used = p

    m = Chem.MolFromSmiles(smi_used)
    if m is None:
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=params)
    if len(cids) == 0:
        cids = AllChem.EmbedMultipleConfs(m, numConfs=nconf, randomSeed=seed,
                                          useRandomCoords=True)
    if len(cids) == 0:
        return None

    mmff = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    energies = [e for _, e in mmff]

    radii = np.array([_PT.GetRvdw(a.GetAtomicNum()) for a in m.GetAtoms()], dtype=float)
    dirs = fibonacci_hemisphere(n_dir)
    basis = _basis_from_normal(dirs)                      # (n_dir, 2, 3)

    per_conf = []
    for c in cids:
        conf = m.GetConformer(c)
        pts = conf.GetPositions()
        pts = pts - pts.mean(0)
        best = np.inf
        for lo in range(0, n_dir, dir_chunk):
            bb = basis[lo:lo + dir_chunk]                 # (D,2,3)
            P = np.einsum("dkc,nc->dnk", bb, pts)         # (D,n,2)
            best = min(best, float(_union_disk_area_batch(P, radii).min()))
        per_conf.append(best)

    per_conf = np.array(per_conf, dtype=float)
    ok = np.isfinite(per_conf)
    if not ok.any():
        return None

    if energies and len(energies) == len(per_conf) and np.isfinite(energies).all():
        lowest = float(per_conf[int(np.argmin(energies))])
    else:
        lowest = float(per_conf[0])

    return {
        "lowest": lowest,
        "min": float(per_conf[ok].min()),
        "mean": float(per_conf[ok].mean()),
        "n_conf": int(ok.sum()),
        "n_dir": int(n_dir),
        "smiles_used": smi_used,
    }


# ---------------------------------------------------------------- 自检
def self_check(verbose: bool = True) -> bool:
    """解析可算的例子 + 数值格点交叉验证。"""
    okall = True

    def _rep(label, got, want, tol):
        nonlocal okall
        good = abs(got - want) <= tol
        okall &= good
        if verbose:
            print(f"  {'✅' if good else '❌'} {label:46s} 得 {got:12.6f}  应 {want:12.6f}  "
                  f"差 {got - want:+.2e}")

    # ① 单圆
    _rep("单圆 r=1.7", union_disk_area([[0, 0]], [1.7]), np.pi * 1.7 ** 2, 1e-9)

    # ② 两个等半径圆，r=1, d=1 —— 透镜面积有闭式解
    lens = 2 * 1.0 ** 2 * np.arccos(1.0 / 2.0) - (1.0 / 2.0) * np.sqrt(4 - 1.0 ** 2)
    _rep("两圆 r=1,d=1（并集 = 2πr² − 透镜）",
         union_disk_area([[0, 0], [1, 0]], [1.0, 1.0]), 2 * np.pi - lens, 1e-9)

    # ③ 两个不等半径圆，r=1.7/1.2, d=1.09（C–H 键长量级）
    R, rr, d = 1.7, 1.2, 1.09
    l1 = R ** 2 * np.arccos((d ** 2 + R ** 2 - rr ** 2) / (2 * d * R))
    l2 = rr ** 2 * np.arccos((d ** 2 + rr ** 2 - R ** 2) / (2 * d * rr))
    l3 = 0.5 * np.sqrt((-d + R + rr) * (d + R - rr) * (d - R + rr) * (d + R + rr))
    _rep("两圆 r=1.7/1.2,d=1.09（C–H）",
         union_disk_area([[0, 0], [d, 0]], [R, rr]), np.pi * R ** 2 + np.pi * rr ** 2 - (l1 + l2 - l3), 1e-9)

    # ④ 完全包含
    _rep("小圆完全落在大圆内", union_disk_area([[0, 0], [0.2, 0]], [2.0, 0.5]), np.pi * 4.0, 1e-9)

    # ⑤ 两个完全重合的同半径圆（平局打破）
    _rep("两圆完全重合 r=1.5", union_disk_area([[0, 0], [0, 0]], [1.5, 1.5]), np.pi * 2.25, 1e-9)

    # ⑥ 相离
    _rep("两圆相离 d=10", union_disk_area([[0, 0], [10, 0]], [1.0, 1.0]), 2 * np.pi, 1e-9)

    # ⑦ 随机点云 vs 数值格点（含可能成环的构型）
    rng = np.random.default_rng(7)
    for trial, npt in enumerate([5, 12, 30]):
        c = rng.normal(0, 2.2, size=(npt, 2))
        rad = rng.uniform(1.1, 1.9, size=npt)
        _rep(f"随机点云 n={npt} vs 格点(0.01Å)",
             union_disk_area(c, rad), _union_disk_area_grid(c, rad, 0.01), 0.02)

    # ⑧ 环状排布（检验有无内部空洞导致的偏差）
    for nring, ringR in [(6, 1.4), (12, 5.0), (18, 9.0)]:
        th = np.linspace(0, 2 * np.pi, nring, endpoint=False)
        c = np.column_stack([ringR * np.cos(th), ringR * np.sin(th)])
        rad = np.full(nring, 1.7)
        _rep(f"环形排布 n={nring} R={ringR}（空洞检验）",
             union_disk_area(c, rad), _union_disk_area_grid(c, rad, 0.01), 0.03)

    if verbose:
        print(f"\n  自检总判定：{'✅ 全部通过' if okall else '❌ 有未通过项'}")
    return okall


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
