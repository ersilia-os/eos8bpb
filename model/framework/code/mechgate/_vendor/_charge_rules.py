#!/usr/bin/env python
"""net_charge_pH74 的 SMARTS 后处理层（未定 34 · 方案 1，2026-08-12）。

★penetration_features.net_charge_pH74（v1）保留不动，本模块与其并存。

【为什么需要后处理层】
  OpenBabel 的 phmodel 在两个官能团上与 ChemAxon 分歧，08-12 逐探针实测：

  (a) 磺酰胺酸性 —— phmodel 里【根本没有这条规则】，是整类缺失：
        asulam(pKa~4) 0 ／ 磺胺甲噁唑(5.7) 0 ／ ★糖精(1.6) 0 ／ 甲磺酰胺 0
      对照（OB 有规则、行为正确）：羧酸(4.8) −1 ✅ ／ 四氮唑(4.9) −1 ✅
      连 pKa 1.6 的糖精都不去质子，说明不是边界情况而是没有规则。

  (b) 脒规则只对【脒碳带碳取代】的生效：
        乙脒 +1 ✅ ／ N-甲基乙脒 +1 ✅ ／ 胍 +1 ✅
        ★甲脒 0 ❌ ／ N-甲基甲脒 0 ❌ ／ imipenem 片段 CSCCNC=N 0 ❌
      imipenem 的基团正是 N-取代甲脒，落在盲区里。

【口径：在 OB 的【输出分子】上匹配，不在输入串上匹配】
  ★这是避免与 OB 重复计数的关键。OB 已经质子化过的位点在输出串里带 +/−，
  而所有规则的 SMARTS 都要求相关原子【中性】（!+;!-），故 OB 已处理的位点
  自动不再匹配。例：乙脒经 OB 变成 CC(=[NH2+])N，规则 2 不会再加一次。

【规则可逐条关掉】
  net_charge_pH74_v2(smi, rules=('sulfonamide',))        只开规则 1
  net_charge_pH74_v2(smi, rules=())                       等价于 v1（纯 OB）

⚠️【与任务书的一处冲突，未自行放宽，交 Lang 裁】
  任务书要求规则 1 同时命中「甲磺酰胺」并【不】命中 sulfanilamide 与对甲苯磺酰胺。
  这三者都是伯磺酰胺 R-SO2-NH2，实测 pKa 分别为 10.8 / 10.4 / 10.2 —— 同一酸性
  档次，pH 7.4 下都基本不解离。任何能命中甲磺酰胺的 SMARTS 必然同时命中另外两个
  （已实证）。故本模块采用【化学上站得住】的窄口径：只命中 N 上带吸电子取代基、
  实测 pKa < 7 的那一类（asulam 4 ／ sulfacetamide 5.4 ／ 磺胺甲噁唑 5.7 ／ 糖精 1.6），
  ★甲磺酰胺不命中。若 Lang 要求改为宽口径，把 SULFONAMIDE_WIDE 接进去即可，
  但那会把 sulfanilamide（Boi 41 里就有）算成 −1。
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = ["net_charge_pH74_v2", "apply_charge_rules", "RULES", "SULFONAMIDE_WIDE"]


# 规则 1：磺酰胺 N 上带吸电子取代基 -> pH 7.4 已解离，每命中一处 −1
#   R1a  N-酰基 / N-氨基甲酰基 / N-氨基甲酸酯（asulam 4、sulfacetamide 5.4、糖精 1.6）
#   R1b  N-杂芳基，芳环上与连接碳相邻处含杂原子（磺胺甲噁唑 5.7、磺胺嘧啶、磺胺噻唑）
#   R1c  双磺酰亚胺（更强的酸）
_SULFONAMIDE = [
    ("R1a", "[NX3;H1;!+;!-](-[CX3]=[OX1,SX1])-[SX4](=[OX1])(=[OX1])"),
    ("R1b", "[NX3;H1;!+;!-](-[SX4](=[OX1])(=[OX1]))-[c;$(c~[n,o,s;a])]"),
    ("R1c", "[NX3;H1;!+;!-](-[SX4](=[OX1])(=[OX1]))-[SX4](=[OX1])(=[OX1])"),
]

# 规则 2：N-取代甲脒（脒碳带 H）-> pH 7.4 已质子化，每命中一处 +1
#   [CX3;H1] 把乙脒/胍/N-甲基乙脒（脒碳无 H）排除在外 —— 那些 OB 已经处理过。
#   [NX2;!+;!-] 与 [NX3;!+;!-] 保证只在【两个 N 都还中性】时才加，不重复计数。
#   芳环上的脒样片段（腺嘌呤、胞嘧啶）用大写 C/N 天然排除。
_FORMAMIDINE = [
    ("R2", "[NX3;!+;!-;H1,H2]-[CX3;H1]=[NX2;!+;!-]"),
]

RULES = {
    "sulfonamide": (-1, _SULFONAMIDE),
    "formamidine": (+1, _FORMAMIDINE),
}

# ★仅供 Lang 参考的宽口径（把伯磺酰胺也算进去）。默认【不】启用，理由见模块 docstring。
SULFONAMIDE_WIDE = ("R1w", "[NX3;H2;!+;!-]-[SX4](=[OX1])(=[OX1])")

_COMPILED = {name: (d, [(lab, Chem.MolFromSmarts(s)) for lab, s in pats])
             for name, (d, pats) in RULES.items()}

DEFAULT_RULES = ("sulfonamide", "formamidine")


def apply_charge_rules(ob_smiles: str, rules=DEFAULT_RULES):
    """在 OB 的输出串上跑后处理规则。-> (电荷修正量, 命中标签列表)。

    ★必须传 OB 【输出】的 SMILES，不是原始输入串 —— 见模块 docstring 的重复计数说明。
    """
    mol = Chem.MolFromSmiles(ob_smiles) if ob_smiles else None
    if mol is None:
        return 0, []
    delta, fired = 0, []
    for name in rules:
        if name not in _COMPILED:
            raise ValueError(f"未知规则 {name!r}；可用：{sorted(_COMPILED)}")
        d, pats = _COMPILED[name]
        for lab, patt in pats:
            if patt is None:
                continue
            n = len(mol.GetSubstructMatches(patt))
            if n:
                delta += d * n
                fired.append(f"{lab}×{n}" if n > 1 else lab)
    return delta, fired


def _ob_ph74(smiles: str):
    """与 penetration_features.net_charge_pH74 逐参数相同的 OB 调用。

    ⚠️ AddHydrogens 就地改的是 OBMol 对象，不返回 SMILES；经 pybel 的 write("smi")
       导出字符串再交回 RDKit —— 与 a2_min_projection_area_v2.protonate_ph74 同一衔接方式。
    """
    from openbabel import pybel
    m = pybel.readstring("smi", smiles)
    m.OBMol.AddHydrogens(False, True, 7.4)
    out = m.write("smi").strip()
    return m.OBMol.GetTotalCharge(), (out.split()[0] if out else "")


def net_charge_pH74_v2(smiles: str, rules=DEFAULT_RULES, with_detail: bool = False):
    """净电荷 @ pH 7.4，v2 = OpenBabel 结果 + SMARTS 后处理层。

    rules        要启用的规则名元组；() 等价于 v1（纯 OB）。可逐条关掉。
    with_detail  True 时返回 (charge, fired, ob_charge, ob_smiles) 便于留档
    """
    try:
        q_ob, ob_smi = _ob_ph74(smiles)
    except Exception:
        return None
    delta, fired = apply_charge_rules(ob_smi, rules)
    q = int(q_ob) + delta
    if with_detail:
        return q, fired, int(q_ob), ob_smi
    return q


def self_check(verbose: bool = True) -> bool:
    """规则 1/2 的命中对照表。★"应不命中却命中"一项都不许有。"""
    cases = [
        # (标签, SMILES, 期望是否命中, 规则)
        ("asulam (pKa~4)", "COC(=O)NS(=O)(=O)c1ccc(N)cc1", True, "sulfonamide"),
        ("磺胺甲噁唑 (5.7)", "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", True, "sulfonamide"),
        ("糖精 (1.6)", "O=C1NS(=O)(=O)c2ccccc21", True, "sulfonamide"),
        ("sulfacetamide (5.4)", "CC(=O)NS(=O)(=O)c1ccc(N)cc1", True, "sulfonamide"),
        ("★sulfanilamide (10.4) 应不命中", "NS(=O)(=O)c1ccc(N)cc1", False, "sulfonamide"),
        ("★对甲苯磺酰胺 (10.2) 应不命中", "Cc1ccc(S(N)(=O)=O)cc1", False, "sulfonamide"),
        ("苯磺酰胺 (10.1) 应不命中", "NS(=O)(=O)c1ccccc1", False, "sulfonamide"),
        ("★甲磺酰胺 (10.8) —— 见 docstring 冲突", "CS(N)(=O)=O", False, "sulfonamide"),
        ("乙酰苯胺 应不命中", "CC(=O)Nc1ccccc1", False, "sulfonamide"),
        ("甲脒", "NC=N", True, "formamidine"),
        ("N-甲基甲脒", "CNC=N", True, "formamidine"),
        ("imipenem 片段", "CSCCNC=N", True, "formamidine"),
        ("★乙脒 应不重复计数", "CC(=N)N", False, "formamidine"),
        ("★胍 应不重复计数", "NC(=N)N", False, "formamidine"),
        ("★N-甲基乙脒 应不重复计数", "CNC(C)=N", False, "formamidine"),
        ("甲酰胺 应不命中", "NC=O", False, "formamidine"),
        ("腺嘌呤(芳香) 应不命中", "Nc1ncnc2[nH]cnc12", False, "formamidine"),
        ("胞嘧啶 应不命中", "Nc1cc[nH]c(=O)n1", False, "formamidine"),
    ]
    over = ok = 0
    for lab, smi, want, rule in cases:
        _, ob_smi = _ob_ph74(smi)
        _, fired = apply_charge_rules(ob_smi, (rule,))
        got = bool(fired)
        good = got == want
        ok += good
        if got and not want:
            over += 1
        if verbose:
            mark = "✅" if good else ("❌应不命中却命中" if got else "⚠️应命中却未命中")
            print(f"  {mark} [{rule:11s}] {lab:36s} 命中={fired or '—'}")
    if verbose:
        print(f"\n  {ok}/{len(cases)} 相符；★应不命中却命中：{over}（必须为 0）")
    return over == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
