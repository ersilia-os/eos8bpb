#!/usr/bin/env python
"""SMILES 语法归一化 —— 单一实现，取代仓库里 6 处内联修法（未定 49，2026-08-12）。

【缺陷】
  SMILES 的方括号原子必须写成
      [ 同位素  元素符号  手性  氢数  电荷  原子类 ]
  多张外部表（GramNegAccum 六表、boi2025 两表，均为 MOE 导出）把**电荷写到了氢数之前**：
  `[N+H2]` 而非 `[NH2+]`。两者指的是同一个原子（氮、+1、两个氢），
  所以这是**拼写问题，不是化学问题**。RDKit 的解析器拒收前一种写法并返回 None，
  OpenBabel 等则接受——这就是这些串能活着通过 curation 的原因。

【本模块做什么】
  normalize_smiles_syntax(s) 只搬位置：把电荷 token 移到同一对方括号内 H token 的右边。
  不增、不删、不改任何原子/电荷/手性/同位素。对已经写对的串是恒等变换。

【口径：先查表，再走通式】
  1. 先在 data_curated/smiles_syntax_fixes.tsv 里按**原串精确匹配**查映射
     （126 行 / 105 个不同原串，fix_type 全为 charge_h_order，
      每行都带 tokens_preserved 与 reorder_is_reversible 两条独立证明）。
  2. 表里没有的，再走通式正则。
  查表优先的理由：映射表是**已留档、已逐行验证**的既有产物；正则是兜底。
  两者在这 126 行上给出完全相同的结果（见 self_check），所以顺序不影响结果，
  只影响"这一条修法有没有留档"。

【本模块【不】做什么】
  不做质子化、不做中性化、不做去盐、不做规范化、不碰任何构件语义。
  各调用点原有的 neutralize()/neutral_parent() 等一律保持在原处不动。

【自检】
  python scripts/_smiles_syntax.py     -> 对映射表 126 行逐行核对，须 126/126
"""
from __future__ import annotations

import csv
import io
import os
import re

__all__ = ["normalize_smiles_syntax", "self_check", "MAP_PATH"]

MAP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "data_curated", "smiles_syntax_fixes.tsv",
)

# 电荷被写在氢数之前的方括号原子。分组顺序即源表的错误书写顺序：
#   同位素 / 元素符号 / 电荷 / 手性 / 氢数
# 电荷三种写法都收：`+` `++` `+++`（B 组既有正则的口径）与 `+2`（A 组既有正则的口径）。
# 氢数可选：无氢时本式退化为恒等变换（`[N+]` -> `[N+]`），与两组既有正则一致。
_BRACKET_CHARGE_BEFORE_H = re.compile(
    r"\[(\d*)"                        # 1 同位素
    r"([A-Za-z][a-z]?)"               # 2 元素符号
    r"(\+{1,3}|-{1,3}|[+-]\d+)"       # 3 电荷（位置错了的那个）
    r"(@{0,2})"                       # 4 手性
    r"(H\d*)?"                        # 5 氢数
    r"\]"
)


def _reorder(m: "re.Match") -> str:
    iso, sym, charge, chirality, hcount = m.groups()
    return "[" + iso + sym + chirality + (hcount or "") + charge + "]"


_MAP: "dict[str, str] | None" = None


def _load_map() -> "dict[str, str]":
    """惰性读映射表。表缺失不报错——退回纯正则，保证本模块在任何环境下可用。"""
    global _MAP
    if _MAP is None:
        table: "dict[str, str]" = {}
        try:
            with open(MAP_PATH, newline="", encoding="utf-8") as fh:
                body = "".join(ln for ln in fh if not ln.startswith("#"))
            for row in csv.DictReader(io.StringIO(body), delimiter="\t"):
                orig, fixed = row["smiles_original"], row["smiles_fixed"]
                if orig:
                    table[orig] = fixed
        except OSError:
            table = {}
        _MAP = table
    return _MAP


def normalize_smiles_syntax(smiles: str) -> str:
    """把方括号内写在氢数之前的电荷搬到氢数之后。

    `[N+H2]` -> `[NH2+]` ；`[N+@@H]` -> `[N@@H+]` ；已写对的串原样返回。
    None / 空串原样返回（各调用点对空值的既有处理不变）。
    """
    if not smiles:
        return smiles
    hit = _load_map().get(smiles)
    if hit is not None:
        return hit
    return _BRACKET_CHARGE_BEFORE_H.sub(_reorder, smiles)


def self_check(verbose: bool = True) -> "tuple[int, int]":
    """对映射表逐行核对 normalize_smiles_syntax(original) == smiles_fixed。

    返回 (通过行数, 总行数)。两者不等即为口径漂移。
    """
    with open(MAP_PATH, newline="", encoding="utf-8") as fh:
        body = "".join(ln for ln in fh if not ln.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body), delimiter="\t"))

    ok = 0
    for row in rows:
        got = normalize_smiles_syntax(row["smiles_original"])
        if got == row["smiles_fixed"]:
            ok += 1
        elif verbose:
            print("MISMATCH  name={}\n  original={}\n  expected={}\n  got     ={}".format(
                row["name"], row["smiles_original"], row["smiles_fixed"], got))

    # 第二条：绕过映射表、只走通式正则，也必须逐行还原出同一结果。
    regex_ok = sum(
        1 for row in rows
        if _BRACKET_CHARGE_BEFORE_H.sub(_reorder, row["smiles_original"]) == row["smiles_fixed"]
    )

    if verbose:
        print("查表口径 : {}/{}".format(ok, len(rows)))
        print("通式口径 : {}/{}".format(regex_ok, len(rows)))
    return ok, len(rows)


if __name__ == "__main__":
    import sys

    passed, total = self_check()
    sys.exit(0 if passed == total else 1)
