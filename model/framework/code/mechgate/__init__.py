# -*- coding: utf-8 -*-
"""MechGate -- five parallel measures of Gram-negative penetration, efflux and activity.

SMILES in, 20 columns out, one row per molecule. Column order follows Table 2 of
article_v3.docx. No total score is formed and nothing is ranked: none of the
columns is a summary of the others.

    >>> import mechgate
    >>> row = mechgate.score("O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O")
    >>> row["activity_percentile"]
    '0.9996445854'
    >>> [row[c] for c in mechgate.COLUMNS][:3]
    ['user_input', 'MYSWGUAQZAJSOK-UHFFFAOYSA-N', 'O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O']

Use ``score_many`` for several molecules, or ``MechGate`` to reuse one set of
loaded models and control its lifetime:

    >>> with mechgate.MechGate(timeout=60) as g:          # doctest: +SKIP
    ...     rows = [g.row(s) for s in smiles_list]

Released under GPL-2.0-only (see LICENSE); its runtime dependency Open Babel is
likewise GPL-2.0.
"""
from .core import (                      # noqa: F401
    MechGate,
    score,
    score_many,
    COLUMNS,
    DEFAULT_TIMEOUT,
    TIMEOUT_MARK,
    NOT_COMPUTABLE,
    resource,
)

__version__ = "0.2.0"

__all__ = ["MechGate", "score", "score_many", "COLUMNS", "DEFAULT_TIMEOUT",
           "TIMEOUT_MARK", "NOT_COMPUTABLE", "resource", "__version__"]
