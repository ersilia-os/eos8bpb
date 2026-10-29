"""Thin wrapper around the unmodified MechGate package vendored in ``mechgate/``.

The package is copied verbatim from https://github.com/qinlang0008-blip/mechgate
(commit 06b154113e562125e58c6da9ed12796ccc2b88cf, GPL-2.0-only). Only two things differ
from running it directly. Its weights and percentile ruler are read from
``model/checkpoints/`` (fetched with eosvc) instead of from the package directory, and
the per-molecule timeout is switched off, which the authors expose as an option
(``timeout=None``), so that machine load cannot blank a row.

SMILES are passed to MechGate exactly as given. Its 3D descriptors come from a seeded
conformer search, so the output is deterministic for a given string but can differ
between two strings that describe the same molecule.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.abspath(os.path.join(ROOT, "..", "..", "checkpoints"))

sys.path.insert(0, ROOT)

import mechgate.core as core  # noqa: E402


def _resource(*parts):
    return os.path.join(CHECKPOINTS, *parts)


# models/ and rulers/ live in model/checkpoints/ rather than inside the package
core.resource = _resource

# The authors' 20 columns minus identifiers, the two uptake flags and the diagnostics
# column. Categorical calls are returned as 1 (high or selected) or 0.
COLUMNS = [
    "ompf_score",
    "ompf_percentile",
    "ompf_cut",
    "multiporin_score",
    "multiporin_percentile",
    "multiporin_cut",
    "lipid_score",
    "lipid_percentile",
    "lipid_cut",
    "efflux_score",
    "efflux_percentile",
    "activity_score",
    "activity_percentile",
    "activity_call",
]
CALLS = {
    "ompf_cut": {core.HIGH: 1, core.LOW: 0},
    "multiporin_cut": {core.HIGH: 1, core.LOW: 0},
    "lipid_cut": {core.HIGH: 1, core.LOW: 0},
    "activity_call": {core.SELECTED: 1, core.NOT_SELECTED: 0},
}


def _value(row, column):
    raw = row.get(column, "")
    if column in CALLS:
        return CALLS[column].get(raw, np.nan)
    try:
        return float(raw)
    except ValueError:
        return np.nan


def my_model(smiles_list):
    outputs = []
    with core.MechGate(timeout=None) as gate:
        for smiles in smiles_list:
            try:
                row = gate.row(smiles)
            except Exception:
                row = {}
            outputs.append([_value(row, c) for c in COLUMNS])
    return np.array(outputs, dtype=np.float64)
