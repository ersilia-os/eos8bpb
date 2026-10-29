#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mechgate.cli -- command line entry point. SMILES in, 20 columns out.

The algorithms and the scoring configuration live in mechgate/core.py; this file
only handles argument parsing and output formatting.
"""
from __future__ import annotations
import io, sys, csv, json, argparse

from .core import MechGate, COLUMNS, DEFAULT_TIMEOUT, TIMEOUT_MARK


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mechgate", description="Score SMILES: 20 columns, one row per molecule")
    ap.add_argument("smiles", nargs="*", help="one or more SMILES")
    ap.add_argument("--in", dest="infile", help="text file with one SMILES per line")
    ap.add_argument("--out", help="output TSV path (default: stdout)")
    ap.add_argument("--set", dest="setname", default="user_input",
                    help="value for the 'set' column")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of TSV")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="per-molecule timeout in seconds (default: %(default)s). "
                         "A molecule that times out is recorded as not computable "
                         "and scoring continues. 0 means no limit")
    a = ap.parse_args(argv)

    smis = list(a.smiles)
    if a.infile:
        smis += [l.strip() for l in io.open(a.infile, encoding="utf-8") if l.strip()]
    if not smis:
        ap.error("no SMILES given")

    timeout = None if not a.timeout else a.timeout
    with MechGate(timeout=timeout) as g:
        rows = []
        for s in smis:
            r = g.row(s, a.setname)
            if r["missing_descriptors"] == TIMEOUT_MARK:
                sys.stderr.write(
                    "timed out after %gs, recorded as not computable, continuing: %s\n"
                    % (timeout, s))
            rows.append(r)

    if a.json:
        out = json.dumps(rows, ensure_ascii=False, indent=1)
        (io.open(a.out, "w", encoding="utf-8") if a.out else sys.stdout).write(out + "\n")
        return 0
    fh = io.open(a.out, "w", encoding="utf-8", newline="") if a.out else sys.stdout
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([r.get(c, "") for c in COLUMNS])
    if a.out:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
