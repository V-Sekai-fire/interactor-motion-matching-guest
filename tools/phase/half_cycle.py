"""Diagnostic: how many off-phase foot strikes sit half a cycle off (a left/right swap).
  python half_cycle.py <Manifolds npz> <dataset dir>"""
import os
import sys

import numpy as np

argv = sys.argv
sys.argv = argv[:1]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "write_phase.py")).read()
exec(src.split("\ndef main():")[0])

m = np.load(argv[1])
phase = m["phase"].reshape(len(m["phase"]), -1)[:, 0]
index = m["index"].reshape(len(m["index"]), -1)[:, 0]
rows, at = [], 0
for line in open(os.path.join(argv[2], "clips.tsv")):
    _, d, n, fps = line.rstrip("\n").split("\t")
    n, fps = int(n), float(fps)
    for f in strikes(d, fps):
        rows.append((int(index[at + f]), float(phase[at + f])))
    at += n


def gap(a, b):
    g = abs(a - b) % 1.0
    return min(g, 1 - g)


tot = {"in": 0, "half": 0, "other": 0}
for c in sorted({r[0] for r in rows}):
    ps = [p for cc, p in rows if cc == c]
    a = 2 * np.pi * np.array(ps)
    mu = (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi)) % 1.0
    k = {"in": 0, "half": 0, "other": 0}
    for p in ps:
        g = gap(p, mu)
        k["in" if g <= 0.06 else "half" if abs(g - 0.5) <= 0.06 else "other"] += 1
    for x in k:
        tot[x] += k[x]
    print("CURVE %d strikes=%d in_phase=%d half_cycle_off=%d other_off=%d" % (c, len(ps), k["in"], k["half"], k["other"]))
print("ALL in_phase=%d half_cycle_off=%d other_off=%d" % (tot["in"], tot["half"], tot["other"]))
