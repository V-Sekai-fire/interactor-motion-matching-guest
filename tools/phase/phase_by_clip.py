"""Which clips put their foot strikes on which learned curve, and how far each clip's strikes sit
from its curve's mean phase.  python phase_by_clip.py <Manifolds npz> <dataset dir>"""
import os
import sys

import numpy as np

sys.argv, argv = sys.argv[:1], sys.argv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "write_phase.py")).read()
exec(src.split("\ndef main():")[0])  # strikes()

m = np.load(argv[1])
phase = m["phase"].reshape(len(m["phase"]), -1)[:, 0]
index = m["index"].reshape(len(m["index"]), -1)[:, 0]
rows, at = [], 0
for line in open(os.path.join(argv[2], "clips.tsv")):
    _, d, n, fps = line.rstrip("\n").split("\t")
    n, fps = int(n), float(fps)
    ph, ix = phase[at:at + n], index[at:at + n]
    at += n
    for f in strikes(d, fps):
        rows.append((os.path.basename(d), int(ix[f]), float(ph[f])))


def circ_mean(ps):
    a = 2 * np.pi * np.array(ps)
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi)) % 1.0


def gap(a, b):
    g = abs(a - b) % 1.0
    return min(g, 1 - g)


curves = sorted({c for _, c, _ in rows})
for c in curves:
    ps = [p for _, cc, p in rows if cc == c]
    mu = circ_mean(ps)
    by = {}
    for k, cc, p in rows:
        if cc == c:
            by.setdefault(k, []).append(gap(p, mu))
    print("CURVE %d strikes=%d" % (c, len(ps)))
    for k, gs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print("  %-24s strikes=%4d median_off=%.3f p90_off=%.3f" % (k, len(gs), np.median(gs), np.percentile(gs, 90)))
