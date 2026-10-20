"""Pick the foot strikes for the phase contact sheets and write their poses.

  python sheet_strikes.py <Manifolds npz> <dataset dir> <out.json> [curves 2,4,6,7]

Per curve: the 12 strikes farthest from the curve's mean phase and the 4 nearest (for
comparison). Each strike carries world joint positions at the strike frame, the left foot's path
over 0.5 s each side, and the clip, time and phase offset for the label.
"""
import json
import os
import sys

import numpy as np

argv = sys.argv
sys.argv = argv[:1]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "write_phase.py")).read()
exec(src.split("\ndef main():")[0])  # strikes()


def world(d):
    parents = [int(x) for x in open(os.path.join(d, "parents.txt")).read().split()]
    names = [l for l in open(os.path.join(d, "skeleton.txt")).read().split("\n")
             if l and not l.startswith(("model ", "prompt ", "frames "))]
    x = np.fromfile(os.path.join(d, "transforms.f32"), dtype=np.float32).reshape(-1, len(parents), 12)
    F, J = x.shape[:2]
    R = x[..., :9].reshape(F, J, 3, 3)
    gr = np.zeros_like(R)
    gp = np.zeros((F, J, 3), np.float32)
    for j, p in enumerate(parents):
        if p < 0:
            gr[:, j], gp[:, j] = R[:, j], x[:, j, 9:]
        else:
            gr[:, j] = gr[:, p] @ R[:, j]
            gp[:, j] = gp[:, p] + np.einsum("fab,fb->fa", gr[:, p], x[:, j, 9:])
    return gp, parents, names


def gap(a, b):
    g = abs(a - b) % 1.0
    return min(g, 1 - g)


m = np.load(argv[1])
phase = m["phase"].reshape(len(m["phase"]), -1)[:, 0]
index = m["index"].reshape(len(m["index"]), -1)[:, 0]
curves = [int(c) for c in (argv[4] if len(argv) > 4 else "2,4,6,7").split(",")]
rows, at, clips = [], 0, {}
for line in open(os.path.join(argv[2], "clips.tsv")):
    _, d, n, fps = line.rstrip("\n").split("\t")
    n, fps = int(n), float(fps)
    clips[d] = fps
    for f in strikes(d, fps):
        rows.append((d, f, fps, int(index[at + f]), float(phase[at + f])))
    at += n

out = []
for c in curves:
    sel = [r for r in rows if r[3] == c]
    a = 2 * np.pi * np.array([r[4] for r in sel])
    mu = (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi)) % 1.0
    sel.sort(key=lambda r: -gap(r[4], mu))
    pick = [(r, "off") for r in sel[:12]] + [(r, "in") for r in sel[-4:]]
    cache = {}
    for (d, f, fps, cc, ph), kind in pick:
        if d not in cache:
            cache[d] = world(d)
        gp, parents, names = cache[d]
        lf = names.index("LeftFoot")
        h = int(round(0.5 * fps))
        path = gp[max(0, f - h):f + h + 1, lf]
        out.append({"curve": c, "kind": kind, "clip": os.path.basename(d), "frame": f, "time": f / fps,
                    "off": gap(ph, mu), "joints": gp[f].tolist(), "parents": parents,
                    "left_foot": lf, "right_foot": names.index("RightFoot"),
                    "foot_path": path.tolist()})
json.dump(out, open(argv[3], "w"))
print("SHEET-STRIKES %d cells, curves %s" % (len(out), curves))
