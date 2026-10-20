"""Write a trained WalkTheDog manifold back next to each clip, and gate it on foot strikes.

  python write_phase.py <Manifold npz> <dataset dir with clips.tsv> [--shift 0.25]

Per clip, phase.f32 holds frames x 4 float32: manifold curve (codebook index), phase 0..1,
amplitude (norm of the latent state), frequency (cycles a second). The gate: left-foot strikes
(the foot comes to rest on the clip's ground, as mm_core.hpp's cycle_phase finds them) must fall
at one learned phase per curve, within 0.06 of a cycle (circular spread). --shift moves every
phase by a quarter cycle at the strikes only, the control that must fail.
"""
import os
import sys

import numpy as np


def strikes(d, fps):
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
    lf, rf = names.index("LeftFoot"), names.index("RightFoot")
    low = np.minimum(gp[:, lf, 1], gp[:, rf, 1])
    ground = np.sort(low)[len(low) // 10]
    v = np.gradient(gp[:, lf], axis=0) * fps
    planted = (gp[:, lf, 1] <= ground + 0.05) & (np.hypot(v[:, 0], v[:, 2]) < 0.5)
    return [f for f in range(1, F) if planted[f] and not planted[f - 1]]


def main():
    npz, ds = sys.argv[1], sys.argv[2]
    shift = 0.25 if "--shift" in sys.argv else 0.0
    m = np.load(npz)
    phase, index, state, freq = m["phase"], m["index"], m["state"], m["frequency"]
    phase = phase.reshape(len(phase), -1)[:, 0]
    freq = freq.reshape(len(freq), -1)[:, 0]
    index = index.reshape(len(index), -1)[:, 0]
    amp = np.linalg.norm(state.reshape(len(state), -1), axis=1)
    at = 0
    per_curve = {}
    write = shift == 0.0
    for line in open(os.path.join(ds, "clips.tsv")):
        _, d, n, fps = line.rstrip("\n").split("\t")
        n, fps = int(n), float(fps)
        sl = slice(at, at + n)
        at += n
        if write:
            np.stack([index[sl], phase[sl], amp[sl], freq[sl]], 1).astype(np.float32).tofile(os.path.join(d, "phase.f32"))
        for f in strikes(d, fps):
            per_curve.setdefault(int(index[sl][f]), []).append((phase[sl][f] + shift * (f % 2)) % 1.0)
    assert at == len(phase), (at, len(phase))
    worst = 0.0
    for c, ps in sorted(per_curve.items()):
        a = 2 * np.pi * np.array(ps)
        r = np.hypot(np.cos(a).mean(), np.sin(a).mean())
        spread = np.sqrt(max(0.0, -2 * np.log(max(r, 1e-9)))) / (2 * np.pi)
        if len(ps) >= 10:
            worst = max(worst, spread)
        print("PHASE-CURVE %d strikes=%d spread=%.3f cycle" % (c, len(ps), spread))
    ok = worst <= 0.06
    print("%s worst=%.3f (curves with 10+ strikes, gate 0.06)%s" % ("PHASE-STRIKES-OK" if ok else "PHASE-STRIKES-FAIL",
          worst, " [control: shifted]" if shift else ""))
    sys.exit(0 if ok else 1)


main()
