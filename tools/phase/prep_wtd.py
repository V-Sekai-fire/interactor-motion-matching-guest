"""Our motion set as a WalkTheDog dataset: one sequence per clip, in the root's floor frame.

  python prep_wtd.py <clip set dir>[:<dir>...] <out dir> [--fps 60]

Only clips recorded at --fps (default 60, the O3DE motion capture) are written: no frame is
made up by resampling, so the 30 fps Kimodo clips are left out and keep foot-strike phase.

Channels per frame: Velocities (30 joints x 3, m/s), Positions (30 x 3, m), Rotations (30 x 6,
the first two columns of each joint's rotation). The root is the hips on the floor facing the
hips' forward. Writes Data.bin, Description.txt, Sequences.txt and clips.tsv (sequence id, clip
dir, frames) so the learned phase can be written back next to each clip.
"""
import os
import sys

import numpy as np


def clip(d):
    sk = [l for l in open(os.path.join(d, "skeleton.txt")).read().split("\n") if l]
    fps = 30.0
    for l in sk:
        if l.startswith("frames "):
            fps = float(l.split()[3])
    parents = [int(x) for x in open(os.path.join(d, "parents.txt")).read().split()]
    x = np.fromfile(os.path.join(d, "transforms.f32"), dtype=np.float32).reshape(-1, len(parents), 12)
    return x, parents, fps


def features(x, parents, fps):
    F, J = x.shape[:2]
    R = x[..., :9].reshape(F, J, 3, 3)
    t = x[..., 9:]
    gr = np.zeros_like(R)
    gp = np.zeros((F, J, 3), np.float32)
    for j, p in enumerate(parents):
        if p < 0:
            gr[:, j], gp[:, j] = R[:, j], t[:, j]
        else:
            gr[:, j] = gr[:, p] @ R[:, j]
            gp[:, j] = gp[:, p] + np.einsum("fab,fb->fa", gr[:, p], t[:, j])
    root = [j for j, p in enumerate(parents) if p < 0][0]
    fwd = gr[:, root] @ np.array([0, 0, 1], np.float32)
    yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Ry = np.zeros((F, 3, 3), np.float32)
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 1, 1], Ry[:, 2, 0], Ry[:, 2, 2] = c, s, 1, -s, c
    o = gp[:, root].copy()
    o[:, 1] = 0
    pos = np.einsum("fab,fjb->fja", Ry, gp - o[:, None])
    vel = np.gradient(gp, axis=0) * fps
    vel = np.einsum("fab,fjb->fja", Ry, vel)
    rot = np.einsum("fab,fjbc->fjac", Ry, gr)[..., :, :2].reshape(F, J, 6)
    return np.concatenate([vel.reshape(F, -1), pos.reshape(F, -1), rot.reshape(F, -1)], 1).astype(np.float32), J


def main():
    roots, out = sys.argv[1].split(":"), sys.argv[2]
    fps_out = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 60.0
    os.makedirs(out, exist_ok=True)
    rows, seq, listing = [], [], []
    J = 0
    fps_all = set()
    for r in roots:
        for k in sorted(os.listdir(r)):
            d = os.path.join(r, k)
            if not os.path.isfile(os.path.join(d, "transforms.f32")):
                continue
            x, parents, fps = clip(d)
            if abs(fps - fps_out) > 1e-3:
                print("WTD-SKIP %s at %g fps" % (k, fps))
                continue
            f, J = features(x, parents, fps)
            fps_all.add(fps)

            rows.append(f)
            seq += [len(listing) + 1] * len(f)
            listing.append((len(listing) + 1, d, len(f), fps))
    assert len(fps_all) == 1, fps_all
    data = np.concatenate(rows)
    data.tofile(os.path.join(out, "Data.bin"))
    with open(os.path.join(out, "Sequences.txt"), "w") as fh:
        fh.write("\n".join(str(s) for s in seq) + "\n")
    with open(os.path.join(out, "Description.txt"), "w") as fh:
        fh.write("Velocities,Positions,Rotations\n%d,%d,%d\n%d\n0\n0\n%d\n" % (J * 3, J * 3, J * 6, len(data), int(fps_out)))
    with open(os.path.join(out, "clips.tsv"), "w") as fh:
        for i, d, n, fps in listing:
            fh.write("%d\t%s\t%d\t%g\n" % (i, d, n, fps))
    print("WTD-DATA frames=%d dims=%d clips=%d fps=%g" % (data.shape[0], data.shape[1], len(listing), fps_out))


main()
