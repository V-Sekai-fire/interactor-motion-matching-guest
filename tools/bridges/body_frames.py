# Kimodo clips on a body: SOMA-2603.16858 (SOMA-X) with the ANNY identity and its pose correctives.
# One-time setup for the GPU kernels, and the CPU oracle they are checked against.
#   python body_frames.py <anny-kernels> <SOMA-X assets> <out> --oracle=<key>,<key> <clip dir>...
# Writes, under <out>:
#   tables/*.bin, meta.json   every static table the kernels read (anny-kernels tools/soma/export_soma.py
#                             export_tables), plus "clips" [{key, F, scale}] in the order given
#   clips/<key>/pose.bin      F x 78 x 9 float32: per public joint a row-major 3x3 (root identity)
#   clips/<key>/transl.bin    F x 3 float32: the hips position, metres, floor at y = 0
#   faces.i32                 the mesh triangles
#   oracle/<key>.on.f32       SOMA-X's own vertices (F x V x 3), correctives on, for the --oracle clips
# and prints BODY <key> frames=F scale=s and ORACLE <key> corrective_mm=c.
# Kimodo's 30 joints are world-aligned locals with identity at the SOMA T-pose, SOMA-X's convention,
# so they feed SOMALayer.pose() directly; the other public joints (fingers, ends) stay at the T-pose.
# The hips position is Kimodo's, scaled by the ratio of the two rigs' leg lengths (hips to ankle,
# summed bone lengths down the left leg): both rigs stand on y = 0, so no lift is added.
import json
import os
import sys

import numpy as np  # before torch: SOMA-X segfaults at load the other way round
import torch


def leg_length(names, parents, local_t, foot="LeftFoot", root="Hips"):
    j, n = names.index(foot), 0.0
    while names[j] != root:
        n += float(np.linalg.norm(local_t[j]))
        j = parents[j]
    return n


def main(argv):
    kernels, assets, out = argv[1], argv[2], argv[3]
    oracle = [a.split("=", 1)[1].split(",") for a in argv[4:] if a.startswith("--oracle=")]
    oracle = oracle[0] if oracle else []
    clips = [a for a in argv[4:] if not a.startswith("--")]
    sys.path.insert(0, os.path.join(kernels, "tools", "soma"))
    import export_soma as ex  # noqa: E402  (anny-kernels' own setup: layer, tables, pose)

    torch.set_grad_enabled(False)
    layer = ex.build_layer(assets)
    names = list(layer.public_joint_names)
    J = len(names)
    # One pose() call first, so the skinning state is the one every pose() rebinds to.
    ex.pose(layer, np.tile(np.eye(3), (1, J - 1, 1, 1)), np.zeros((1, 3)), True)
    os.makedirs(out, exist_ok=True)
    meta = ex.export_tables(layer, out)
    par = layer.output_joint_parent_ids.tolist()
    par[0] = -1
    bs = layer.batched_skinning
    soma_t = (bs.source_local_translations[0] if bs.source_local_translations.ndim == 3
              else bs.source_local_translations).cpu().numpy()
    soma_leg = leg_length(names, par, soma_t)
    faces = np.asarray((layer.faces if hasattr(layer, "faces") else layer.faces_tensor).cpu(), np.int32)
    faces.tofile(os.path.join(out, "faces.i32"))
    meta["faces"] = int(len(faces))
    meta["clips"] = []
    os.makedirs(os.path.join(out, "oracle"), exist_ok=True)
    for d in clips:
        key = os.path.basename(d.rstrip("/"))
        kn = [n.strip() for n in open(f"{d}/skeleton.txt").read().split("\n")[3:] if n.strip()]
        kp = [int(x) for x in open(f"{d}/parents.txt").read().split()]
        data = np.fromfile(f"{d}/transforms.f32", np.float32).reshape(-1, len(kn), 12).astype(np.float64)
        F = len(data)
        poses = np.tile(np.eye(3), (F, J, 1, 1))
        for kj, n in enumerate(kn):
            if n not in names:
                continue
            sj = names.index(n)
            if kp[kj] >= 0 and par[sj] != names.index(kn[kp[kj]]):
                continue  # Kimodo's parent differs (finger ends): leave at the T-pose
            poses[:, sj] = data[:, kj, :9].reshape(F, 3, 3)
        scale = soma_leg / leg_length(kn, kp, data[0, :, 9:12])
        transl = data[:, 0, 9:12] * scale
        cd = os.path.join(out, "clips", key)
        os.makedirs(cd, exist_ok=True)
        poses.astype(np.float32).tofile(os.path.join(cd, "pose.bin"))
        transl.astype(np.float32).tofile(os.path.join(cd, "transl.bin"))
        meta["clips"].append({"key": key, "F": F, "scale": scale})
        print(f"BODY {key} frames={F} scale={scale:.4f}", flush=True)
        if key not in oracle:
            continue
        on, worst = [], 0.0
        for s in range(0, F, 60):
            p, t = poses[s:s + 60, 1:], transl[s:s + 60]
            a = ex.pose(layer, p, t, True)["vertices"].numpy()
            b = ex.pose(layer, p, t, False)["vertices"].numpy()
            worst = max(worst, float(np.linalg.norm(a - b, axis=-1).max()))
            on.append(a)
        v = np.concatenate(on).astype(np.float32)
        v.tofile(os.path.join(out, "oracle", f"{key}.on.f32"))
        meta.setdefault("oracle", {})[key] = {"corrective_mm": worst * 1000, "floor_m": float(v[..., 1].min())}
        print(f"ORACLE {key} corrective_mm={worst * 1000:.2f} lowest_vertex_m={v[..., 1].min():.4f}", flush=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"META V={meta['V']} NJ={meta['NJ']} NT={meta['NT']} faces={meta['faces']} clips={len(meta['clips'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
