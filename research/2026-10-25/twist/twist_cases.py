# Every twist case SOMA-X's aligned mode meets on the bridge clips: SOMA-X's angle (4·atan2 of the
# half-angle quaternion), the standard swing-twist angle (2·atan2(q_x, q_w), w >= 0), and the
# Kabsch matrix split (numpy port of SomaCommon.swingTwistAngle). Prints where they disagree and why.
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import body_frames as bf  # noqa: E402  (installs the anny forward patch)
from soma import SOMALayer  # noqa: E402
import soma.procedural_transforms as pt  # noqa: E402
from soma.geometry.transforms import quaternion_multiply_xyzw, quaternion_conjugate_xyzw, quaternion_normalize_xyzw  # noqa: E402

CAP = []
_orig = pt.SOMAProceduralTransforms._aligned_twist_channels_from_world if hasattr(pt, "SOMAProceduralTransforms") else None


def find_cls():
    for n in dir(pt):
        c = getattr(pt, n)
        if isinstance(c, type) and hasattr(c, "_aligned_twist_channels_from_world"):
            return c


C = find_cls()
_orig = C._aligned_twist_channels_from_world


def spy(self, W):
    dev, dt = W.device, W.dtype
    vq = pt._aligned_virtual_quaternions(W[..., :3, :3], self.source_bind_quaternions.to(dt), self.segment_bind_align_quaternions.to(dt),
                                         self.aligned_virtual_segment_ids, self.aligned_virtual_joint_ids)
    n = self.segment_source_ids.numel()
    qe, qs, qp = vq[:, :n], vq[:, n:2 * n], vq[:, 2 * n:]
    loc = quaternion_normalize_xyzw(quaternion_multiply_xyzw(quaternion_conjugate_xyzw(qs), qe))
    inh = quaternion_normalize_xyzw(quaternion_multiply_xyzw(quaternion_conjugate_xyzw(qp), qs))
    CAP.append((loc.detach().cpu().numpy().astype(np.float64), inh.detach().cpu().numpy().astype(np.float64),
                self.segment_reverse_mask.cpu().numpy()))
    return _orig(self, W)


C._aligned_twist_channels_from_world = spy


def qmat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def soma_angle(q):
    q = np.where(q[..., 3:] < 0, -q, q)
    h = np.concatenate([q[..., :3], q[..., 3:] + 1], -1)
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    return 4 * np.arctan2(h[..., 0], h[..., 3])


def std_angle(q):
    q = np.where(q[..., 3:] < 0, -q, q)
    return 2 * np.arctan2(q[..., 0], q[..., 3])


def kabsch_angle(R):
    out = np.empty(R.shape[:-2])
    v0 = np.array([1.0, 0, 0])
    ref = np.array([0, 1.0, 0])
    v1 = np.cross(v0, ref); v1 /= np.linalg.norm(v1)
    v2 = np.cross(v0, v1); v2 /= np.linalg.norm(v2)
    V = np.stack([v0, v1, v2], 1)
    for idx in np.ndindex(out.shape):
        r = R[idx]
        u0 = r @ v0; u0 /= np.linalg.norm(u0)
        c = np.cross(v0, u0); s = np.linalg.norm(c); cs = v0 @ u0
        if s < 1e-9:
            R0 = np.eye(3) if cs > 0 else None
            if R0 is None:  # antipodal: any perpendicular
                p = np.array([0, 0, 1.0]) if abs(u0[2]) < 0.9 else np.array([0, 1.0, 0])
                c1 = np.cross(u0, p); c1 /= np.linalg.norm(c1); c2 = np.cross(u0, c1)
                R0 = np.stack([u0, c1, c2], 1) @ V.T
        else:
            k = c / s
            rod = lambda w: cs * w + s * np.cross(k, w) + (k @ w) * (1 - cs) * k  # noqa: E731
            R0 = np.stack([u0, rod(v1), rod(v2)], 1) @ V.T
        T = R0.T @ r
        out[idx] = np.arctan2(T[2, 1] - T[1, 2], T[1, 1] + T[2, 2])
    return out


def swing_deg(R):
    return np.degrees(np.arccos(np.clip(R[..., 0, 0], -1, 1)))


def main(argv):
    assets, clips = argv[1], argv[2:]
    layer = SOMALayer(data_root=assets, identity_model_type="anny", device="cpu",
                      correctives_model_path=os.path.join(assets, "correctives_model.pt"))
    labels = list(layer.identity_model.identity_model.phenotype_labels)
    co = torch.full((1, len(labels)), 0.5); co[0, labels.index("gender")] = 0.0
    layer.prepare_identity(co)
    names = list(layer.public_joint_names); par = layer.output_joint_parent_ids.tolist(); par[0] = -1
    J = len(names)
    ptr = layer.procedural_transforms
    seg_names = [names[i] if i < J else str(i) for i in ptr.segment_end_source_ids.tolist()]
    rows = []
    for d in clips:
        key = os.path.basename(d.rstrip("/"))
        kn = [n.strip() for n in open(f"{d}/skeleton.txt").read().split("\n")[3:] if n.strip()]
        kp = [int(x) for x in open(f"{d}/parents.txt").read().split()]
        data = np.fromfile(f"{d}/transforms.f32", np.float32).reshape(-1, len(kn), 12).astype(np.float64)
        F = len(data)
        poses = np.tile(np.eye(3), (F, J - 1, 1, 1))
        for kj, n in enumerate(kn):
            if n in names:
                sj = names.index(n)
                if kp[kj] < 0 or par[sj] == names.index(kn[kp[kj]]):
                    poses[:, sj - 1] = data[:, kj, :9].reshape(F, 3, 3)
        CAP.clear()
        with torch.no_grad():
            layer.pose(torch.tensor(poses, dtype=torch.float32), transl=torch.zeros(F, 3), pose2rot=False, apply_correctives=False)
        loc, inh, rev = CAP[-1]
        for kind, Q in (("local", loc), ("inherited", inh)):
            R = qmat(Q)
            a, b, c = soma_angle(Q), std_angle(Q), kabsch_angle(R)
            sw = swing_deg(R)
            for s in range(Q.shape[1]):
                if kind == "inherited" and not rev[s]:
                    continue
                jumps = lambda x: float(np.abs(np.diff(x[:, s])).max())  # noqa: E731
                rows.append(dict(clip=key, kind=kind, seg=seg_names[s],
                                 kabsch_vs_std=float(np.abs(c[:, s] - b[:, s]).max()),
                                 soma_vs_std_deg=float(np.degrees(np.abs(a[:, s] - b[:, s]).max())),
                                 at_frame=int(np.abs(a[:, s] - b[:, s]).argmax()),
                                 swing_max_deg=float(sw[:, s].max()),
                                 twist_max_deg=float(np.degrees(np.abs(b[:, s]).max())),
                                 jump_soma_deg=float(np.degrees(jumps(a))), jump_std_deg=float(np.degrees(jumps(b))),
                                 jump_kabsch_deg=float(np.degrees(jumps(c)))))
    json.dump(rows, open(os.environ.get("OUT", "/tmp/twist_cases.json"), "w"), indent=1)
    print(len(rows), "segment-clip cases")


if __name__ == "__main__":
    main(sys.argv)
