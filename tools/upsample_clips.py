#!/usr/bin/env python3
"""Upsample the Kimodo clips of data/kimodo-locomotion from 30 fps to 60 fps, the rate of data/o3de-motionmatching.

  uv run --with numpy tools/upsample_clips.py data/kimodo-locomotion [--fps=60] [--check]

Each clip dir holds transforms.f32 (frames x 30 joints x 12 float32: a row-major 3x3 local rotation, then the local offset) and
skeleton.txt (3 header lines, the third "frames N fps F joints J ..."). The tool rewrites both and writes upsampled.txt with the
source rate. It refuses a clip already at the target rate, so a second run changes nothing.

Method: modified Akima (makima, as Godot's Animation::makima_interpolate_in_time) per channel. Rotations go through quaternions
(sign-aligned to one hemisphere, makima per component, renormalised) and come back as matrices. The source frames are kept
exactly (every second output frame). A one-shot gets 2N-1 frames. A loop (loop.txt present: a steady gait cut from one foot
strike to the next) gets 2N frames so its period stays N/30 s: the extra frame sits between the last frame and the wrap and
continues the motion along the spline's end tangent. It is not interpolated toward the first frame, because the host carries the root
on at the wrap (a turning gait wraps with a turn and a shift of one pass, so the midpoint of last and first is a wrong pose).

--check verifies without writing: every source frame survives, rotations stay orthonormal, and the clip duration is unchanged.
"""

import pathlib
import sys

import numpy as np


def makima_tangent(p0, p1, p2, p3):
    """Port of Godot's Animation::makima_tangent: slopes p0..p3 around one knot."""
    w1 = np.abs(p3 - p2) + np.abs(p3 + p2) * 0.5
    w2 = np.abs(p1 - p0) + np.abs(p1 + p0) * 0.5
    den = w1 + w2
    return np.where(den == 0, (p1 + p2) * 0.5, (w1 * p1 + w2 * p2) / np.where(den == 0, 1, den))


def makima(y, tau):
    """Modified Akima through y (F, C) at unit spacing, evaluated at knot coordinates tau (T,) in [0, F-1].

    The missing slopes at both ends come from akima_extrapolate (2*edge - next)."""
    F = y.shape[0]
    m = np.diff(y, axis=0)  # slope of interval k, (F-1, C)
    m = np.concatenate([[2 * m[0] - m[1]], m, [2 * m[-1] - m[-2]]])
    m = np.concatenate([[2 * m[0] - m[1]], m, [2 * m[-1] - m[-2]]])  # m[k+2] is the slope of interval k
    i = np.minimum(np.floor(tau + 1e-9).astype(int), F - 2)
    w = (tau - i)[:, None]
    t_from = makima_tangent(m[i], m[i + 1], m[i + 2], m[i + 3])
    t_to = makima_tangent(m[i + 1], m[i + 2], m[i + 3], m[i + 4])
    s2, s3 = w * w, w * w * w
    return (2 * s3 - 3 * s2 + 1) * y[i] + (s3 - 2 * s2 + w) * t_from + (-2 * s3 + 3 * s2) * y[i + 1] + (s3 - s2) * t_to


def mat_to_quat(R):
    """(..., 3, 3) -> wxyz (..., 4), w >= 0."""
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    out = np.empty(R.shape[:-2] + (4,))
    for idx in np.ndindex(*R.shape[:-2]):
        m = R[idx]
        if t[idx] > 0:
            s = 2 * np.sqrt(1 + t[idx])
            q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] >= m[2, 2]:
            s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
        q = np.array(q)
        out[idx] = q if q[0] >= 0 else -q
    return out


def quat_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1).reshape(q.shape[:-1] + (3, 3))


def hemisphere(q):
    """Sign-align a (F, J, 4) quaternion track so its components are continuous."""
    q = q.copy()
    for k in range(1, q.shape[0]):
        flip = (q[k] * q[k - 1]).sum(-1) < 0
        q[k][flip] *= -1
    return q


def upsample(xf, loop):
    """xf (N, J, 12) -> (M, J, 12) at twice the rate. Rotations by quaternion makima, offsets by makima.

    A loop gets one extra knot (linear extrapolation) so the half step after its last frame continues the motion."""
    N, J, _ = xf.shape
    M = 2 * N if loop else 2 * N - 1
    tau = np.arange(M) / 2.0
    rot = hemisphere(mat_to_quat(xf[:, :, :9].reshape(N, J, 3, 3))).reshape(N, J * 4)
    off = xf[:, :, 9:].reshape(N, J * 3)
    if loop:
        rot, off = (np.concatenate([a, 2 * a[-1:] - a[-2:-1]]) for a in (rot, off))
    q_up = makima(rot, tau).reshape(M, J, 4)
    out = np.empty((M, J, 12), np.float32)
    out[:, :, :9] = quat_to_mat(q_up).reshape(M, J, 9)
    out[:, :, 9:] = makima(off, tau).reshape(M, J, 3)
    out[0 : 2 * N : 2] = xf[: len(out[0 : 2 * N : 2])]  # the source frames stay exactly
    return out


def read_header(path):
    lines = path.read_text().splitlines(True)
    toks = lines[2].split()
    return lines, toks


def process(clip: pathlib.Path, target: float, check: bool):
    xf_path, sk_path = clip / "transforms.f32", clip / "skeleton.txt"
    if not (xf_path.exists() and sk_path.exists()):
        return None
    lines, toks = read_header(sk_path)
    frames, fps, joints = int(toks[toks.index("frames") + 1]), float(toks[toks.index("fps") + 1]), int(toks[toks.index("joints") + 1])
    if abs(fps - target) < 1e-6:
        return f"{clip.name}: already {fps:g} fps, skipped"
    if abs(target / fps - 2.0) > 1e-9:
        raise SystemExit(f"{clip.name}: only a 2x rate change is supported ({fps:g} -> {target:g})")
    xf = np.fromfile(xf_path, "<f4").reshape(frames, joints, 12).astype(np.float64)
    loop = (clip / "loop.txt").exists()
    up = upsample(xf, loop)
    R = up[:, :, :9].reshape(-1, 3, 3)
    ortho = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    kept = np.abs(up[0 : 2 * frames : 2][:frames] - xf.astype(np.float32)).max()
    dur_old, dur_new = frames / fps, up.shape[0] / target if loop else (up.shape[0] - 1) / target
    dur_ref = frames / fps if loop else (frames - 1) / fps
    ok = ortho < 1e-5 and kept < 1e-6 and abs(dur_new - dur_ref) < 1e-6
    msg = f"{clip.name}: {frames} -> {up.shape[0]} frames ({'loop' if loop else 'one-shot'}), ortho {ortho:.1e}, kept {kept:.1e}, duration {dur_ref:.3f} s -> {dur_new:.3f} s {'ok' if ok else 'FAIL'}"
    if not ok:
        raise SystemExit(msg)
    if not check:
        up.astype("<f4").tofile(xf_path)
        toks[toks.index("frames") + 1] = str(up.shape[0])
        toks[toks.index("fps") + 1] = f"{target:.1f}"
        lines[2] = " ".join(toks) + "\n"
        sk_path.write_text("".join(lines))
        (clip / "upsampled.txt").write_text(f"from {fps:g} fps, {frames} frames; makima (Godot), rotations by quaternion; tools/upsample_clips.py\n")
    return msg


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--fps=")), 60.0)
    check = "--check" in sys.argv
    n = 0
    for clip in sorted(p for p in pathlib.Path(args[0]).iterdir() if p.is_dir()):
        msg = process(clip, target, check)
        if msg:
            print(msg)
            n += 1
    print(f"{n} clips {'checked' if check else 'processed'}")


if __name__ == "__main__":
    main()
