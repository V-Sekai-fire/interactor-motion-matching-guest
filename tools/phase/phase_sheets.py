"""Contact sheets of foot strikes, rendered with Mitsuba 3 (cuda_ad_rgb) and labelled with Pillow.

  pixi run python phase_sheets.py <strikes.json> <out dir>

One sheet per manifold curve: 12 strikes farthest from the curve's phase (red label) and the 4
nearest (green label). Each cell shows the skeleton at the strike frame, the left foot (red) and
right foot (blue), and the left foot's path over 0.5 s each side (orange), seen from the side of
the travel direction.
"""
import json
import math
import os
import sys

import mitsuba as mi
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mi.set_variant("cuda_ad_rgb")
cells = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
W = H = 300


def diffuse(rgb):
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": rgb}}


def render(c):
    J = np.array(c["joints"])
    P = np.array(c["foot_path"])
    ground = min(P[:, 1].min(), J[:, 1].min())
    hips = J[0]
    d = P[-1] - P[0]
    d[1] = 0
    if np.linalg.norm(d) < 0.2:
        d = np.array([0.0, 0.0, 1.0])
    d /= np.linalg.norm(d)
    side = np.array([d[2], 0.0, -d[0]])
    target = np.array([hips[0], ground + 0.85, hips[2]])
    eye = target + side * 3.6 + d * 0.6 + np.array([0, 0.5, 0])
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {"type": "perspective", "fov": 40,
                   "to_world": mi.ScalarTransform4f().look_at(origin=eye.tolist(), target=target.tolist(), up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": W, "height": H, "pixel_format": "rgb"},
                   "sampler": {"type": "independent", "sample_count": 64}},
        "sky": {"type": "constant", "radiance": {"type": "rgb", "value": 0.6}},
        "sun": {"type": "directional", "direction": [-0.4, -1.0, -0.3], "irradiance": {"type": "rgb", "value": 2.5}},
        "floor": {"type": "rectangle", "bsdf": diffuse([0.75, 0.75, 0.72]),
                  "to_world": mi.ScalarTransform4f().translate([target[0], ground, target[2]]).rotate([1, 0, 0], -90).scale(4)},
    }
    bone = diffuse([0.35, 0.35, 0.38])
    for j, p in enumerate(c["parents"]):
        if p >= 0 and np.linalg.norm(J[j] - J[p]) > 1e-3:
            scene["b%d" % j] = {"type": "cylinder", "p0": J[p].tolist(), "p1": J[j].tolist(), "radius": 0.022, "bsdf": bone}
        r, col = 0.03, [0.35, 0.35, 0.38]
        if j == c["left_foot"]:
            r, col = 0.06, [0.85, 0.1, 0.1]
        elif j == c["right_foot"]:
            r, col = 0.06, [0.1, 0.3, 0.85]
        scene["j%d" % j] = {"type": "sphere", "center": J[j].tolist(), "radius": r, "bsdf": diffuse(col)}
    for i, q in enumerate(P[:: max(1, len(P) // 30)]):
        scene["p%d" % i] = {"type": "sphere", "center": q.tolist(), "radius": 0.015, "bsdf": diffuse([0.95, 0.55, 0.1])}
    img = mi.render(mi.load_dict(scene))
    a = np.clip(np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)), 0, 255)
    return Image.fromarray(a.astype(np.uint8))


try:
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
except OSError:
    font = title = ImageFont.load_default()

for curve in sorted({c["curve"] for c in cells}):
    cs = [c for c in cells if c["curve"] == curve]
    cols, lab, top = 4, 40, 36
    rows = math.ceil(len(cs) / cols)
    sheet = Image.new("RGB", (cols * W, top + rows * (H + lab)), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((8, 6), "Curve %d: 12 strikes farthest from the curve's phase, then the 4 nearest" % curve,
            fill="black", font=title)
    for i, c in enumerate(cs):
        x, y = (i % cols) * W, top + (i // cols) * (H + lab)
        sheet.paste(render(c), (x, y))
        col = (180, 20, 20) if c["kind"] == "off" else (20, 130, 40)
        dr.text((x + 6, y + H + 2), "%s  t=%.2f s" % (c["clip"], c["time"]), fill="black", font=font)
        dr.text((x + 6, y + H + 20), "off phase by %.2f of a stride" % c["off"], fill=col, font=font)
    p = os.path.join(out, "curve%d.png" % curve)
    sheet.save(p)
    print("SHEET", p, sheet.size)
