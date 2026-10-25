import os, sys
import numpy as np  # noqa: F401  (numpy before torch, or SOMA-X segfaults at load)
import torch
sys.path.insert(0, os.path.dirname(__file__))
import body_frames  # noqa: F401
from soma import SOMALayer
A = sys.argv[1]
L = SOMALayer(data_root=A, identity_model_type="anny", device="cpu", correctives_model_path=os.path.join(A, "correctives_model.pt"))
P = L.procedural_transforms
print("modes", list(P.rotation_extraction_mode_names), flush=True)
M = P.rotation_parameter_matrices_by_mode
for i, m in enumerate(P.rotation_extraction_mode_names):
    nz = (M[i].abs() > 0)
    print(f"  {m}: nonzero weights {int(nz.sum())}, twist joints driven {int(nz.any(1).sum())} of {M[i].shape[0]}, max |w| {float(M[i].abs().max()):.4g}", flush=True)
print("single_twist_axis", P._single_twist_axis, flush=True)
for a in ("twist_axis_ids", "source_twist_axis_ids", "twist_axis_signs"):
    if hasattr(P, a):
        print(a, sorted(set(getattr(P, a).tolist())), flush=True)
labels = list(L.identity_model.identity_model.phenotype_labels)
print("labels", labels, flush=True)
for extra in ({"gender": 0.0}, {"gender": 1.0}, {"gender": 0.0, "height": 1.0}, {"gender": 0.0, "weight": 1.0}, {"gender": 1.0, "age": 0.0}, {"gender": 0.0, "proportions": 1.0}):
    co = torch.full((1, len(labels)), 0.5)
    for k, v in extra.items():
        if k in labels:
            co[0, labels.index(k)] = v
    L.prepare_identity(co)
    bs = L._cached_scale_params
    gs = L._cached_global_scale
    print("identity", extra, "bone scales:", None if bs is None else tuple(bs.shape), "global scale", float(gs) if torch.is_tensor(gs) else gs, "lod_transfer", L.identity_lod_transfer is not None, flush=True)
