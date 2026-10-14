from types import SimpleNamespace as NS

import numpy as np

from b2crig.export.gltf import _gate_layer_labels


def test_far_garment_splats_rebind_to_body():
    tri = np.array([[0, 1, 2]])
    body = NS(classes=[0, 15], verts=np.zeros((3, 3)), faces=tri)
    upper = NS(classes=[23], verts=np.array([[0, 1, 0], [0.1, 1, 0], [0, 1.1, 0]], float), faces=tri)   # a torso patch
    pos = np.array([[0, 1.02, 0], [0, -0.8, 0], [0, -0.7, 0.01], [0, -0.75, 0]], float)
    f = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "seg_label": np.array([23, 23, 15, 23], np.float32),
         "seg_conf": np.ones(4, np.float32)}
    out = _gate_layer_labels(f, [body, upper], 0.5)
    assert list(out["seg_label"]) == [23, 15, 15, 15]   # the leg splats leave the torso layer; the near one stays


def test_trousers_labelled_upper_become_lower():
    from b2crig.rig.cage import reconcile_garment_labels
    body_V = np.array([[0, 1, 0], [0, -1, 0]], float)             # a torso vertex and a thigh vertex
    region = np.array(["spine2", "l_hip"])
    xyz = np.array([[0, 1, 0.03], [0, -1, 0.03], [0, -1, 0.04], [0, -1, 0.05]], float)
    lab = np.array([23, 23, 1, 13], np.float32)
    out, n = reconcile_garment_labels(xyz, lab, body_V, region)
    assert list(out) == [23, 13, 13, 13] and n == 2
