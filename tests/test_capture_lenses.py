"""A capture with one camera per lens (b2crunner 24be811: the orbit extension's telephoto frames each carry their own):
every capture image keeps its own lens, views of our own making take the orbit's."""
import json

import numpy as np

from b2crig import cameras as C
from b2crig import visibility as VI

ORBIT = (1080, 1920, 1776.0, 1776.0, 540.0, 960.0)
TELE = (1080, 1920, 4617.4, 4617.4, 540.0, 960.0)


def _capture(tmp_path):
    # camera 1 is the telephoto lens (b2crunner numbers lenses in image order, so the first frame's comes first)
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n"
        f"1 PINHOLE {' '.join(map(str, TELE))}\n"
        f"2 PINHOLE {' '.join(map(str, ORBIT))}\n")
    rows = []
    for i, cid in enumerate([1, 2, 2, 2], start=1):
        rows += [f"{i} 1 0 0 0 0 0 {2.0 + i} {cid} frame_{i:05d}_.png", ""]
    (tmp_path / "images.txt").write_text("# Image list\n" + "\n".join(rows) + "\n")
    return tmp_path


def test_each_image_keeps_its_own_lens(tmp_path):
    ds = _capture(tmp_path)
    assert VI.colmap_view_intrinsics(ds) == [TELE, ORBIT, ORBIT, ORBIT]
    lenses, images = VI.colmap_lenses(ds)
    assert [cid for cid, *_ in images] == [1, 2, 2, 2]
    assert [name for *_, name in images][0] == "frame_00001_.png"


def test_the_orbit_lens_is_the_one_most_images_share_not_the_first(tmp_path):
    intr, cams = VI.colmap_cams(_capture(tmp_path))
    assert intr == ORBIT
    assert len(cams) == 4


def test_cameras_json_carries_each_views_lens(tmp_path):
    ds = _capture(tmp_path)
    VI.colmap_to_cameras_json(ds, tmp_path / "cameras.json")
    cj = json.loads((tmp_path / "cameras.json").read_text())
    assert [c["fx"] for c in cj["cameras"]] == [TELE[2], ORBIT[2], ORBIT[2], ORBIT[2]]
    assert (cj["width"], cj["height"]) == (1080, 1920)


def test_visibility_counts_each_camera_through_its_own_lens():
    # a vertex 1.5 m off axis at 3 m: inside the orbit lens's frame, outside the telephoto's
    V = np.array([[0.0, 1.5, 3.0]])
    cam = (np.eye(3), np.zeros(3))
    assert VI.visible_counts(V, ORBIT, [cam]).tolist() == [1]
    assert VI.visible_counts(V, TELE, [cam]).tolist() == [0]
    assert VI.visible_counts(V, [ORBIT, TELE], [cam, cam]).tolist() == [1]


def test_orbit_from_record_takes_the_most_common_lens():
    k_tele, k_orbit = [4617.4, 4617.4, 540.0, 960.0], [1776.0, 1776.0, 540.0, 960.0]
    record = {"orbit_cameras": {"image_size": [1080, 1920], "intrinsics": [k_tele, k_orbit, k_orbit, k_tele, k_orbit],
                                "position": [[0, 0, 3.0]] * 5},
              "extras": {"orbit_target": [0.0, 0.0, 0.0]}}
    K, target, radius = C.orbit_from_record(record)
    assert (K.fx, K.width) == (1776.0, 1080)
    assert radius == 3.0
