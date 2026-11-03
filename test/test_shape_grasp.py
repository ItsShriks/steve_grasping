"""Shape-based grasp, touch check and retargeting (no simulation):  python3 -m pytest test -v"""

import numpy as np
import pytest

from steve_grasping.hand_feedback import TouchConfig, closure_of, grasp_state
from steve_grasping.retarget import Retargeter, SourceGrasp, frame
from steve_grasping.shape_grasp import (ObjectCloud, Primitive, choose_grasp, fit_primitive, grasp_candidates,
                                        profile_primitive)
from steve_grasping.tools import load_tool

TABLE = 0.80
rng = np.random.default_rng(0)


def visible(pts, normals, cam):
    """Points facing the camera (a single view)."""
    return pts[np.einsum("ij,ij->i", normals, cam - pts) > 0]


def sphere_pts(c, r, cam, n=4000, noise=0.001):
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = c + r * u
    p = visible(p, u, cam)
    return p[p[:, 2] > TABLE] + rng.normal(0, noise, (len(p[p[:, 2] > TABLE]), 3))


def cylinder_pts(cxy, d, h, cam, n=6000, noise=0.001):
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(TABLE, TABLE + h, n)
    side = np.column_stack([cxy[0] + d / 2 * np.cos(a), cxy[1] + d / 2 * np.sin(a), z])
    nside = np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])
    rr = d / 2 * np.sqrt(rng.uniform(0, 1, n // 3))
    b = rng.uniform(0, 2 * np.pi, n // 3)
    top = np.column_stack([cxy[0] + rr * np.cos(b), cxy[1] + rr * np.sin(b), np.full(n // 3, TABLE + h)])
    p = np.vstack([visible(side, nside, cam), top])
    return p + rng.normal(0, noise, p.shape)


def box_pts(c, size, yaw, cam, n=6000, noise=0.001):
    L, W, H = size
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pts, nrm = [], []
    for ax, s in ((0, L), (1, W)):
        for sign in (-1, 1):
            q = rng.uniform(-0.5, 0.5, (n // 6, 3)) * [L, W, H]
            q[:, ax] = sign * s / 2
            nn = np.zeros((n // 6, 3))
            nn[:, ax] = sign
            pts.append(q), nrm.append(nn)
    q = rng.uniform(-0.5, 0.5, (n // 6, 3)) * [L, W, H]
    q[:, 2] = H / 2
    pts.append(q), nrm.append(np.tile([0, 0, 1.0], (n // 6, 1)))
    p = np.vstack(pts) @ R.T + [c[0], c[1], TABLE + H / 2]
    p = visible(p, np.vstack(nrm) @ R.T, cam)
    return p + rng.normal(0, noise, p.shape)


CAM = np.array([1.5, 0.2, 1.3])   # a wrist camera ~0.7 m away, looking down at the object


def test_fit_sphere():
    p = sphere_pts(np.array([2.0, 0.0, TABLE + 0.025]), 0.025, CAM)
    f = fit_primitive(p, TABLE)
    assert f is not None and f.kind == "sphere"
    assert abs(f.width - 0.05) < 0.006 and np.linalg.norm(f.center - [2.0, 0.0, TABLE + 0.025]) < 0.006


def test_fit_cylinder():
    f = fit_primitive(cylinder_pts([2.0, 0.1], 0.07, 0.12, CAM), TABLE)
    assert f is not None and f.kind == "cylinder"
    assert abs(f.width - 0.07) < 0.008 and abs(f.height - 0.12) < 0.008


def test_fit_box_narrow_axis():
    f = fit_primitive(box_pts([2.0, -0.1], (0.08, 0.03, 0.04), np.radians(30), CAM), TABLE)
    assert f is not None and f.kind == "box"
    n = f.narrow_axis()
    want = np.array([-np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0])
    assert abs(abs(n @ want) - 1.0) < 0.05
    assert abs(f.width - 0.03) < 0.008


def test_fit_rejects_garbage():
    assert fit_primitive(rng.uniform([1.9, -0.1, TABLE], [2.1, 0.1, TABLE + 0.1], (300, 3)), TABLE) is None
    assert fit_primitive(np.zeros((5, 3)), TABLE) is None


def test_cloud_keeps_points_relative_to_the_track():
    c = ObjectCloud()
    p = sphere_pts(np.array([2.0, 0.0, TABLE + 0.025]), 0.025, CAM)
    c.add(p, [2.0, 0.0, TABLE + 0.025])
    c.add(p + [0.05, 0, 0], [2.05, 0.0, TABLE + 0.025])  # the object rolled 5 cm
    f = fit_primitive(c.points([2.1, 0.0, TABLE + 0.025]), TABLE)
    assert f is not None and f.kind == "sphere" and abs(f.center[0] - 2.1) < 0.006
    c.add(p + rng.normal(0, 0.01, p.shape), [2.0, 0.0, TABLE + 0.025], rng=1.6)   # a far, noisy frame
    assert c.n_frames(0.9) == 2 and c.n_frames() == 3
    assert fit_primitive(c.points([2.0, 0.0, TABLE + 0.025], max_range=0.9), TABLE).kind == "sphere"


@pytest.fixture(scope="module")
def hand():
    return load_tool("ability_hand")


@pytest.mark.parametrize("prim,want", [
    (Primitive("sphere", np.zeros(3), np.full(3, 0.05)), "wrap"),
    (Primitive("box", np.zeros(3), np.array([0.05, 0.05, 0.05])), "wrap"),
    (Primitive("box", np.zeros(3), np.array([0.06, 0.02, 0.03])), "pinch"),
    (Primitive("cylinder", np.zeros(3), np.array([0.06, 0.06, 0.16])), "side"),
])
def test_choose_grasp(hand, prim, want):
    assert choose_grasp(prim, hand)[0] == want


def test_candidates(hand):
    box = Primitive("box", np.array([2.0, 0.0, TABLE + 0.015]), np.array([0.06, 0.02, 0.03]), np.radians(20))
    u = np.array([0.0, 1.0, 0.0])
    (c,) = grasp_candidates(box, hand, "pinch", (1.0, 0.0), u)
    assert np.allclose(c.R[:, 2], [0, 0, -1])                       # top-down
    assert abs(abs(c.R @ u @ box.narrow_axis()) - 1.0) < 1e-6        # closes across the narrow side
    assert c.p[2] >= TABLE + 0.012
    cyl = Primitive("cylinder", np.array([2.0, 0.0, TABLE + 0.08]), np.array([0.06, 0.06, 0.16]))
    (s,) = grasp_candidates(cyl, hand, "side", (1.0, 0.0))
    assert s.approach[0] > 0.8 and s.approach[2] < 0                 # from the robot (-x), tilted down
    assert abs(s.p[2] - (TABLE + 0.55 * 0.16)) < 1e-9
    ball = profile_primitive(type("T", (), {"shape": "sphere", "size": 0.05})(), [2.0, 0.0, TABLE + 0.025])
    assert len(grasp_candidates(ball, hand, "wrap", (1.0, 0.0))) == 2   # round: the WBC's yaws


def test_touch_states():
    cfg = TouchConfig(threshold=0.3)
    t = np.zeros(30)
    assert grasp_state(t, 0.0, 0.0, cfg).state == "open"
    assert grasp_state(t, 0.5, 1.0, cfg).state == "blocked"            # a wrap: no fingertip touch, stopped short
    assert grasp_state(t, 0.5, 1.0, cfg).holding
    assert grasp_state(t, 0.97, 1.0, cfg).state == "empty"
    assert not grasp_state(t, 0.97, 1.0, cfg).holding
    t[0:6], t[24:30] = 0.2, 0.1        # index + thumb, 1.2 / 0.6 summed
    s = grasp_state(t, 0.6, 1.0, cfg)
    assert s.state == "held" and s.n_fingers == 2
    t[24:30] = 0.0                     # no thumb: touching, not holding
    assert grasp_state(t, 0.6, 1.0, cfg).state == "contact"
    assert abs(closure_of([50, 0], [0, 0], [100, 0]) - 0.5) < 1e-9


def test_touch_config_per_source(hand):
    assert TouchConfig.from_tool(hand, "sim").threshold == 0.3
    assert TouchConfig.from_tool(hand, "real").threshold == 1.0


def test_retarget_maps_approach_and_closing():
    rt = Retargeter(type("T", (), {"config": {"grasp_types": {"wrap": {}, "pinch": {"max_width": 0.045}}},
                                   "name": "t"})(),
                    closing_axes={"wrap": np.array([0.3, -0.95, 0.0]) / np.linalg.norm([0.3, -0.95]),
                                  "pinch": np.array([0.0, 1.0, 0.0])})
    kn = np.array([2.0, 0.0, 1.0])
    # power grasp from above: object 6 cm under the knuckles, 7 cm aperture
    g = rt(SourceGrasp({"index": [2.0, 0.035, 0.94], "middle": [2.01, 0.035, 0.94], "thumb": [2.0, -0.035, 0.94]},
                       kn, 0.9))
    assert g.gtype == "wrap" and np.allclose(g.approach, [0, 0, -1], atol=0.05)
    assert np.allclose(g.R[:, 2], g.approach)
    u = np.array([2.005, 0.035, 0.94]) - [2.0, -0.035, 0.94]
    assert np.allclose(g.R @ rt.axes["wrap"], u / np.linalg.norm(u), atol=0.02)    # thumb -> fingers kept
    # precision grasp: fingertips 10 cm out, 2 cm apart
    g = rt(SourceGrasp({"index": [2.0, 0.01, 0.90], "thumb": [2.0, -0.01, 0.90]}, kn))
    assert g.gtype == "pinch"
    R = frame([0, 0, -1], [1, 0, 0])
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_hand_model_axes(hand):
    from steve_grasping.hand_geometry import HandKinematics, site_in_type
    hk = HandKinematics(hand)
    for t in ("wrap", "pinch", "side"):
        u = hk.closing_axis(t, 0.3)
        assert abs(np.linalg.norm(u) - 1) < 1e-9 and abs(u[2]) < 1e-9
    R, p = site_in_type(hand, "wrap")
    assert np.allclose(R, np.eye(3)) and np.allclose(p, 0)
    mid, gap = hk.pinch_point("pinch", 1.0)
    assert gap < 0.01   # the pinch synergy brings the pads together


def test_fit_ignores_a_background_rim():
    """Rim pixels with the table's depth behind the ball (lifted 1 cm by a depth-scale bias) made a plain fit of a
    5 cm ball 8.5-8.9 cm (sim, 2026-10-30)."""
    c = np.array([2.0, 0.0, TABLE + 0.025])
    p = sphere_pts(c, 0.025, CAM, noise=0.002)
    a = rng.uniform(-1.2, 1.2, len(p) // 5)
    ring = np.column_stack([2.03 + 0.035 * np.cos(a), 0.035 * np.sin(a), np.full(len(a), TABLE + 0.010)])
    f = fit_primitive(np.vstack([p, ring + rng.normal(0, 0.002, ring.shape)]), TABLE)
    assert f is not None and f.kind == "sphere" and abs(f.width - 0.05) < 0.004
    assert np.linalg.norm(f.center - c) < 0.004


def test_fit_elongated_box_at_d415_noise():
    """~3 mm depth noise (D415 at ~0.9 m): the round models fit about as well; the footprint decides."""
    f = fit_primitive(box_pts([2.0, -0.1], (0.08, 0.03, 0.04), np.radians(30), CAM, noise=0.003), TABLE)
    assert f is not None and f.kind == "box" and abs(f.width - 0.03) < 0.012


def test_fit_size_of_an_eroded_cap():
    """The rim-free (eroded) cap of a ball is a small, flat patch: the algebraic fit made the 5 cm ball 3.8 cm
    (sim, 2026-10-30); the geometric refinement keeps its size."""
    c = np.array([2.0, 0.0, TABLE + 0.025])
    u = rng.normal(size=(6000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = (CAM - c) / np.linalg.norm(CAM - c)
    p = (c + 0.025 * u)[(u @ v) > np.cos(np.radians(45))]
    f = fit_primitive(p + rng.normal(0, 0.003, p.shape), TABLE)
    assert f is not None and f.kind == "sphere" and abs(f.width - 0.05) < 0.005
