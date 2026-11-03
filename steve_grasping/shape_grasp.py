"""
shape_grasp.py — shape-based grasp selection for a hand: fit a primitive to the object's depth points, pick the
hand's grasp type from its size (tool config `grasp_types`), and place that type's grasp frame on the object.

    from steve_grasping.shape_grasp import ObjectCloud, fit_primitive, choose_grasp, grasp_candidates
    cloud.add(points_world, centre_world)                 # every detection (points relative to the tracked centre)
    prim = fit_primitive(cloud.points(centre_now), table_z) or profile_primitive(target, centre_now)
    gtype, why = choose_grasp(prim, tool)
    for c in grasp_candidates(prim, tool, gtype, base_xy): ...   # GraspCandidate: the type frame's world pose

Primitives (objects stand upright on a table: cylinders and boxes are vertical):
  sphere    centre, diameter                                  ball
  cylinder  axis xy, diameter, height above the table        can, bottle, cup
  box       centre, horizontal size (long x, short y) + yaw, height    block, box, flat objects
The fit tries all three on the points (minus the table) and keeps the one with the smallest surface distance
(RMS); a fit worse than MAX_RMS returns None — the caller then uses the target profile's shape and size.

Grasp frames (tool config): z = approach into the object, y = the closing direction (thumb -> fingers);
grasp_candidates() gives the type frame's world pose; the WBC's controlled pinch site follows from
hand_geometry.site_in_type().
"""

from dataclasses import dataclass

import numpy as np

R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # = active_perception_wbc
MAX_RMS = 0.006       # m  a primitive fitting worse than this: no fit (use the target profile)
MIN_POINTS = 40
TRIM_MIN = 0.003      # m  points nearer the fitted surface than this are never outliers
TABLE_CLEAR = 0.006   # m  points this close above the table top belong to the table
FIT_RANGE = 0.9       # m  frames from at most this camera distance are fitted: the D415's depth noise grows with
                      #    range^2 (~3 mm here, ~1 cm at 1.6 m: no primitive fits within MAX_RMS)


@dataclass
class Primitive:
    kind: str                  # sphere | cylinder | box
    center: np.ndarray         # geometric centre (world)
    size: np.ndarray           # extent along the primitive's x (long), y (short), z [m]
    yaw: float = 0.0           # box: heading of its long horizontal side
    rms: float = 0.0
    n: int = 0
    source: str = "fit"        # fit | profile

    @property
    def width(self):
        return float(min(self.size[0], self.size[1]))

    @property
    def length(self):
        return float(max(self.size[0], self.size[1]))

    @property
    def height(self):
        return float(self.size[2])

    def narrow_axis(self):
        """Horizontal unit vector across the narrow side (box), None for round shapes."""
        if self.kind != "box" or self.size[0] - self.size[1] < 0.005:
            return None
        return np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])

    def __str__(self):
        s = (f"{self.kind} {self.width * 100:.1f} cm" if self.kind == "sphere" else
             f"{self.kind} {self.length * 100:.1f} x {self.width * 100:.1f} x {self.height * 100:.1f} cm")
        return s + (f" (fit RMS {self.rms * 1000:.1f} mm, {self.n} points)" if self.source == "fit"
                    else " (target profile)")


# ── point cloud of the object, gathered over many detections ─────────────────────────────────
class ObjectCloud:
    """Object points relative to the tracked centre at the time of their frame (a moving object's points
    stay aligned), with the frame's camera distance, voxel-thinned, the newest MAX_FRAMES frames."""

    MAX_FRAMES = 30
    VOXEL = 0.003

    def __init__(self):
        self.frames = []

    def add(self, points, centre, rng=None):
        """rng: the camera's distance to the object in this frame [m] (None: unknown, always used)."""
        p = np.asarray(points, float).reshape(-1, 3)
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) == 0:
            return
        self.frames = (self.frames + [(p - np.asarray(centre, float), rng)])[-self.MAX_FRAMES:]

    def n_frames(self, max_range=None):
        return sum(1 for _, r in self.frames if max_range is None or r is None or r <= max_range)

    def clear(self):
        self.frames = []

    def __len__(self):
        return sum(len(f) for f, _ in self.frames)

    def points(self, centre, max_range=None):
        """The points (world, at `centre`) of the frames from at most max_range camera distance."""
        fr = [f for f, r in self.frames if max_range is None or r is None or r <= max_range]
        if not fr:
            return np.zeros((0, 3))
        p = np.vstack(fr)
        _, keep = np.unique(np.floor(p / self.VOXEL).astype(np.int64), axis=0, return_index=True)
        return p[np.sort(keep)] + np.asarray(centre, float)


# ── fitting ──────────────────────────────────────────────────────────────────────────────
def _circle(xy):
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = np.sum(xy ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 1e-12)))
    return c, r


def _sphere(p):
    A = np.column_stack([2 * p, np.ones(len(p))])
    b = np.sum(p ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 1e-12)))
    return c, r


def _geometric(p, c0, r0):
    """Refine a sphere / circle (dimension of p) on the geometric distance |p - c| - r, robustly (soft L1, ~D415
    noise): the algebraic fit shrinks the radius of a partial noisy cap / arc — a 5 cm ball's eroded cap 4.1-4.8 cm,
    geometric 4.9-5.1 (as red_ball_detector's sphere_center)."""
    from scipy.optimize import least_squares
    res = least_squares(lambda x: np.linalg.norm(p - x[:-1], axis=1) - x[-1], np.r_[c0, r0], loss="soft_l1",
                        f_scale=0.002)
    return (res.x[:-1], float(abs(res.x[-1]))) if res.success else (c0, r0)


def _rms(d, keep=0.8):
    """RMS of the best `keep` share of the surface distances: a few outliers do not decide between the models."""
    d2 = np.sort(np.square(np.asarray(d, float)))
    return float(np.sqrt(np.mean(d2[:max(1, int(np.ceil(keep * len(d2))))])))


def _sphere_ransac(p, thr=0.003, iters=300, seed=0):
    """Sphere through the most points (within thr): random 4-point spheres, then least squares on the inliers.
    A ball's visible cap with rim pixels of the background (a ring a few cm out) misleads a plain fit (5 -> 8.5 cm)."""
    rng = np.random.default_rng(seed)
    n = len(p)
    best, best_in = None, 0
    for _ in range(iters):
        q = p[rng.choice(n, 4, replace=False)]
        A = np.column_stack([2 * q, np.ones(4)])
        try:
            sol = np.linalg.solve(A, np.sum(q ** 2, axis=1))
        except np.linalg.LinAlgError:
            continue
        c = sol[:3]
        r2 = sol[3] + c @ c
        if not 0.005 ** 2 < r2 < 0.20 ** 2:
            continue
        inl = np.abs(np.linalg.norm(p - c, axis=1) - np.sqrt(r2)) < thr
        if inl.sum() > best_in:
            best, best_in = inl, int(inl.sum())
    q = p if best is None or best_in < max(MIN_POINTS, 0.4 * n) else p[best]
    return _geometric(q, *_sphere(q))


def _hull(xy):
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, np.round(xy, 6)))
    if len(pts) < 3:
        return np.asarray(pts)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and np.cross(np.subtract(h[-1], h[-2]), np.subtract(q, h[-2])) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, hi = half(pts), half(reversed(pts))
    return np.asarray(lo[:-1] + hi[:-1])


def _min_rect(xy):
    """Minimum-area rectangle around 2-D points: (centre, (long, short), yaw of the long side)."""
    h = _hull(xy)
    best = None
    edges = np.diff(np.vstack([h, h[:1]]), axis=0) if len(h) >= 3 else [np.array([1.0, 0.0])]
    for e in edges:
        a = np.arctan2(e[1], e[0])
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = xy @ R.T
        lo, hi = np.percentile(q, 1, axis=0), np.percentile(q, 99, axis=0)  # (noise / stray points: not the extremes)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, a, lo, hi, R)
    _, a, lo, hi, R = best
    c = R.T @ ((lo + hi) / 2)
    ext = hi - lo
    if ext[1] > ext[0]:
        a, ext = a + np.pi / 2, ext[::-1]
    return c, ext, float(np.arctan2(np.sin(a), np.cos(a)))


def _fits(p, base, top):
    """[(Primitive, per-point surface distance)] of the three models on points p."""
    h = top - base
    out = []
    c, r = _sphere_ransac(p)
    if 0.005 < r < 0.20:
        d = np.linalg.norm(p - c, axis=1) - r
        out.append((Primitive("sphere", c, np.full(3, 2 * r), 0.0, _rms(d), len(p)), d))
    # upright cylinder: side (radial) or top cap, whichever is nearer; the circle from the side points (the cap's
    # points lie inside it and would shrink it)
    side = p[p[:, 2] < top - 0.008]
    xy = (side if len(side) >= 20 else p)[:, :2]
    cxy, r = _geometric(xy, *_circle(xy))
    if 0.005 < r < 0.20:
        rho = np.linalg.norm(p[:, :2] - cxy, axis=1)
        d = np.minimum(np.abs(rho - r), np.where(rho <= r, np.abs(p[:, 2] - top), np.inf))
        out.append((Primitive("cylinder", np.array([*cxy, base + h / 2]), np.array([2 * r, 2 * r, h]), 0.0,
                              _rms(d), len(p)), d))
    # upright box: nearest side face or the top
    bc, ext, yaw = _min_rect(p[:, :2])
    if np.all(ext > 0.003):
        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        q = np.abs((p[:, :2] - bc) @ R.T)
        sd = np.min(np.abs(q - ext / 2), axis=1)
        inside = np.all(q <= ext / 2 + 1e-9, axis=1)
        d = np.minimum(sd, np.where(inside, np.abs(p[:, 2] - top), np.inf))
        out.append((Primitive("box", np.array([*bc, base + h / 2]), np.array([ext[0], ext[1], h]), yaw, _rms(d),
                              len(p)), d))
    return out


# prefer the simpler round shapes when they fit about as well (a box fits a ball's partial cap loosely)
_ORDER = {"sphere": 0.0, "cylinder": 0.0005, "box": 0.001}
ELONGATED = 1.3       # box footprint length / width from which the round models are out: at the D415's ~3 mm noise
                      # (0.9 m) their errors differ by < 0.5 mm, the footprint's shape does not depend on it


def _elongated(fits):
    """An elongated box footprint rules out the round models — when the box fits about as well as they do (a ball's
    background rim stretches the box's footprint too, but the sphere then fits far better)."""
    box = next((f for f, _ in fits if f.kind == "box"), None)
    rnd = [f.rms for f, _ in fits if f.kind != "box"]
    if box is not None and box.length >= ELONGATED * box.width and (not rnd or box.rms <= 1.2 * min(rnd)):
        return [(f, d) for f, d in fits if f.kind == "box"]
    return fits


def fit_primitive(points, table_z=None):
    """Best of sphere / upright cylinder / upright box for the object's points (world, z up), or None.
    Robust: the best model's outliers (surface distance > max(3 x the robust spread, TRIM_MIN) — rim pixels with
    background depth, a stray blob part) are dropped and all models fitted again."""
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    if table_z is not None:
        p = p[p[:, 2] > table_z + TABLE_CLEAR]
    if len(p) < MIN_POINTS:
        return None
    base = table_z if table_z is not None else float(np.percentile(p[:, 2], 1))
    top = float(np.percentile(p[:, 2], 99))
    if top - base < 0.005:
        return None
    fits = _elongated(_fits(p, base, top))
    if not fits:
        return None
    best, d = min(fits, key=lambda f: f[0].rms + _ORDER[f[0].kind])
    keep = np.abs(d) <= max(3.0 * 1.4826 * np.median(np.abs(d)), TRIM_MIN)
    if MIN_POINTS <= keep.sum() < len(p):
        p = p[keep]
        top = float(np.percentile(p[:, 2], 99))
        fits = _elongated(_fits(p, base, top)) or fits
        best, _ = min(fits, key=lambda f: f[0].rms + _ORDER[f[0].kind])
    return best if best.rms <= MAX_RMS else None


def profile_primitive(target, centre):
    """The target profile's shape and size (steve_wbc TargetProfile: shape sphere | box, size) at `centre`."""
    s = float(getattr(target, "size", 0.05))
    kind = {"sphere": "sphere", "box": "box", "cube": "box", "cylinder": "cylinder"}.get(
        getattr(target, "shape", "sphere"), "sphere")
    return Primitive(kind, np.asarray(centre, float), np.full(3, s), 0.0, 0.0, 0, "profile")


# ── grasp type and pose ──────────────────────────────────────────────────────────────────
def choose_grasp(prim, tool):
    """(grasp type, reason): the first of the tool's grasp_types whose size limits fit the primitive;
    (None, ...) for a tool with a single grasp."""
    from steve_grasping.hand_geometry import grasp_types
    types = grasp_types(tool)
    if not types:
        return None, "tool has a single grasp"
    why = []
    for name, g in types.items():
        lim = [("min_width", prim.width, lambda v, x: v >= x), ("max_width", prim.width, lambda v, x: v <= x),
               ("min_height", prim.height, lambda v, x: v >= x), ("max_height", prim.height, lambda v, x: v <= x)]
        bad = [f"{k} {g[k] * 100:.1f}" for k, v, ok in lim if k in g and not ok(v, g[k])]
        if not bad:
            return name, f"{prim} -> {name}" + (f" ({'; '.join(why)})" if why else "")
        why.append(f"not {name}: {', '.join(bad)} cm")
    first = next(iter(types))
    return first, f"{prim}: no grasp type fits ({'; '.join(why)}) -> {first}"


@dataclass
class GraspCandidate:
    label: str
    R: np.ndarray              # grasp type frame in the world (z = approach)
    p: np.ndarray              # its origin (world)
    approach: np.ndarray       # unit, into the object
    flip_ok: bool              # the same grasp turned 180 deg about the vertical is equivalent (top-down)


def _topdown(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_TOPDOWN


def grasp_candidates(prim, tool, gtype, base_xy, closing_axis=None, yaws=(0.0, np.pi / 2)):
    """World poses of the grasp type's frame on the primitive, best first.

    topdown: approach down. A box closes across its narrow side (the type frame's closing axis `closing_axis`,
    default its y, aligned with prim.narrow_axis()); round shapes: `yaws` (the WBC's GRASP_YAWS).
    The point: the object centre (wrap), its centre height but >= 1.2 cm above the table (pinch).
    side: approach horizontal from base_xy towards the object's axis, tilted down by elevation_deg, gripped at
    grip_height of the object's height."""
    from steve_grasping.hand_geometry import grasp_types
    g = grasp_types(tool).get(gtype) or {}
    c = np.asarray(prim.center, float)
    base_z = c[2] - prim.height / 2
    if g.get("approach", "topdown") == "side":
        h = c[:2] - np.asarray(base_xy, float)[:2]
        h = h / max(np.linalg.norm(h), 1e-9)
        e = np.radians(g.get("elevation_deg", 30.0))
        z = np.array([h[0] * np.cos(e), h[1] * np.cos(e), -np.sin(e)])
        y = np.array([-h[1], h[0], 0.0])
        R = np.column_stack([np.cross(y, z), y, z])
        p = np.array([c[0], c[1], base_z + g.get("grip_height", 0.55) * prim.height])
        return [GraspCandidate("side", R, p, z, False)]
    p = c.copy()
    if gtype == "pinch" or g.get("point") == "mid_height":
        p[2] = max(c[2], base_z + 0.012)
    u = np.array([0.0, 1.0, 0.0]) if closing_axis is None else np.asarray(closing_axis, float)
    u_w0 = R_TOPDOWN @ u                    # closing direction at yaw 0
    a0 = np.arctan2(u_w0[1], u_w0[0])
    n = prim.narrow_axis()
    if n is not None:
        yaw = np.arctan2(n[1], n[0]) - a0
        out = [GraspCandidate("across", _topdown(yaw), p, np.array([0.0, 0.0, -1.0]), True)]
    else:
        out = [GraspCandidate(f"{np.degrees(y):.0f}°", _topdown(y), p, np.array([0.0, 0.0, -1.0]), True)
               for y in yaws]
    return out
