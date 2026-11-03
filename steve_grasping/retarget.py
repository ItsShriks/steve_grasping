"""
retarget.py — a multi-finger grasp of another hand (DexGraspNet 2.0's LEAP hand) as a grasp of our hand (the
Ability Hand's grasp types, tool config `grasp_types`), by what the grasp does, not by joint angles (the hands'
kinematics differ: LEAP 16 DoF with an opposable thumb, Ability Hand 6 motors with coupled fingers):

  grasp point   c = midway between the thumb pad and the mean pad of the fingers
  approach      a = knuckles -> c (into the object)
  closing       u = thumb -> fingers, orthogonal to a
  aperture      |fingers - thumb|;  depth = |c - knuckles| (how deep the object sits in the hand)

The Ability Hand type: pinch when the source grasp holds the object out at the fingertips (depth >= the
type's `retarget_min_depth`, default 8 cm) with a small aperture (<= the type's max_width), else the wrap.
Its frame (z = approach, closing axis from hand_geometry) is turned onto (a, u) and placed at c.

    from steve_grasping.retarget import Retargeter, SourceGrasp
    rt = Retargeter(tool)                           # Ability Hand closing axes from its model
    g = rt(SourceGrasp(tips={"index": .., "middle": .., "ring": .., "thumb": ..}, knuckles=.., score=..))
    g.gtype, g.R, g.p, g.approach                   # world pose of the grasp type's frame
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SourceGrasp:
    tips: dict                 # finger -> pad centre (world); "thumb" + any of index, middle, ring, pinky
    knuckles: np.ndarray       # mean of the finger base joints (world)
    score: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class HandGrasp:
    gtype: str
    R: np.ndarray              # grasp type frame in the world (z = approach, y = closing direction)
    p: np.ndarray              # grasp point (world)
    approach: np.ndarray
    aperture: float
    depth: float
    score: float

    def __str__(self):
        tilt = np.degrees(np.arccos(np.clip(-self.approach[2], -1.0, 1.0)))
        return (f"{self.gtype} at {np.round(self.p, 3)}, approach {tilt:.0f} deg from vertical, aperture "
                f"{self.aperture * 100:.1f} cm, {self.depth * 100:.1f} cm in the hand, score {self.score:.2f}")


def frame(z, y):
    """Rotation with columns (x, y, z): z exactly, y orthogonalised to it."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    y = np.asarray(y, float) - (np.asarray(y, float) @ z) * z
    n = np.linalg.norm(y)
    if n < 1e-9:  # y parallel to z: any orthogonal
        y = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
        n = np.linalg.norm(y)
    y = y / n
    return np.column_stack([np.cross(y, z), y, z])


class Retargeter:
    def __init__(self, tool, closing_axes=None):
        """closing_axes: {grasp type: thumb -> fingers in that type's frame}; default from the hand model."""
        from steve_grasping.hand_geometry import grasp_types
        self.tool, self.types = tool, grasp_types(tool)
        if closing_axes is None:
            from steve_grasping.hand_geometry import HandKinematics
            hk = HandKinematics(tool)
            closing_axes = {t: hk.closing_axis(t, 0.3) for t in (self.types or {None: {}})}
        self.axes = closing_axes

    def choose(self, aperture, depth):
        if "pinch" in self.types:
            g = self.types["pinch"]
            if depth >= g.get("retarget_min_depth", 0.08) and aperture <= g.get("max_width", 0.045):
                return "pinch"
        return "wrap" if "wrap" in self.types else next(iter(self.types), None)

    def __call__(self, src: SourceGrasp) -> HandGrasp:
        th = np.asarray(src.tips["thumb"], float)
        fingers = [np.asarray(v, float) for k, v in src.tips.items() if k != "thumb"]
        f = np.mean(fingers, axis=0)
        c = 0.5 * (f + th)
        a = c - np.asarray(src.knuckles, float)
        depth = float(np.linalg.norm(a))
        a = a / max(depth, 1e-9)
        u = f - th
        aperture = float(np.linalg.norm(u))
        gtype = self.choose(aperture, depth)
        u_h = np.asarray(self.axes.get(gtype, self.axes.get(None, [0.0, 1.0, 0.0])), float)
        R = frame(a, u) @ frame([0.0, 0.0, 1.0], u_h).T
        return HandGrasp(gtype, R, c, a, aperture, depth, float(src.score))
