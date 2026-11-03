"""
hand_geometry.py — kinematics of a hand tool (Ability Hand) from its MuJoCo model alone, no physics: where the
fingertip pads are for a synergy, and from that the grasp frames the WBC and the retargeting need.

    from steve_grasping.hand_geometry import HandKinematics
    hk = HandKinematics(load_tool("ability_hand"))
    tips = hk.fingertips(hk.synergy("pinch", 0.7))     # {finger: pad centre in the hand base frame}
    hk.closing_axis("wrap")                            # thumb -> fingers direction in the pinch frame

The hand base frame is the hand model's root body = the frame the tool config's `mount`, `pinch` and grasp type
frames are given in. Finger joints coupled by joint equalities (Ability Hand: PIP = f(MCP)) follow their driver.
Fingertip pad = the mean of the finger's 6 FSR sites (abh_fsr<6 i> .. <6 i + 5>, fingers index, middle, ring,
pinky, thumb), the hand's touch surface.
"""

import numpy as np

from steve_grasping.hand_feedback import FINGERS


class HandKinematics:
    def __init__(self, tool):
        import mujoco
        self.mj, self.tool = mujoco, tool
        m = self.m = mujoco.MjModel.from_xml_path(str(tool.model_path))
        self.d = mujoco.MjData(m)
        self.root = 1  # the hand's root body (first body under the world)
        sim = tool.sim or {}
        prefix = sim.get("actuator_prefix", "")
        self.units = np.radians(1.0) if sim.get("units") == "deg" else 1.0
        names = sim.get("actuators") or []
        aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n) for n in names]
        if not names or min(aid) < 0:
            raise ValueError(f"tool {tool.name}: sim.actuators missing / not in {tool.model_path.name}")
        self.act_qpos = [m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in aid]
        # joint couplings: q1 - q1_0 = sum_k c_k (q2 - q2_0)^k
        self.couple = []
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0:
                j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
                self.couple.append((m.jnt_qposadr[j1], m.jnt_qposadr[j2], m.eq_data[e, :5].copy()))
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        self.pads = {f: [sid(f"abh_fsr{6 * i + k}") for k in range(6)] for i, f in enumerate(FINGERS)}
        if any(s < 0 for v in self.pads.values() for s in v):
            raise ValueError(f"tool {tool.name}: no FSR sites abh_fsr0..29 in {tool.model_path.name}")

    # ── synergies (the tool config's units: Ability Hand API degrees) ──────────────────────
    def synergy(self, grasp_type=None, closure=1.0):
        """Motor targets of a grasp type's synergy at a closure 0 (open) .. 1 (close), config units."""
        o, c = synergies(self.tool, grasp_type)
        return o + float(closure) * (c - o)

    def set(self, targets):
        """Pose the hand: motors at `targets` (config units), coupled joints following."""
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        for adr, v in zip(self.act_qpos, np.asarray(targets, float) * self.units):
            d.qpos[adr] = v
        for a1, a2, c in self.couple:
            x = d.qpos[a2] - m.qpos0[a2]
            d.qpos[a1] = m.qpos0[a1] + sum(c[k] * x ** k for k in range(5))
        self.mj.mj_kinematics(m, d)

    def fingertips(self, targets):
        """{finger: pad centre} in the hand base frame for motor targets (config units)."""
        self.set(targets)
        R, p = self.d.xmat[self.root].reshape(3, 3), self.d.xpos[self.root]
        return {f: R.T @ (np.mean([self.d.site_xpos[s] for s in ids], axis=0) - p) for f, ids in self.pads.items()}

    # ── grasp frames ──────────────────────────────────────────────────────────────────────
    def frame(self, grasp_type=None):
        """(R, p) of a grasp type's frame in the hand base frame (the pinch site for `wrap` / none)."""
        return grasp_frame(self.tool, grasp_type)

    def closing_axis(self, grasp_type=None, closure=0.6):
        """Unit thumb -> fingers direction (fingers that close in this synergy) in the grasp type's frame,
        orthogonal to its approach (z)."""
        tips = self.fingertips(self.synergy(grasp_type, closure))
        used = closing_fingers(self.tool, grasp_type)
        f = np.mean([tips[k] for k in used], axis=0)
        u = f - tips["thumb"]
        R, _ = self.frame(grasp_type)
        u = R.T @ u
        u[2] = 0.0
        n = np.linalg.norm(u)
        return u / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])

    def pinch_point(self, grasp_type, closure):
        """Midpoint between the thumb pad and the mean pad of the closing fingers, and their distance."""
        tips = self.fingertips(self.synergy(grasp_type, closure))
        f = np.mean([tips[k] for k in closing_fingers(self.tool, grasp_type)], axis=0)
        return 0.5 * (f + tips["thumb"]), float(np.linalg.norm(f - tips["thumb"]))


def _rpy(r):
    from steve_grasping.tools import rpy_matrix
    return rpy_matrix(*np.radians(r))


def grasp_types(tool):
    """The tool config's `grasp_types` (empty: the tool has one grasp, its pinch site + synergies)."""
    return dict((tool.config or {}).get("grasp_types") or {})


def grasp_frame(tool, grasp_type=None):
    """(R, p) of a grasp type's frame in the tool base frame; `frame: pinch` / no type: the pinch site."""
    g = grasp_types(tool).get(grasp_type) or {}
    f = g.get("frame", "pinch")
    if f == "pinch":
        return tool.pinch_R.copy(), tool.pinch_pos.copy()
    return _rpy(f.get("rpy_deg", [0, 0, 0])), np.asarray(f.get("xyz", [0, 0, 0]), float)


def site_in_type(tool, grasp_type=None):
    """(R, p) of the WBC's pinch site in the grasp type's frame: the site pose for a type pose X is X @ this."""
    Rg, pg = grasp_frame(tool, grasp_type)
    return Rg.T @ tool.pinch_R, Rg.T @ (tool.pinch_pos - pg)


def synergies(tool, grasp_type=None):
    """(open, close) motor targets of a grasp type (config units), default: the tool's synergies."""
    syn = (grasp_types(tool).get(grasp_type) or {}).get("synergy") or (tool.config or {}).get("synergies") or {}
    if "open" not in syn or "close" not in syn:
        raise ValueError(f"tool {tool.name}: no synergy open / close for grasp type {grasp_type}")
    return np.asarray(syn["open"], float), np.asarray(syn["close"], float)


def closing_fingers(tool, grasp_type=None):
    return tuple((grasp_types(tool).get(grasp_type) or {}).get("fingers") or ("index", "middle", "ring", "pinky"))
