#!/usr/bin/env python3
"""
sim_grasp_test.py — lift test of a tool's grasp in MuJoCo, without the robot: the tool flies on a mocap
carriage, top-down like the WBC's grasp (pinch z down, closing axis = pinch y) — or, with --side ELEV, from
the side: the approach (pinch z) horizontal, tilted down by ELEV deg, the hand rolled about it by --roll.

    python3 tools/sim_grasp_test.py --tool ability_hand                 # config pinch / synergies
    python3 tools/sim_grasp_test.py --tool ability_hand --viewer        # watch it
    python3 tools/sim_grasp_test.py --tool ability_hand --sweep         # search the pinch point (y, z)
    python3 tools/sim_grasp_test.py --tool robotiq_2f85 --grasp_dz -0.015
    python3 tools/sim_grasp_test.py --tool ability_hand_side --side 30 --viewer   # side grasp, 30 deg down
    python3 tools/sim_grasp_test.py --tool ability_hand_side --side_sweep        # elevation x grip height
    python3 tools/sim_grasp_test.py --tool ability_hand --grasp pinch --box 0.06 0.025 0.03   # a grasp type
    python3 tools/sim_grasp_test.py --tool ability_hand --grasp side --cylinder 0.06 0.16     #   on its object

--grasp TYPE: a grasp type of the tool config (grasp_types) placed on the object as active_perception_wbc's
shape-based grasp does (steve_grasping.shape_grasp; the hand comes from -x), with that type's synergy. Every run
also reports the touch check (steve_grasping.hand_feedback) after closing and after the hold, next to the result.

Sequence (like active_perception_wbc): above the object -> descend to the grasp point (object centre
+ grasp_dz) -> close (GRASP_TIME) -> lift 15 cm -> hold. Success: the object rose with the tool and is
held within 2 cm of where it was when the lift started. Object = the WBC's ball (r 2.5 cm, 50 g,
friction 1.5) unless --radius / --mass / --box.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steve_grasping.gripper import SimGripper          # noqa: E402
from steve_grasping.robot_model import _quat_from_R, _fmt  # noqa: E402
from steve_grasping.tools import load_tool             # noqa: E402

R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # = active_perception_wbc
TABLE_Z = 0.80


def side_R(elev_deg, roll_deg):
    """Grasp frame of a side grasp: z (approach) along +x tilted down by elev, y (palm normal for a hand with
    pinch = hand axes) horizontal at roll 0, rolled about z by roll."""
    e, r = np.radians(elev_deg), np.radians(roll_deg)
    z = np.array([np.cos(e), 0.0, -np.sin(e)])
    y0 = np.array([0.0, 1.0, 0.0])
    x0 = np.cross(y0, z)
    y = np.cos(r) * y0 + np.sin(r) * x0
    return np.column_stack([np.cross(y, z), y, z])


def obj_geom(radius, box, shape):
    """(MJCF geom type, size, half height) of the test object."""
    if shape is not None:
        kind, dims = shape
        if kind == "box":
            return "box", "{} {} {}".format(*(np.asarray(dims) / 2)), dims[2] / 2
        return "cylinder", f"{dims[0] / 2} {dims[1] / 2}", dims[1] / 2
    return ("box" if box else "sphere"), f"{radius} {radius} {radius}", radius


def scene(tool, radius, mass, box, wrist=False, shape=None):
    import xml.etree.ElementTree as ET
    from steve_grasping.robot_model import compose
    # a minimal "robot": a mocap carriage whose child `hand_mount` is the flange (tool0)
    robot = f"""<mujoco model="grasp_test">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom name="table" type="box" pos="0 0 {TABLE_Z / 2}" size="0.4 0.4 {TABLE_Z / 2}" rgba="0.6 0.45 0.3 1"
          friction="0.7 0.005 0.0001"/>
    <body name="target" pos="0 0 {TABLE_Z + obj_geom(radius, box, shape)[2]}">
      <freejoint name="obj"/>
      <geom name="obj" type="{obj_geom(radius, box, shape)[0]}" size="{obj_geom(radius, box, shape)[1]}" mass="{mass}"
            friction="1.5 0.01 0.001" rgba="0.9 0.1 0.1 1"/>
    </body>
    <body name="carriage" mocap="true"/>
    <body name="wrist_3_link" pos="0 0 1.5">
      <freejoint name="carriage_free"/>
      <geom type="cylinder" size="0.0375 0.01" pos="0 0 -0.01" mass="0.2" contype="0" conaffinity="0"/>
      {'<geom name="wrist_proxy" type="capsule" fromto="0 0 -0.02 0 0 -0.12" size="0.045" mass="0.01" rgba="0.3 0.3 0.3 0.4"/>' if wrist else ''}
      <body name="hand_mount"/>
    </body>
  </worldbody>
  <equality><weld body1="carriage" body2="wrist_3_link" relpose="0 0 0 1 0 0 0" solref="0.005 1"/></equality>
</mujoco>"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(robot)
    xml = compose(f.name, tool)
    Path(f.name).unlink()
    return xml


def run(tool, pinch_pos=None, grasp_dz=None, radius=0.025, mass=0.05, box=False, viewer=False, verbose=True,
        side=None, roll=0.0, grasp=None, shape=None):
    import mujoco
    from steve_grasping.hand_feedback import TouchConfig, grasp_state
    if pinch_pos is not None:
        tool.pinch_pos = np.asarray(pinch_pos, float)
    dz = tool.grasp_dz if grasp_dz is None else grasp_dz
    half = obj_geom(radius, box, shape)[2]
    R_goal = R_TOPDOWN if side is None else side_R(side, roll)
    goal = np.array([0.0, 0.0, TABLE_Z + half + dz])
    approach = R_goal[:, 2]
    if grasp is not None:  # a grasp type, placed like the WBC's shape-based grasp; the pinch site follows
        from steve_grasping.hand_geometry import HandKinematics, grasp_types, site_in_type
        from steve_grasping.shape_grasp import Primitive, grasp_candidates
        if grasp not in grasp_types(tool):
            sys.exit(f"--grasp {grasp}: not in {tool.name}'s grasp_types {list(grasp_types(tool))}")
        size = (np.full(3, 2 * radius) if shape is None else
                np.array(shape[1]) if shape[0] == "box" else np.array([shape[1][0], shape[1][0], shape[1][1]]))
        kind = "sphere" if shape is None and not box else ("box" if box and shape is None else shape[0])
        prim = Primitive(kind, np.array([0.0, 0.0, TABLE_Z + half]), size, 0.0)
        c = grasp_candidates(prim, tool, grasp, (-1.0, 0.0), HandKinematics(tool).closing_axis(grasp, 0.3), (0.0,))[0]
        R_st, p_st = site_in_type(tool, grasp)
        R_goal, goal, approach = c.R @ R_st, c.p + c.R @ p_st + [0.0, 0.0, dz], c.approach
        side = None if grasp_types(tool)[grasp].get("approach", "topdown") != "side" else \
            grasp_types(tool)[grasp].get("elevation_deg", 30.0)
    # side: the UR wrist behind the flange (capsule r 4.5 cm) collides with the table — at ball height it cannot
    m = mujoco.MjModel.from_xml_string(scene(tool, radius, mass, box, wrist=side is not None, shape=shape))
    d = mujoco.MjData(m)
    grip = SimGripper(m, tool)
    grip.set_type(grasp)
    touch_cfg = TouchConfig.from_tool(tool, "sim")
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    mujoco.mj_forward(m, d)
    # flange pose that puts `pinch` at p with orientation R_TOPDOWN
    R_fp = d.xmat[flange].reshape(3, 3).T @ d.site_xmat[pinch].reshape(3, 3)       # pinch in flange
    p_fp = d.xmat[flange].reshape(3, 3).T @ (d.site_xpos[pinch] - d.xpos[flange])
    R_f = R_goal @ R_fp.T
    q_f = _quat_from_R(R_f)
    flange_at = lambda p: p - R_f @ p_fp
    above = goal - 0.12 * approach  # pre-grasp: back along the approach (top-down: 12 cm above)
    # start: carriage above the object, gripper open
    jq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "carriage_free")]
    d.qpos[jq:jq + 7] = [*flange_at(above), *q_f]
    d.mocap_pos[0], d.mocap_quat[0] = flange_at(above), q_f
    grip.command(d, close=False)
    mujoco.mj_forward(m, d)

    phases = [("settle", 0.5, above, False), ("descend", 1.5, goal, False), ("close", 1.0, goal, True),
              ("lift", 1.5, goal + [0, 0, 0.15], True), ("hold", 1.0, goal + [0, 0, 0.15], True)]
    view = mujoco.viewer.launch_passive(m, d) if viewer else None
    log, start = {}, flange_at(above)
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wrist_proxy")
    wrist_hit = False
    for name, dur, target, close in phases:
        p0 = d.mocap_pos[0].copy()
        p1 = flange_at(target)
        grip.command(d, close=close)
        n = int(dur / m.opt.timestep)
        for i in range(n):
            s = min(1.0, (i + 1) / (0.8 * n))
            s = s * s * (3 - 2 * s)
            d.mocap_pos[0] = p0 + s * (p1 - p0)
            mujoco.mj_step(m, d)
            if wrist >= 0 and not wrist_hit:
                wrist_hit = any({d.contact[k].geom1, d.contact[k].geom2} == {wrist, table} for k in range(d.ncon))
            if view is not None and i % 8 == 0:
                view.sync()
                if not view.is_running():
                    return None
        log[name] = (d.xpos[obj].copy(), d.site_xpos[pinch].copy())
        if name in ("close", "hold"):
            log["touch_" + name] = grasp_state(grip.touch(d), grip.closure(d), grip.cmd, touch_cfg)
    if view is not None:
        view.close()
    z0 = TABLE_Z + half
    obj_end, pinch_end = log["hold"]
    rel_lift = log["lift"][0] - log["lift"][1]
    rel_hold = obj_end - pinch_end
    lifted = obj_end[2] - z0
    slip = float(np.linalg.norm(rel_hold - rel_lift))
    # the arm cannot push the fingers through the table like the stiff carriage here: the descent counts
    # only if the grasp point got within 1.5 cm of its goal (fingertips resting on the table)
    short = float(np.linalg.norm(log["descend"][1] - goal))
    ok = lifted > 0.12 and slip < 0.02 and short < 0.015 and not wrist_hit
    at_close = log["close"][0] - log["close"][1]
    if verbose:
        tc, th = log.get("touch_close"), log.get("touch_hold")
        if th is not None:
            agree = th.holding == ok
            print(f"touch check: after closing {tc} | after the hold {th} -> "
                  f"{'agrees' if agree else 'DISAGREES'} with the lift result")
        print(f"{tool.name}: " + (f"grasp type {grasp} | " if grasp else "")
              + (f"side {side:.0f} deg down, roll {roll:.0f} | " if side is not None else "")
              + ("WRIST HITS THE TABLE | " if wrist_hit else "")
              + f"pinch {np.round(tool.pinch_pos, 3)} grasp_dz {dz:+.3f} | object pushed "
              f"{np.linalg.norm(log['descend'][0][:2]) * 100:.1f} cm on the descent, grasp point "
              f"{short * 100:.1f} cm short of its goal | after closing "
              f"{np.round(at_close * 100, 1)} cm from the pinch point | lifted {lifted * 100:.1f} cm, slip "
              f"{slip * 100:.1f} cm -> {'SUCCESS' if ok else 'FAIL'}")
    return ok, lifted, slip, wrist_hit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", default="ability_hand")
    p.add_argument("--pinch", type=float, nargs=3, help="override the pinch xyz (tool base frame) [m]")
    p.add_argument("--grasp_dz", type=float)
    p.add_argument("--radius", type=float, default=0.025)
    p.add_argument("--mass", type=float, default=0.05)
    p.add_argument("--box", type=float, nargs="*", default=None, metavar=("L", "W"),
                   help="a box: no values = a cube of --radius; L W H = its size [m] (L along x)")
    p.add_argument("--cylinder", type=float, nargs=2, default=None, metavar=("D", "H"),
                   help="an upright cylinder of diameter D, height H [m]")
    p.add_argument("--grasp", default=None, help="a grasp type of the tool config (grasp_types), e.g. wrap / pinch / side")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--sweep", action="store_true", help="grid over pinch y / z around the config value")
    p.add_argument("--side", type=float, default=None, metavar="ELEV",
                   help="side grasp: approach horizontal, tilted down by ELEV deg (default: the tool's side_elevation_deg)")
    p.add_argument("--roll", type=float, default=None, help="side grasp: hand rolled about the approach [deg]")
    p.add_argument("--side_sweep", action="store_true", help="side grasp: grid over elevation x grip height (pinch x)")
    a = p.parse_args()
    a.shape = (("box", a.box) if a.box and len(a.box) == 3 else
               ("cylinder", a.cylinder) if a.cylinder else None)
    if a.box and len(a.box) not in (0, 3):
        sys.exit("--box: no values (a cube of --radius) or L W H")
    a.box = a.box is not None and not a.box
    if a.viewer:
        import mujoco.viewer  # noqa: F401
    tool0 = load_tool(a.tool)
    side = a.side if a.side is not None else (tool0.config.get("side_elevation_deg")
                                              if tool0.grasp_style == "side" else None)
    roll = a.roll if a.roll is not None else float(tool0.config.get("side_roll_deg", 0.0))
    if a.side_sweep:
        base = np.asarray(a.pinch, float) if a.pinch else tool0.pinch_pos
        xs = np.arange(-0.04, 0.0201, 0.01)
        for elev in (0, 10, 20, 30, 40, 50):
            row = []
            for dx in xs:
                ok, lifted, slip, hit = run(load_tool(a.tool), base + [dx, 0, 0], a.grasp_dz, a.radius, a.mass,
                                            a.box, verbose=False, side=elev, roll=roll)
                row.append("  ok " if ok else (" wrist" if hit else f"{lifted * 100:4.0f} "))
            print(f"elev {elev:2d} deg: " + " ".join(row) + "   (pinch x " + " ".join(f"{base[0] + dx:+.2f}" for dx in xs)
                  + "; else: lift cm / wrist on the table)")
        return
    if not a.sweep:
        r = run(tool0, a.pinch, a.grasp_dz, a.radius, a.mass, a.box, a.viewer, side=side, roll=roll,
                grasp=a.grasp, shape=a.shape)
        sys.exit(0 if r and r[0] else 1)
    base = np.asarray(a.pinch, float) if a.pinch else load_tool(a.tool).pinch_pos
    good = []
    for dy in np.arange(-0.02, 0.0201, 0.01):
        row = []
        for dz in np.arange(-0.03, 0.0301, 0.01):
            pp = base + [0, dy, dz]
            ok, lifted, slip, _ = run(load_tool(a.tool), pp, a.grasp_dz, a.radius, a.mass, a.box, verbose=False,
                                      side=side, roll=roll)
            row.append("  ok " if ok else f"{lifted * 100:4.0f} ")
            if ok:
                good.append(pp)
        print(f"y {base[1] + dy:+.3f}: " + " ".join(row) + "   (z " + " ".join(
            f"{base[2] + dz:.3f}" for dz in np.arange(-0.03, 0.0301, 0.01)) + "; else: lift cm)")
    if good:
        print(f"successful pinch points: {len(good)}; centre {np.round(np.mean(good, axis=0), 3)}")


if __name__ == "__main__":
    main()
