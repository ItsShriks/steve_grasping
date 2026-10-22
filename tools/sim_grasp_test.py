#!/usr/bin/env python3
"""
sim_grasp_test.py — lift test of a tool's grasp in MuJoCo, without the robot: the tool flies on a mocap
carriage, top-down like the WBC's grasp (pinch z down, closing axis = pinch y).

    python3 tools/sim_grasp_test.py --tool ability_hand                 # config pinch / synergies
    python3 tools/sim_grasp_test.py --tool ability_hand --viewer        # watch it
    python3 tools/sim_grasp_test.py --tool ability_hand --sweep         # search the pinch point (y, z)
    python3 tools/sim_grasp_test.py --tool robotiq_2f85 --grasp_dz -0.015

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


def scene(tool, radius, mass, box):
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
    <body name="target" pos="0 0 {TABLE_Z + radius}">
      <freejoint name="obj"/>
      <geom name="obj" type="{'box' if box else 'sphere'}" size="{radius} {radius} {radius}" mass="{mass}"
            friction="1.5 0.01 0.001" rgba="0.9 0.1 0.1 1"/>
    </body>
    <body name="carriage" mocap="true"/>
    <body name="wrist_3_link" pos="0 0 1.5">
      <freejoint name="carriage_free"/>
      <geom type="cylinder" size="0.0375 0.01" pos="0 0 -0.01" mass="0.2" contype="0" conaffinity="0"/>
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


def run(tool, pinch_pos=None, grasp_dz=None, radius=0.025, mass=0.05, box=False, viewer=False, verbose=True):
    import mujoco
    if pinch_pos is not None:
        tool.pinch_pos = np.asarray(pinch_pos, float)
    dz = tool.grasp_dz if grasp_dz is None else grasp_dz
    m = mujoco.MjModel.from_xml_string(scene(tool, radius, mass, box))
    d = mujoco.MjData(m)
    grip = SimGripper(m, tool)
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    mujoco.mj_forward(m, d)
    # flange pose that puts `pinch` at p with orientation R_TOPDOWN
    R_fp = d.xmat[flange].reshape(3, 3).T @ d.site_xmat[pinch].reshape(3, 3)       # pinch in flange
    p_fp = d.xmat[flange].reshape(3, 3).T @ (d.site_xpos[pinch] - d.xpos[flange])
    R_f = R_TOPDOWN @ R_fp.T
    q_f = _quat_from_R(R_f)
    flange_at = lambda p: p - R_f @ p_fp
    goal = np.array([0.0, 0.0, TABLE_Z + radius + dz])
    above = goal + [0, 0, 0.12]
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
            if view is not None and i % 8 == 0:
                view.sync()
                if not view.is_running():
                    return None
        log[name] = (d.xpos[obj].copy(), d.site_xpos[pinch].copy())
    if view is not None:
        view.close()
    z0 = TABLE_Z + radius
    obj_end, pinch_end = log["hold"]
    rel_lift = log["lift"][0] - log["lift"][1]
    rel_hold = obj_end - pinch_end
    lifted = obj_end[2] - z0
    slip = float(np.linalg.norm(rel_hold - rel_lift))
    # the arm cannot push the fingers through the table like the stiff carriage here: the descent counts
    # only if the grasp point got within 1.5 cm of its goal (fingertips resting on the table)
    short = float(np.linalg.norm(log["descend"][1] - goal))
    ok = lifted > 0.12 and slip < 0.02 and short < 0.015
    at_close = log["close"][0] - log["close"][1]
    if verbose:
        print(f"{tool.name}: pinch {np.round(tool.pinch_pos, 3)} grasp_dz {dz:+.3f} | object pushed "
              f"{np.linalg.norm(log['descend'][0][:2]) * 100:.1f} cm on the descent, grasp point "
              f"{short * 100:.1f} cm short of its goal | after closing "
              f"{np.round(at_close * 100, 1)} cm from the pinch point | lifted {lifted * 100:.1f} cm, slip "
              f"{slip * 100:.1f} cm -> {'SUCCESS' if ok else 'FAIL'}")
    return ok, lifted, slip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", default="ability_hand")
    p.add_argument("--pinch", type=float, nargs=3, help="override the pinch xyz (tool base frame) [m]")
    p.add_argument("--grasp_dz", type=float)
    p.add_argument("--radius", type=float, default=0.025)
    p.add_argument("--mass", type=float, default=0.05)
    p.add_argument("--box", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--sweep", action="store_true", help="grid over pinch y / z around the config value")
    a = p.parse_args()
    if a.viewer:
        import mujoco.viewer  # noqa: F401
    if not a.sweep:
        r = run(load_tool(a.tool), a.pinch, a.grasp_dz, a.radius, a.mass, a.box, a.viewer)
        sys.exit(0 if r and r[0] else 1)
    base = np.asarray(a.pinch, float) if a.pinch else load_tool(a.tool).pinch_pos
    good = []
    for dy in np.arange(-0.02, 0.0201, 0.01):
        row = []
        for dz in np.arange(-0.03, 0.0301, 0.01):
            pp = base + [0, dy, dz]
            ok, lifted, slip = run(load_tool(a.tool), pp, a.grasp_dz, a.radius, a.mass, a.box, verbose=False)
            row.append("  ok " if ok else f"{lifted * 100:4.0f} ")
            if ok:
                good.append(pp)
        print(f"y {base[1] + dy:+.3f}: " + " ".join(row) + "   (z " + " ".join(
            f"{base[2] + dz:.3f}" for dz in np.arange(-0.03, 0.0301, 0.01)) + "; else: lift cm)")
    if good:
        print(f"successful pinch points: {len(good)}; centre {np.round(np.mean(good, axis=0), 3)}")


if __name__ == "__main__":
    main()
