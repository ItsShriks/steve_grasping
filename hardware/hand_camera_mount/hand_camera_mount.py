#!/usr/bin/env python3
"""
hand_camera_mount.py — 3D-printable mount for the UR5e wrist (Ability Hand on the flange): a two-part collar
clamped around wrist 3 that carries
  * the RealSense D415 beside the palm (like the Psyonic D405 bracket of the robot model, mounted higher), and
  * the Ability Hand electronics on the back of the hand: the PSYONIC breakout board and power switch.

    python3 -m venv ~/cadenv && ~/cadenv/bin/pip install manifold3d trimesh numpy
    ~/cadenv/bin/python hand_camera_mount.py            # -> stl/*.stl + the assembly preview

Coordinates: UR tool0 (= ur5etool0 in the URDF), mm — z out of the flange, the Ability Hand's palm towards -y
(steve_grasping ability_hand.yaml: mount roll 180 deg), so the camera sits on -y and the electronics on +y.

Sources of the dimensions:
  UR5e wrist 3        steve_wbc robot model: Ø76 just behind the flange face; the tool connector sticks out on
                      the palm side 30-47 mm behind it, wrist 2 starts 47 mm behind it -> the collar stays in front
  Psyonic bracket     steve_wbc/models/mmo_700/meshes/cameras/d405/UR5_LastJointSensor_D405_Psyonic.STL:
                      collar bore Ø75, z -15..+20 mm, D405 79 mm off the axis on the palm side, 47 mm out
  D415                Intel RealSense D400 datasheet 337029-017: 99 x 23 x 20 mm, 64 g, 2x M3 on the back 45 mm
                      apart, max screw insertion 3 mm, 0.4 Nm
  Breakout board,     NOT published by PSYONIC — PLACEHOLDERS below (marked MEASURE): measure yours and
  power switch        regenerate. Both are also held with zip ties, so only the outer size has to fit
"""

from pathlib import Path

import numpy as np
from manifold3d import Manifold

SEG = 96  # circle resolution

# ── UR5e wrist 3 collar ─────────────────────────────────────────────────────────────────────────────
BORE_D = 76.6          # wrist 3 Ø76 (robot model) + 0.6 clearance — MEASURE your wrist with calipers
WALL = 6.0
COLLAR_Z = (-15.0, 5.0)  # along the flange axis: 15 mm behind the flange face .. 5 mm beyond it
SPLIT_GAP = 1.0        # between the halves: the bolts pull them onto the wrist
M4_HOLE, M4_HEAD_D, M4_NUT_AF, M4_NUT_H = 4.5, 8.5, 7.4, 3.6
LUG_W = 13.0           # lug size beyond the ring (x), its half-height per half (y)
LUG_H = 12.0

# ── D415 beside the palm ────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H, CAM_D = 99.0, 23.0, 20.0     # datasheet
CAM_HOLES = 45.0                           # 2x M3 on the back, centred
CAM_R = 100.0          # camera centre from the flange axis on the palm side (Psyonic D405: 79 — "higher")
CAM_Z = 38.0           # camera centre along the flange axis
CAM_TILT = 40.0        # view tilted from the flange axis away from it [deg]: at 0.3-0.5 m (D415 min. range ~0.3 m)
                       # it looks along the Ability Hand's approach line (the WBC's pinch z), i.e. at the object
PLATE_T = 3.0          # back plate: M3 x 6 screws -> 3 mm into the camera (datasheet max.)
M3_HOLE, M3_HEAD_D = 3.4, 6.2
CAM_CLEAR = 0.4        # camera pocket clearance

# ── electronics tray on the back of the hand ────────────────────────────────────────────────────────
TRAY_Y = 50.0          # underside of the tray from the flange axis (back of the hand: <= 28 mm up to z 120)
TRAY_T = 3.0
TRAY_W = 74.0          # across (x): board and switch side by side
TRAY_Z = (-13.0, 58.0)
RIB_H = 6.0            # stiffening ribs along both long edges
# PSYONIC breakout board — MEASURE: outline (along z, across x) and mounting holes (z, x) from its centre
BOARD_L, BOARD_W = 45.0, 32.0
BOARD_HOLES = [(-18.0, -12.0), (-18.0, 12.0), (18.0, -12.0), (18.0, 12.0)]
BOARD_X, BOARD_Z = -17.0, 25.0   # board centre on the tray (across, along)
STANDOFF_D, STANDOFF_H, STANDOFF_PILOT = 6.0, 5.0, 2.2   # pilot for M2.5 self-tapping screws
# PSYONIC power switch — MEASURE: outer size (along z, across x, height); the open end (+z) leaves its
# button / USB-C / status LED free
SWITCH_L, SWITCH_W, SWITCH_H = 40.0, 26.0, 14.0
SWITCH_X, SWITCH_Z = 19.0, 25.0  # switch centre on the tray (across, along)
CRADLE_WALL = 2.5
ZIP_W, ZIP_L = 2.4, 5.0  # zip-tie slots (up to 4.8 mm ties)

R_IN, R_OUT = BORE_D / 2.0, BORE_D / 2.0 + WALL
OUT = Path(__file__).resolve().parent / "stl"


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────
def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def zcyl(r, z0, z1, x=0.0, y=0.0):
    return Manifold.cylinder(z1 - z0, r, r, SEG).translate([x, y, z0])


def ycyl(r, y0, y1, x, z):  # cylinder along y
    return Manifold.cylinder(y1 - y0, r, r, SEG).rotate([-90, 0, 0]).translate([x, y0, z])


def hexagon_y(af, y0, y1, x, z):  # hex prism along y (nut trap), across-flats af
    r = af / np.sqrt(3.0)
    return Manifold.cylinder(y1 - y0, r, r, 6).rotate([0, 0, 30]).rotate([-90, 0, 0]).translate([x, y0, z])


def frame(origin, u, h, a):
    """4x3 affine of a local frame (columns u, h, a) at origin — manifold3d transform."""
    return np.column_stack([u, h, a, origin])


def camera_axes():
    t = np.radians(CAM_TILT)
    a = np.array([0.0, -np.sin(t), np.cos(t)])   # view direction: along the flange axis, tilted outwards (-y)
    u = np.array([1.0, 0.0, 0.0])                # long side across the hand
    h = np.cross(a, u)                           # 23 mm side
    return u, h, a


def camera_centre():
    return np.array([0.0, -CAM_R, CAM_Z])


def local_box(sx, sy, sz, centre_local, T):
    return Manifold.cube([sx, sy, sz], True).translate(list(centre_local)).transform(T)


# ── parts ───────────────────────────────────────────────────────────────────────────────────────────
def collar_half(sign):
    """Half ring (sign -1: palm side, +1: back) with its two bolt lugs."""
    z0, z1 = COLLAR_Z
    g = SPLIT_GAP / 2.0
    ring = zcyl(R_OUT, z0, z1) - zcyl(R_IN, z0 - 1, z1 + 1)
    lugs = Manifold()
    for sx in (-1, 1):
        x0, x1 = (R_OUT - 3.0, R_OUT + LUG_W) if sx > 0 else (-R_OUT - LUG_W, -R_OUT + 3.0)
        lugs = lugs + box(x0, x1, -LUG_H, LUG_H, z0, z1)
    half = ring + lugs
    keep = box(-200, 200, g, 200, -100, 100) if sign > 0 else box(-200, 200, -200, -g, -100, 100)
    half = half ^ keep
    half = half - zcyl(R_IN, z0 - 1, z1 + 1)
    zc = 0.5 * (z0 + z1)
    for sx in (-1, 1):
        xb = sx * (R_OUT + LUG_W / 2.0 - 1.0)
        half = half - ycyl(M4_HOLE / 2.0, -50, 50, xb, zc)
        if sign > 0:   # screw head on the back half
            half = half - ycyl(M4_HEAD_D / 2.0, LUG_H - 4.0, 50, xb, zc)
        else:          # nut on the palm half
            half = half - hexagon_y(M4_NUT_AF, -50, -LUG_H + M4_NUT_H, xb, zc)
    return half


def camera_arm():
    """Back plate behind the D415 joined to the palm-side collar by a solid web; camera pocket and screw
    channels (Allen key access through the web) cut out."""
    u, h, a = camera_axes()
    C = camera_centre()
    T = frame(C, u, h, a)
    back = -CAM_D / 2.0 - PLATE_T / 2.0                   # plate centre (local a) behind the camera's back face
    plate = local_box(CAM_HOLES + 16.0, CAM_H, PLATE_T, [0, 0, back], T)
    z0, z1 = COLLAR_Z
    pad = box(-20.0, 20.0, -R_OUT - 1.0, -R_OUT + 3.0, z0, z1)
    web = Manifold.batch_hull([pad, plate])
    # lips around the camera's sides (holds it square while tightening)
    lip = (local_box(CAM_W + 2 * CAM_CLEAR + 5.0, CAM_H + 2 * CAM_CLEAR + 5.0, 6.0, [0, 0, -CAM_D / 2.0 + 3.0 - 1.0], T)
           - local_box(CAM_W + 2 * CAM_CLEAR, CAM_H + 2 * CAM_CLEAR, 20.0, [0, 0, 0], T))
    arm = web + lip
    arm = arm - local_box(CAM_W + 2 * CAM_CLEAR, CAM_H + 2 * CAM_CLEAR, CAM_D + 2 * CAM_CLEAR, [0, 0, 0], T)
    for s in (-1, 1):
        hole = Manifold.cylinder(PLATE_T + 2.0, M3_HOLE / 2.0, M3_HOLE / 2.0, SEG).translate(
            [s * CAM_HOLES / 2.0, 0.0, back - PLATE_T / 2.0 - 1.0])
        chan = Manifold.cylinder(120.0, M3_HEAD_D / 2.0, M3_HEAD_D / 2.0, SEG).translate(
            [s * CAM_HOLES / 2.0, 0.0, back - PLATE_T / 2.0 - 120.0])
        arm = arm - (hole + chan).transform(T)
    return arm - zcyl(R_IN, z0 - 50, z1 + 50)


def electronics_tray():
    """Tray on the back of the hand: board on standoffs, power switch in an open-ended cradle, zip-tie slots."""
    z0, z1 = TRAY_Z
    w2 = TRAY_W / 2.0
    y0, y1 = TRAY_Y, TRAY_Y + TRAY_T
    tray = box(-w2, w2, y0, y1, z0, z1)
    tray = tray + box(-w2, -w2 + 3.0, y0, y1 + RIB_H, z0, z1) + box(w2 - 3.0, w2, y0, y1 + RIB_H, z0, z1)
    cz0, cz1 = COLLAR_Z
    riser = box(-24.0, 24.0, R_OUT - 3.0, y1, cz0, cz1)            # onto the back half of the collar
    gusset = Manifold.batch_hull([box(-6.0, 6.0, R_OUT - 3.0, R_OUT, cz0, cz1),
                                  box(-6.0, 6.0, y0 - 0.01, y0, cz1, cz1 + 35.0)])
    tray = tray + riser + gusset
    # breakout board standoffs (M2.5 pilots)
    for dz, dx in BOARD_HOLES:
        s = ycyl(STANDOFF_D / 2.0, y1, y1 + STANDOFF_H, BOARD_X + dx, BOARD_Z + dz)
        tray = tray + s - ycyl(STANDOFF_PILOT / 2.0, y0 - 1.0, y1 + STANDOFF_H + 1.0, BOARD_X + dx, BOARD_Z + dz)
    # power switch cradle: side walls + rear stop, open towards +z (button, USB-C, LED)
    sw2 = SWITCH_W / 2.0 + 0.3
    sz0, sz1 = SWITCH_Z - SWITCH_L / 2.0, SWITCH_Z + SWITCH_L / 2.0
    wall_h = 0.6 * SWITCH_H
    for sx in (-1, 1):
        x0 = SWITCH_X + (sw2 if sx > 0 else -sw2 - CRADLE_WALL)
        tray = tray + box(x0, x0 + CRADLE_WALL, y1, y1 + wall_h, sz0, sz1)
    tray = tray + box(SWITCH_X - sw2 - CRADLE_WALL, SWITCH_X + sw2 + CRADLE_WALL, y1, y1 + wall_h,
                      sz0 - CRADLE_WALL - 0.3, sz0 - 0.3)
    # zip-tie slots through the tray, across the board and across the switch (ties run around them along x)
    for xc, zc, half_l in ((BOARD_X, BOARD_Z, BOARD_L / 2.0), (SWITCH_X, SWITCH_Z, SWITCH_L / 2.0)):
        for sz in (-1, 1):
            zs = zc + sz * (half_l - 6.0)
            for sx in (-1, 1):
                xs = xc + sx * ((BOARD_W if xc == BOARD_X else SWITCH_W) / 2.0 + 1.5 + (CRADLE_WALL if xc == SWITCH_X else 0.0))
                xs = float(np.clip(xs, -w2 + 4.5, w2 - 4.5))
                tray = tray - box(xs - ZIP_W / 2.0, xs + ZIP_W / 2.0, y0 - 1.0, y1 + 1.0, zs - ZIP_L / 2.0, zs + ZIP_L / 2.0)
    return tray - zcyl(R_IN, cz0 - 50, cz1 + 50)


def dummies():
    """Not printed: D415, board, switch volumes for the preview / clearance checks."""
    u, h, a = camera_axes()
    cam = local_box(CAM_W, CAM_H, CAM_D, [0, 0, 0], frame(camera_centre(), u, h, a))
    y1 = TRAY_Y + TRAY_T
    board = box(BOARD_X - BOARD_W / 2, BOARD_X + BOARD_W / 2, y1 + STANDOFF_H, y1 + STANDOFF_H + 1.6,
                BOARD_Z - BOARD_L / 2, BOARD_Z + BOARD_L / 2)
    switch = box(SWITCH_X - SWITCH_W / 2, SWITCH_X + SWITCH_W / 2, y1, y1 + SWITCH_H,
                 SWITCH_Z - SWITCH_L / 2, SWITCH_Z + SWITCH_L / 2)
    return cam, board, switch


def parts():
    palm_half = collar_half(-1) + camera_arm()
    back_half = collar_half(+1) + electronics_tray()
    return palm_half, back_half


def save(man, path):
    import trimesh
    mesh = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)
    tm.export(path)
    return tm


def main():
    OUT.mkdir(exist_ok=True)
    palm_half, back_half = parts()
    cam, board, switch = dummies()
    for name, man in (("collar_palm_half_D415", palm_half), ("collar_back_half_electronics", back_half)):
        tm = save(man, OUT / f"{name}.stl")
        print(f"{name}.stl: {tm.extents.round(1)} mm, {man.volume() / 1000:.1f} cm3, watertight {tm.is_watertight}, "
              f"genus {man.genus()}")
    save(palm_half + back_half + cam + board + switch, OUT / "assembly_preview.stl")
    u, h, a = camera_axes()
    print(f"D415 centre (tool0) {camera_centre().round(1)} mm, view {a.round(3)}; "
          f"colour optical frame: see ability_hand.yaml camera:")


if __name__ == "__main__":
    main()
