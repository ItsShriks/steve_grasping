#!/usr/bin/env python3
"""
touch_calib.py — record the Ability Hand's per-finger touch sums and closure for one situation, to
calibrate the touch check (config/tools/ability_hand.yaml `touch:`: real.threshold, closed_frac).
Reads /ability_hand_gripper/grasp_state (steve_grasping ability_hand.launch.py must be running).

    python3 tools/touch_calib.py --label empty_open   --seconds 10    # hand open, nothing touching
    python3 tools/touch_calib.py --label empty_cycle  --seconds 30    # with tools/hand_cycle.py running
    python3 tools/touch_calib.py --label ball         --seconds 10    # closed (1.0) on the ball
    python3 tools/touch_calib.py --suggest                            # threshold / closed_frac from the runs

The touch values are what the check compares with the threshold: with touch.real.tare the rise over each
sensor's resting value (send an open goal first so the tare is taken). Each run is appended to --out (JSON lines). --suggest takes the "empty*" runs as no-contact and every
other run as holding: threshold between the highest empty value and the typical held value, and
closed_frac between the closure the held object stops the fingers at and an empty close.
"""

import argparse
import json
import os
import time

import numpy as np

FINGERS = ["index", "middle", "ring", "pinky", "thumb"]


def record(a):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("touch_calib")
    rows = []

    def cb(msg):
        d = json.loads(msg.data)
        rows.append((d["touch"], d["closure"], d["cmd_closure"], d["state"]))

    node.create_subscription(String, a.topic, cb, 10)
    print(f"[{a.label}] recording {a.seconds:.0f} s from {a.topic} ...")
    t_end = time.monotonic() + a.seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if not rows:
        print("no messages: is ability_hand.launch.py running, and has the hand had a goal yet?")
        return
    touch = np.array([r[0] for r in rows])
    closure = np.array([r[1] for r in rows])
    cmd = np.array([r[2] for r in rows])
    states = {s: sum(1 for r in rows if r[3] == s) for s in {r[3] for r in rows}}
    print(f"{len(rows)} samples, states {states}")
    print(f"{'finger':8s} {'min':>7s} {'median':>7s} {'p95':>7s} {'max':>7s}")
    for i, f in enumerate(FINGERS):
        t = touch[:, i]
        print(f"{f:8s} {t.min():7.2f} {np.median(t):7.2f} {np.percentile(t, 95):7.2f} {t.max():7.2f}")
    closed = cmd > 0.5
    if closed.any():
        frac = closure[closed] / cmd[closed]
        print(f"closure / commanded while commanded > 0.5: median {np.median(frac):.3f}, "
              f"min {frac.min():.3f}, max {frac.max():.3f}")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(label=a.label, touch=touch.tolist(), closure=closure.tolist(),
                                 cmd=cmd.tolist())) + "\n")
    print(f"appended to {a.out}")


def suggest(a):
    if not os.path.exists(a.out):
        print(f"{a.out} does not exist: record some runs first")
        return
    runs = [json.loads(line) for line in open(a.out) if line.strip()]
    empty = [r for r in runs if r["label"].startswith("empty")]
    held = [r for r in runs if not r["label"].startswith("empty")]
    print("runs:", ", ".join(f"{r['label']} ({len(r['touch'])})" for r in runs))
    if not empty or not held:
        print("need at least one 'empty*' run and one holding run")
        return
    e = np.concatenate([np.array(r["touch"]) for r in empty])
    h_rows = [np.array(r["touch"])[np.array(r["cmd"]) > 0.5] for r in held]
    h = np.concatenate([x for x in h_rows if len(x)]) if any(len(x) for x in h_rows) else None
    e_max = float(e.max())
    print(f"empty: highest finger sum {e_max:.2f} (per finger max: "
          f"{', '.join(f'{f} {v:.2f}' for f, v in zip(FINGERS, e.max(0)))})")
    if h is None:
        print("no holding samples with the hand commanded closed (> 0.5)")
        return
    # a finger that touches the object: its median while held; untouched fingers (e.g. the pinky on a
    # small ball) stay near the empty level and are left out
    med = np.median(h, axis=0)
    touching = med > 2 * max(e_max, 1e-6)
    print(f"held: median per finger {', '.join(f'{f} {v:.2f}' for f, v in zip(FINGERS, med))}")
    if not touching.any():
        print("no finger is clearly above the empty level while holding: check the FSR topic / the grasp")
    else:
        lo_held = float(med[touching].min())
        thr = float(np.sqrt(max(e_max, 1e-3) * lo_held))   # geometric middle: robust to the scale
        thr = max(thr, 1.5 * e_max)
        print(f"weakest touching finger while held: {lo_held:.2f}")
        print(f"-> touch.real.threshold: {thr:.2f}   (empty max {e_max:.2f}, held {lo_held:.2f})")
    ec = [np.array(r["closure"])[np.array(r["cmd"]) > 0.5] / np.array(r["cmd"])[np.array(r["cmd"]) > 0.5]
          for r in empty]
    hc = [np.array(r["closure"])[np.array(r["cmd"]) > 0.5] / np.array(r["cmd"])[np.array(r["cmd"]) > 0.5]
          for r in held]
    ec = np.concatenate([x for x in ec if len(x)]) if any(len(x) for x in ec) else None
    hc = np.concatenate([x for x in hc if len(x)]) if any(len(x) for x in hc) else None
    if ec is not None and hc is not None:
        # an empty close: the highest closure fractions (fingers at the end); held: where the object stops them
        e_hi, h_med = float(np.percentile(ec, 90)), float(np.median(hc))
        print(f"closure fraction: empty close reaches {e_hi:.3f}, the held object stops at {h_med:.3f}")
        if e_hi > h_med:
            print(f"-> closed_frac: {(e_hi + h_med) / 2:.2f}")
        else:
            print("   the object does not stop the fingers short of an empty close: keep closed_frac")
    else:
        print("closed_frac: need closed samples in both an empty and a holding run")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--topic", default="/ability_hand_gripper/grasp_state")
    ap.add_argument("--out", default=os.path.expanduser("~/ros_logs/touch_calib_tared.jsonl"))
    ap.add_argument("--suggest", action="store_true")
    a = ap.parse_args()
    suggest(a) if a.suggest else record(a)


if __name__ == "__main__":
    main()
