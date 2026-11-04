#!/usr/bin/env python3
"""
hand_cycle.py — open/close the Ability Hand continuously through the GripperCommand bridge
(steve_grasping ability_hand.launch.py must be running).

    python3 tools/hand_cycle.py                          # 0 -> 1 -> 0 ... until Ctrl+C
    python3 tools/hand_cycle.py --max 0.7 --pause 1.0    # only to 70 % closure, 1 s at each end
    python3 tools/hand_cycle.py --steps 10               # sweep in 10 steps each way (slow, smooth)
    python3 tools/hand_cycle.py --cycles 20              # stop after 20 open/close cycles

Each goal waits for the bridge's result (reached or stalled), so a blocked finger does not pile up
goals. If the hand does not get back open (the open goal stalls short of it, e.g. the thumb caught on a
finger), the cycle stops instead of closing it again. Ctrl+C opens the hand (closure --min) before exiting.
Run nothing else that sends gripper goals at the same time.
"""

import argparse
import time

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--action", default="/gripper_controller/gripper_cmd")
    ap.add_argument("--min", type=float, default=0.0, help="open end of the cycle (closure 0..1)")
    ap.add_argument("--max", type=float, default=1.0, help="closed end of the cycle (closure 0..1)")
    ap.add_argument("--steps", type=int, default=1, help="goals per half cycle (1: straight to the end)")
    ap.add_argument("--pause", type=float, default=0.5, help="wait at each end [s]")
    ap.add_argument("--cycles", type=int, default=0, help="0: forever")
    ap.add_argument("--timeout", type=float, default=10.0, help="per goal [s]")
    ap.add_argument("--open_tol", type=float, default=0.1,
                    help="the open goal must get within this closure of --min, else the cycle stops")
    a = ap.parse_args()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl+C keeps the context: open the hand
    node = Node("hand_cycle")
    client = ActionClient(node, GripperCommand, a.action)
    log = node.get_logger()
    if not client.wait_for_server(timeout_sec=5.0):
        log.error(f"no action server {a.action} (is ability_hand.launch.py running?)")
        return

    def go(pos):
        goal = GripperCommand.Goal()
        goal.command.position = float(pos)
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=a.timeout)
        handle = fut.result()
        if handle is None or not handle.accepted:
            log.error(f"goal {pos:.2f} not accepted")
            return None
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(node, res_fut, timeout_sec=a.timeout)
        if res_fut.result() is None:
            log.warn(f"goal {pos:.2f}: no result within {a.timeout:.0f} s")
            handle.cancel_goal_async()
            return None
        return res_fut.result().result

    lo, hi = float(np.clip(a.min, 0, 1)), float(np.clip(a.max, 0, 1))
    close = np.linspace(lo, hi, a.steps + 1)[1:]
    open_ = np.linspace(hi, lo, a.steps + 1)[1:]
    log.info(f"cycling {lo:.2f} <-> {hi:.2f} ({a.steps} step(s) each way), Ctrl+C to stop")
    n = 0
    try:
        go(lo)
        while rclpy.ok() and (a.cycles == 0 or n < a.cycles):
            for half in (close, open_):
                for p in half:
                    r = go(p)
                    if r is not None and r.stalled:
                        log.warn(f"stalled at closure {r.position:.2f} (target {p:.2f})")
                if half is open_ and (r is None or r.position > lo + a.open_tol):
                    got = "no result" if r is None else f"closure {r.position:.2f}"
                    log.error(f"the hand did not open ({got}, target {lo:.2f}): stopping, check the thumb / "
                              "fingers are free")
                    return
                time.sleep(a.pause)
            n += 1
            log.info(f"cycle {n} done")
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            log.info("opening the hand")
            go(lo)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
