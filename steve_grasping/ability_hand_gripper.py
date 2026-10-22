#!/usr/bin/env python3
"""
ability_hand_gripper — the PSYONIC Ability Hand as the robot's gripper: a control_msgs/GripperCommand action
server (the interface steve_wbc's active perception and steve_manipulation use) on top of ah_ros_py's ah_node
(github.com/ItsShriks/ability-hand-ros2), which talks to the hand over serial.

    ros2 launch steve_grasping ability_hand.launch.py                 # ah_node + this bridge
    ros2 run steve_grasping ability_hand_gripper                      # bridge only (ah_node running)
    ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 1.0}}"

GripperCommand position = closure, 0 (open synergy) .. 1 (close synergy), the synergies of the tool config
(config/tools/ability_hand.yaml, API degrees [index, middle, ring, pinky, thumb flexor, thumb rotator]) —
the same numbers the simulation uses. max_effort is not used (the hand's motors squeeze at their own limit).

The target is streamed to /ability_hand/<side>/target/position (ah_messages/Digits) at --rate: ah_node runs
without its write thread and the hand leaves API mode without a command for 0.3 s. Nothing is sent before the
first goal (the hand stays as it is) unless open_on_start. The goal ends
  reached_goal  every motor within tol_deg of its target
  stalled       the fingers stopped (no motor moved > 0.5 deg in stall_time) before reaching it: an object
                is held (closing), or something blocks the hand
  aborted       no position feedback from ah_node, or timeout
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from steve_grasping.tools import load_tool

MOTORS = ("index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator")


class AbilityHandGripper(Node):
    def __init__(self):
        super().__init__("ability_hand_gripper")
        p = self.declare_parameter
        p("tool", "ability_hand")
        p("side", "")                                  # default: tool config real.side
        p("action", "/gripper_controller/gripper_cmd")
        p("rate", 100.0)                               # Hz target stream (ah_node: >= 5 Hz, ideally ~200)
        p("tol_deg", 4.0)
        p("stall_time", 0.3)                           # s without motion -> stalled
        p("timeout", 4.0)                              # s per goal
        p("open_on_start", False)
        g = lambda n: self.get_parameter(n).value
        tool = load_tool(g("tool"))
        syn = tool.config.get("synergies", {})
        if "open" not in syn or "close" not in syn:
            raise SystemExit(f"[ERROR] tool '{tool.name}' has no synergies.open / .close (API degrees)")
        self.open, self.close = np.asarray(syn["open"], float), np.asarray(syn["close"], float)
        side = g("side") or tool.real.get("side", "right")
        try:
            from ah_messages.msg import Digits
        except ImportError:
            raise SystemExit("[ERROR] ah_messages not found: clone github.com/ItsShriks/ability-hand-ros2 into the "
                             "workspace's src/ and build it (colcon build --packages-select ah_messages ah_ros_py)")
        self.Digits = Digits
        self.tol, self.stall_time, self.timeout = g("tol_deg"), g("stall_time"), g("timeout")
        self.lock = threading.Lock()
        self.target = self.open.copy() if g("open_on_start") else None
        self.fb, self.fb_t, self.touch = None, -np.inf, None
        cb = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Digits, f"/ability_hand/{side}/target/position", 10)
        self.create_subscription(Float32MultiArray, f"/ability_hand/{side}/feedback/position", self._fb_cb, 10,
                                 callback_group=cb)
        self.create_subscription(Float32MultiArray, f"/ability_hand/{side}/feedback/touch", self._touch_cb, 10,
                                 callback_group=cb)
        self.create_timer(1.0 / g("rate"), self._stream, callback_group=cb)
        from control_msgs.action import GripperCommand
        self.GripperCommand = GripperCommand
        ActionServer(self, GripperCommand, g("action"), execute_callback=self._execute, callback_group=cb,
                     goal_callback=lambda _: GoalResponse.ACCEPT, cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.get_logger().info(f"Ability Hand ({side}) as GripperCommand {g('action')}: 0 = open "
                               f"{self.open.tolist()}, 1 = close {self.close.tolist()} [deg]")

    # ── ah_node topics ──────────────────────────────────────────────────────
    def _fb_cb(self, msg):
        if len(msg.data) >= 6:
            with self.lock:
                self.fb, self.fb_t = np.asarray(msg.data[:6], float), time.monotonic()

    def _touch_cb(self, msg):
        self.touch = np.asarray(msg.data, float)

    def _stream(self):
        with self.lock:
            target = None if self.target is None else self.target.copy()
        if target is not None:
            msg = self.Digits()
            msg.reply_mode = 0
            msg.data = [float(x) for x in target]
            self.pub.publish(msg)

    def closure(self, pos):
        """Closure 0..1 of a hand position (mean over the motors that differ between the synergies)."""
        d = self.close - self.open
        m = np.abs(d) > 1e-6
        return float(np.clip(np.mean((pos[m] - self.open[m]) / d[m]), 0.0, 1.0)) if m.any() else 0.0

    # ── action ──────────────────────────────────────────────────────────────
    def _execute(self, goal_handle):
        cmd = goal_handle.request.command
        f = float(np.clip(cmd.position, 0.0, 1.0))
        target = self.open + f * (self.close - self.open)
        with self.lock:
            self.target = target
        self.get_logger().info(f"closure {f:.2f} -> {np.round(target, 1).tolist()} deg")
        res = self.GripperCommand.Result()
        t0 = time.monotonic()
        last_move_t, last_pos = t0, None
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return res
            with self.lock:
                fb, fb_age = (None if self.fb is None else self.fb.copy()), time.monotonic() - self.fb_t
            now = time.monotonic()
            if fb is not None and fb_age < 0.5:
                if last_pos is None or np.max(np.abs(fb - last_pos)) > 0.5:
                    last_pos, last_move_t = fb, now
                err = np.abs(fb - target)
                fbk = self.GripperCommand.Feedback()
                fbk.position, fbk.reached_goal = self.closure(fb), bool(np.all(err < self.tol))
                goal_handle.publish_feedback(fbk)
                res.position = self.closure(fb)
                if np.all(err < self.tol):
                    res.reached_goal = True
                    goal_handle.succeed()
                    return res
                if now - t0 > 0.3 and now - last_move_t > self.stall_time:
                    res.stalled = True
                    touch = "" if self.touch is None else f", touch sum {float(np.sum(self.touch)):.0f}"
                    self.get_logger().info(f"stalled at closure {res.position:.2f} (worst motor "
                                           f"{MOTORS[int(np.argmax(err))]} {np.max(err):.0f} deg off{touch})")
                    goal_handle.succeed()
                    return res
            elif now - t0 > 1.0:
                self.get_logger().error("no position feedback from ah_node (/ability_hand/<side>/feedback/position)")
                goal_handle.abort()
                return res
            if now - t0 > self.timeout:
                self.get_logger().warn(f"timeout after {self.timeout:.1f} s")
                goal_handle.abort()
                return res
            time.sleep(0.01)
        return res


def main():
    rclpy.init()
    node = AbilityHandGripper()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
