#!/usr/bin/env python3
"""
ability_hand_gripper — the PSYONIC Ability Hand as the robot's gripper: a control_msgs/GripperCommand action
server (the interface steve_wbc's active perception and steve_manipulation use) on top of taco_bandit's
smart_hand_node (taco_bandit `controller`, slip-and-deformation-aware-hand-control), which talks to the hand over
serial and closes the grasp with its slip / deformation reflexes.

    ros2 launch steve_grasping ability_hand.launch.py                 # smart_hand_node + this bridge
    ros2 run steve_grasping ability_hand_gripper                      # bridge only (smart_hand_node running)
    ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 1.0}}"

GripperCommand position = closure, 0 (open synergy) .. 1 (close synergy), the synergies of the tool config
(config/tools/ability_hand.yaml, API degrees [index, middle, ring, pinky, thumb flexor, thumb rotator]) —
the same numbers the simulation uses. max_effort is not used (smart_hand_node limits the motor currents).

smart_hand_node runs with grasp_gate_enabled: its grasp state machine (touch -> approach -> hold, slip reflex)
may only take over while <hand_ns>/grasp_enable is true. A closing goal (closure >= touch.open_below) arms it, an
opening goal releases it — the node drops the grasp and position-controls the hand open. Without the gate the
hand would grasp whatever touches it during the approach and never let go of a held object on command.
The target goes to <hand_ns>/cmd_pos (Float32MultiArray, radians of the API degrees; the thumb rotator in the
node's frame — positive = opposed — with thumb_invert_feedback, the launch default) and, with grasp_enable, is
repeated at `rate`. Nothing is sent before the first goal (the hand stays as it is) unless open_on_start.
The goal ends
  reached_goal  every motor within tol_deg of its target
  stalled       the fingers stopped (no motor moved > 0.5 deg in stall_time, after stall_grace) before reaching
                it: an object is held (closing), or something blocks the hand; closing, also once the touch check
                reads held for stall_time (smart_hand_node squeezes a soft object, the fingers may still creep)
  aborted       no feedback from smart_hand_node (<hand_ns>/tactile_frame), or timeout

Grasp types (tool config grasp_types, e.g. wrap / pinch): a std_msgs/String on grasp_type_topic switches the
synergy the closure refers to (active_perception_wbc sends it when it chooses the grasp; an open hand pre-shapes
to the new type at once). Touch check (steve_grasping.hand_feedback): on grasp_state_topic, at 10 Hz, a
std_msgs/String with JSON {state: open | held | contact | blocked | empty, fingers, touch, closure, cmd_closure,
slip, deform, unstable, object_grasp} from the FSRs (tactile_frame touch_raw), the finger positions and
smart_hand_node's slip / deformation detector (<hand_ns>/tactile_event) — the WBC's success check on the robot.
Thresholds: tool config `touch:` (real). With touch.real.tare the FSRs are tared (hand_feedback.TouchTare):
their resting values are learnt whenever the hand is commanded open and its fingers are still, and the threshold
applies to the rise over them — send an open goal once after start so the tare is taken open.
"""

import threading
import time

import json

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32MultiArray, String

from steve_grasping.hand_feedback import TouchConfig, TouchTare, closure_of, finger_touch, grasp_state
from steve_grasping.hand_geometry import grasp_types, synergies
from steve_grasping.tools import load_tool

MOTORS = ("index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator")


def to_node_rad(deg, thumb_inverted=True):
    """API degrees -> smart_hand_node's cmd_pos / joint_position radians (thumb rotator flipped if inverted)."""
    rad = np.deg2rad(np.asarray(deg, float))
    if thumb_inverted:
        rad[5] = -rad[5]
    return rad


def from_node_rad(rad, thumb_inverted=True):
    """smart_hand_node's joint_position radians -> API degrees (inverse of to_node_rad)."""
    deg = np.rad2deg(np.asarray(rad, float))
    if thumb_inverted:
        deg[5] = -deg[5]
    return deg


class AbilityHandGripper(Node):
    def __init__(self):
        super().__init__("ability_hand_gripper")
        p = self.declare_parameter
        p("tool", "ability_hand")
        p("hand_ns", "/hand")                          # smart_hand_node's topics: <hand_ns>/cmd_pos, ...
        p("thumb_inverted", True)                      # = smart_hand_node thumb_invert_enabled and _feedback
        p("action", "/gripper_controller/gripper_cmd")
        p("rate", 10.0)                                # Hz target + grasp_enable repeat (smart_hand_node keeps
                                                       # the hand in API mode itself)
        p("tol_deg", 4.0)
        p("stall_time", 0.3)                           # s without motion -> stalled
        p("stall_grace", 1.0)                          # s after the goal before a stall counts: smart_hand_node's
                                                       # velocity loop ramps the current up first
        p("timeout", 6.0)                              # s per goal
        p("open_on_start", False)
        p("grasp_type_topic", "/ability_hand_gripper/grasp_type")
        p("grasp_state_topic", "/ability_hand_gripper/grasp_state")
        g = lambda n: self.get_parameter(n).value
        tool = load_tool(g("tool"))
        try:
            self.open, self.close = synergies(tool, None, "real")
        except ValueError:
            raise SystemExit(f"[ERROR] tool '{tool.name}' has no synergies.open / .close (API degrees)")
        self.tool, self.gtype, self.cmd = tool, None, 0.0
        self.touch_cfg = TouchConfig.from_tool(tool, "real")
        self.tare = TouchTare(self.touch_cfg.tare_alpha, self.touch_cfg.tare_cap) if self.touch_cfg.tare else None
        try:
            from psyonic_msgs.msg import TactileEvent, TactileFrame
        except ImportError:
            raise SystemExit("[ERROR] psyonic_msgs not found: put taco_bandit's psyonic_msgs, taco_bandit_msgs and "
                             "taco_bandit packages into the workspace's src/ and build them")
        ns = g("hand_ns").rstrip("/")
        self.thumb_inverted = bool(g("thumb_inverted"))
        self.tol, self.stall_time, self.timeout = g("tol_deg"), g("stall_time"), g("timeout")
        self.stall_grace = g("stall_grace")
        self.lock = threading.Lock()
        self.target = self.open.copy() if g("open_on_start") else None
        self.fb, self.fb_t, self.touch = None, -np.inf, None
        self.event, self.event_t = None, -np.inf
        self.still_ref, self.moved_t = None, time.monotonic()   # fingers at rest since moved_t (tare)
        cb = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Float32MultiArray, f"{ns}/cmd_pos", 10)
        self.arm_pub = self.create_publisher(Bool, f"{ns}/grasp_enable", 10)
        self.create_subscription(TactileFrame, f"{ns}/tactile_frame", self._frame_cb, 10, callback_group=cb)
        self.create_subscription(TactileEvent, f"{ns}/tactile_event", self._event_cb, 10, callback_group=cb)
        self.create_timer(1.0 / g("rate"), self._stream, callback_group=cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, g("grasp_type_topic"), self._type_cb, latched, callback_group=cb)
        self.state_pub = self.create_publisher(String, g("grasp_state_topic"), 10)
        self.create_timer(0.1, self._publish_state, callback_group=cb)
        from control_msgs.action import GripperCommand
        self.GripperCommand = GripperCommand
        ActionServer(self, GripperCommand, g("action"), execute_callback=self._execute, callback_group=cb,
                     goal_callback=lambda _: GoalResponse.ACCEPT, cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.get_logger().info(f"Ability Hand (smart_hand_node {ns}) as GripperCommand {g('action')}: 0 = open "
                               f"{self.open.tolist()}, 1 = close {self.close.tolist()} [deg]")

    # ── smart_hand_node topics ──────────────────────────────────────────────
    def _frame_cb(self, msg):
        raw = np.asarray(msg.touch_raw, float)
        if len(msg.joint_position) >= 6:
            fb, now = from_node_rad(msg.joint_position[:6], self.thumb_inverted), time.monotonic()
            with self.lock:
                self.fb, self.fb_t = fb, now
                if self.still_ref is None or np.max(np.abs(fb - self.still_ref)) > 1.0:
                    self.still_ref, self.moved_t = fb, now
        if self.tare is not None:
            # resting: commanded open, the fingers there and still for 0.5 s (nothing should touch the FSRs)
            with self.lock:
                now = time.monotonic()
                resting = (self.target is not None and self.cmd < self.touch_cfg.open_below
                           and now - self.fb_t < 0.5 and now - self.moved_t > 0.5)
            first = self.tare.n_rest == 0
            self.tare.update(raw, resting)
            if first and self.tare.n_rest:
                self.get_logger().info("touch tare: resting FSR values taken (hand open), per finger "
                                       f"{np.round(finger_touch(self.tare.base), 2).tolist()}")
        self.touch = raw

    def _event_cb(self, msg):
        with self.lock:
            self.event = dict(slip=bool(msg.slip), deform=bool(msg.deform), unstable=bool(msg.grasp_unstable),
                              object_grasp=bool(msg.object_grasp))
            self.event_t = time.monotonic()

    def _type_cb(self, msg):
        name = msg.data.strip() or None
        if name is not None and name not in grasp_types(self.tool):
            self.get_logger().warn(f"grasp type '{name}' not in the tool config: default synergy")
            name = None
        if name is not None and "real_synergy" not in grasp_types(self.tool)[name] \
                and "synergy" in grasp_types(self.tool)[name]:
            self.get_logger().warn(f"grasp type '{name}' has no real_synergy (its sim synergy is untested on the "
                                   "real hand): the real default synergy")
        o, c = synergies(self.tool, name, "real")
        with self.lock:
            self.gtype, self.open, self.close = name, o, c
            if self.target is not None:  # keep the commanded closure in the new synergy (open: pre-shape)
                self.target = o + self.cmd * (c - o)
        self.get_logger().info(f"grasp type {name or 'default'}: open {o.tolist()}, close {c.tolist()} [deg]")

    def state(self):
        """hand_feedback.GraspState of the hand now (None without position feedback)."""
        with self.lock:
            fb, fresh = (None if self.fb is None else self.fb.copy()), time.monotonic() - self.fb_t < 0.5
            o, c, cmd = self.open.copy(), self.close.copy(), self.cmd
        if fb is None or not fresh:
            return None
        touch = self.touch if self.tare is None else self.tare.apply(self.touch)
        return grasp_state(touch, closure_of(fb, o, c), cmd, self.touch_cfg)

    def _publish_state(self):
        st = self.state()
        if st is not None:
            d = dict(st.as_dict(), type=self.gtype)
            with self.lock:
                ev = self.event if time.monotonic() - self.event_t < 0.5 else None
            d.update(ev or dict(slip=False, deform=False, unstable=False, object_grasp=False))
            if self.tare is not None:   # touch above = the rise over rest; the raw sums and whether rest was seen open
                d.update(touch_raw=[float(t) for t in finger_touch(self.touch)], tared_open=self.tare.n_rest > 0)
            self.state_pub.publish(String(data=json.dumps(d)))

    def _send(self, target, armed):
        self.pub.publish(Float32MultiArray(data=[float(x) for x in to_node_rad(target, self.thumb_inverted)]))
        self.arm_pub.publish(Bool(data=bool(armed)))

    def _stream(self):
        with self.lock:
            target = None if self.target is None else self.target.copy()
            armed = self.cmd >= self.touch_cfg.open_below
        if target is not None:
            self._send(target, armed)

    def closure(self, pos):
        """Closure 0..1 of a hand position (mean over the motors that differ between the synergies)."""
        return closure_of(pos, self.open, self.close)

    # ── action ──────────────────────────────────────────────────────────────
    def _execute(self, goal_handle):
        cmd = goal_handle.request.command
        f = float(np.clip(cmd.position, 0.0, 1.0))
        armed = f >= self.touch_cfg.open_below
        with self.lock:
            target = self.open + f * (self.close - self.open)
            self.target, self.cmd = target, f
        self._send(target, armed)
        self.get_logger().info(f"closure {f:.2f} -> {np.round(target, 1).tolist()} deg"
                               f"{' (grasp armed)' if armed else ' (grasp released)'}")
        res = self.GripperCommand.Result()
        t0 = time.monotonic()
        last_move_t, last_pos, held_since = t0, None, None
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
                    st = self.state()
                    if f > 0.5 and st is not None:
                        self.get_logger().info(f"closed through: {st}")
                    goal_handle.succeed()
                    return res
                st = self.state() if armed else None
                held_since = (held_since or now) if st is not None and st.state == "held" else None
                if held_since is not None and now - held_since > self.stall_time:
                    # smart_hand_node holds the object (touch control: the fingers may still creep on a soft one)
                    res.stalled = True
                    self.get_logger().info(f"held at closure {res.position:.2f}: {st}")
                    goal_handle.succeed()
                    return res
                if now - t0 > self.stall_grace and now - last_move_t > self.stall_time:
                    res.stalled = True
                    st = self.state()
                    touch = "" if st is None else f"; touch check: {st}"
                    self.get_logger().info(f"stalled at closure {res.position:.2f} (worst motor "
                                           f"{MOTORS[int(np.argmax(err))]} {np.max(err):.0f} deg off{touch})")
                    goal_handle.succeed()
                    return res
            elif now - t0 > 1.0:
                self.get_logger().error("no feedback from smart_hand_node (<hand_ns>/tactile_frame)")
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
