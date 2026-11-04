#!/usr/bin/env python3
"""
The PSYONIC Ability Hand as Steve's gripper: taco_bandit's smart_hand_node (serial to the hand, slip / deformation
reflexes; slip-and-deformation-aware-hand-control) + steve_grasping's ability_hand_gripper (GripperCommand
/gripper_controller/gripper_cmd -> the tool config's synergies).

    ros2 launch steve_grasping ability_hand.launch.py                         # the robot's hand (HAND_PORT)
    ros2 launch steve_grasping ability_hand.launch.py port:=/dev/serial/by-id/...
    ros2 launch steve_grasping ability_hand.launch.py simulated_hand:=true    # no hardware (taco_bandit's mock hand)

Then e.g. the WBC on the robot with tool:=ability_hand, or by hand:
    ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 1.0}}"

The port is never auto-detected: smart_hand_node would take the first FTDI device, and on Steve that is the base's
relay board (ttyUSB0; the SICK scanners are ttyUSB1/2). Its /joint_states go to /hand/joint_states, away from the
arm's.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node

# the Ability Hand's FTDI TTL-232R cable on Steve (the base's own FTDI devices are the relay board and the scanners)
HAND_PORT = "/dev/serial/by-id/usb-FTDI_TTL232R_FTAM6SIZ-if00-port0"


def _hand_node(context):
    sim = C("simulated_hand").perform(context).lower() == "true"
    port = C("port").perform(context) or HAND_PORT
    inv = C("thumb_invert_feedback").perform(context).lower() == "true"
    params = {
        "port": port, "hand_side": C("hand_side").perform(context).lower(),
        "use_mock": sim, "require_real_hardware": not sim,
        "grasp_gate_enabled": True,              # grasp only while ability_hand_gripper arms it
        "thumb_invert_enabled": True, "thumb_invert_feedback": inv,
        "remote_fallback_enabled": False,        # no TCP hand bridge on Steve: a failed hand stays failed
        "fsm_handover_enabled": False,           # a robot grasp: do not let go when the held object slips
        # the grasp state machine squeezes fingers + thumb flexor; the thumb rotator stays on the synergy (its
        # constant grip current turned the thumb fully under the palm and trapped it against the fingers, 2026-11-04)
        "fsm_motor_mask": [1, 1, 1, 1, 1, 0],
        # the hand's hot/cold byte is also set while a finger is blocked (end stop, the thumb, a held object):
        # 0 A only after it stays set for 15 s instead of 0.5 s (a WBC pick: close + 15 cm lift + --hold_time) —
        # holds longer than that still go limp
        "hot_cold_grace_sec": 15.0,
        # no D term in the velocity loop: under load ~half the hand's replies are lost (2026-11-04), the velocity
        # estimate from the remaining positions is noisy and kd turned it into current spikes (up to 19 A asked)
        "kd": 0.0,
        "max_current_limit": float(C("max_current").perform(context)),
    }
    return [Node(package="taco_bandit", executable="controller", name="smart_hand_node", output="screen",
                 condition=IfCondition(C("launch_hand_node")), parameters=[params],
                 remappings=[("joint_states", "/hand/joint_states")])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="", description=f"serial port of the hand (empty: {HAND_PORT})"),
        DeclareLaunchArgument("hand_side", default_value="Right", choices=["Right", "Left", "right", "left"]),
        DeclareLaunchArgument("simulated_hand", default_value="False", choices=["True", "False", "true", "false"]),
        DeclareLaunchArgument("thumb_invert_feedback", default_value="true", choices=["true", "false"],
                              description="flip the thumb rotator's position / velocity in smart_hand_node's readout "
                                          "like its current command (positive = opposed); the bridge follows"),
        DeclareLaunchArgument("max_current", default_value="0.5",
                              description="smart_hand_node max_current_limit [A], all motors: lower = gentler grip "
                                          "and less load on the hand's supply (fewer lost replies)"),
        DeclareLaunchArgument("tool", default_value="ability_hand", description="steve_grasping tool config"),
        DeclareLaunchArgument("action", default_value="/gripper_controller/gripper_cmd"),
        DeclareLaunchArgument("launch_hand_node", default_value="true",
                              description="false: smart_hand_node runs elsewhere (e.g. on the robot, the hand on "
                                          "its USB)"),
        OpaqueFunction(function=_hand_node),
        Node(package="steve_grasping", executable="ability_hand_gripper", name="ability_hand_gripper",
             output="screen", parameters=[{"tool": C("tool"), "action": C("action"),
                                           "thumb_inverted": C("thumb_invert_feedback")}]),
    ])
