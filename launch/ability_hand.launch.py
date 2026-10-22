#!/usr/bin/env python3
"""
The PSYONIC Ability Hand as Steve's gripper: ah_ros_py's ah_node (serial to the hand,
github.com/ItsShriks/ability-hand-ros2) + steve_grasping's ability_hand_gripper (GripperCommand
/gripper_controller/gripper_cmd -> the tool config's synergies).

    ros2 launch steve_grasping ability_hand.launch.py                         # port auto-detected
    ros2 launch steve_grasping ability_hand.launch.py port:=/dev/ttyUSB0
    ros2 launch steve_grasping ability_hand.launch.py simulated_hand:=true    # no hardware (ah_wrapper's virtual hand)

Then e.g. the WBC on the robot with tool:=ability_hand, or by hand:
    ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 1.0}}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="", description="serial port of the hand (empty: auto)"),
        DeclareLaunchArgument("baud_rate", default_value="0", description="0: auto"),
        DeclareLaunchArgument("hand_side", default_value="Right", choices=["Right", "Left"]),
        DeclareLaunchArgument("simulated_hand", default_value="False", choices=["True", "False", "true", "false"]),
        DeclareLaunchArgument("tool", default_value="ability_hand", description="steve_grasping tool config"),
        DeclareLaunchArgument("action", default_value="/gripper_controller/gripper_cmd"),
        DeclareLaunchArgument("launch_ah_node", default_value="true",
                              description="false: ah_node runs elsewhere (e.g. on the robot, the hand on its USB)"),
        Node(package="ah_ros_py", executable="ah_node", name="ah_node", output="screen",
             condition=IfCondition(C("launch_ah_node")),
             parameters=[{"write_thread": False,          # the bridge streams the target at 100 Hz
                          "port": C("port"), "baud_rate": C("baud_rate"), "hand_side": C("hand_side"),
                          "js_publisher": True, "simulated_hand": C("simulated_hand")}]),
        Node(package="steve_grasping", executable="ability_hand_gripper", name="ability_hand_gripper",
             output="screen", parameters=[{"tool": C("tool"), "action": C("action")}]),
    ])
