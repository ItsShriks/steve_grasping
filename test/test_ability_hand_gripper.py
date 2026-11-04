"""ability_hand_gripper's conversion between the tool config's API degrees and smart_hand_node's radians."""

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("control_msgs")

from steve_grasping.ability_hand_gripper import from_node_rad, to_node_rad  # noqa: E402


def test_round_trip_and_thumb_rotator_sign():
    deg = np.array([25.0, 25.0, 25.0, 20.0, 30.0, -30.0])
    rad = to_node_rad(deg)
    assert np.allclose(rad[:5], np.deg2rad(deg[:5]))
    assert np.isclose(rad[5], np.deg2rad(30.0))        # opposed thumb: positive in smart_hand_node's frame
    assert np.allclose(from_node_rad(rad), deg)


def test_not_inverted_keeps_sign():
    deg = np.array([10.0, 20.0, 30.0, 40.0, 50.0, -60.0])
    assert np.allclose(to_node_rad(deg, thumb_inverted=False), np.deg2rad(deg))
    assert np.allclose(from_node_rad(np.deg2rad(deg), thumb_inverted=False), deg)
