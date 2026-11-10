import unittest

import numpy as np
from openpi_client import msgpack_numpy

from scripts.check_libero_env import make_policy_observation


class LiberoObservationTest(unittest.TestCase):
    def test_camera_orientation_state_order_and_wire_roundtrip(self):
        external = np.zeros((224, 224, 3), dtype=np.uint8)
        wrist = np.zeros_like(external)
        external[0, 0] = [255, 0, 0]
        wrist[-1, 0] = [0, 255, 0]
        obs = {
            "agentview_image": external,
            "robot0_eye_in_hand_image": wrist,
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
            "robot0_eef_quat": np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
        }
        converted = make_policy_observation(obs, "pick up the bowl")
        decoded = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(converted))
        np.testing.assert_array_equal(decoded["observation/image"][-1, -1], [255, 0, 0])
        np.testing.assert_array_equal(decoded["observation/wrist_image"][0, -1], [0, 255, 0])
        np.testing.assert_allclose(decoded["observation/state"], [1, 2, 3, 0, 0, np.pi / 2, 0.02, -0.02])
        self.assertEqual(decoded["observation/image"].dtype, np.uint8)
        self.assertEqual(decoded["prompt"], "pick up the bowl")
        np.testing.assert_array_equal(obs["agentview_image"][0, 0], [255, 0, 0])

    def test_identity_quaternion_and_camera_resize(self):
        obs = {
            "agentview_image": np.ones((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }
        converted = make_policy_observation(obs, "test")
        self.assertEqual(converted["observation/image"].shape, (224, 224, 3))
        np.testing.assert_array_equal(converted["observation/state"], np.zeros(8))


if __name__ == "__main__":
    unittest.main()
