"""CPU-only checks for checkpoint provenance and transparent policy recording."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy

from scripts.serve_pi05_libero import RecordedPolicy, advance_policy_rng, checkpoint_manifest, warmup_policy


class Pi05ServiceTest(unittest.TestCase):
    def test_restored_jax_rng_matches_successive_upstream_splits(self):
        try:
            import jax
        except ImportError:
            self.skipTest("JAX is tested in the Python 3.11 model environment")
        with jax.default_device(jax.devices("cpu")[0]):
            expected = jax.random.key(0)
            for _ in range(17):
                expected, _ = jax.random.split(expected)
            policy = SimpleNamespace(_rng=jax.random.key(0))
            advance_policy_rng(policy, 17)
            np.testing.assert_array_equal(jax.random.key_data(policy._rng), jax.random.key_data(expected))

    def test_warmup_restores_sampling_key_and_starts_evaluation_at_request_one(self):
        class Policy:
            def __init__(self):
                self._rng = 0

            def infer(self, obs):
                self._rng += 1
                return {"actions": np.full((10, 7), self._rng, dtype=np.float32)}

        base = Policy()
        warm_response = warmup_policy(base, {})
        self.assertEqual(base._rng, 0)
        provenance = {"service_id": "test", "request_count": 0}
        response = RecordedPolicy(base, provenance).infer({})
        np.testing.assert_array_equal(response["actions"], warm_response["actions"])
        self.assertEqual(response["evaluation"]["request_id"], 1)
        self.assertEqual(base._rng, 1)

    def test_warmup_restores_sampling_key_when_inference_fails(self):
        class Policy:
            _rng = 0

            def infer(self, obs):
                self._rng = 1
                raise RuntimeError("warmup failed")

        policy = Policy()
        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            warmup_policy(policy, {})
        self.assertEqual(policy._rng, 0)

    def test_checkpoint_identity_is_path_independent_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            for root in (left, right):
                (root / "params").mkdir(parents=True)
                (root / "params/weights").write_bytes(b"weights")
                (root / "norm_stats.json").write_text("{}", encoding="utf-8")
            first = checkpoint_manifest(left)
            self.assertEqual(first, checkpoint_manifest(right))
            self.assertEqual(first["total_bytes"], 9)
            self.assertEqual(len(first["files"]), 2)
            (right / "params/weights").write_bytes(b"changed")  # Same byte count.
            self.assertNotEqual(first["sha256"], checkpoint_manifest(right)["sha256"])

    def test_empty_checkpoint_is_not_a_valid_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                checkpoint_manifest(Path(directory))

    def test_wrapper_preserves_actions_and_advances_one_request_at_a_time(self):
        class Policy:
            def __init__(self):
                self.calls = 0

            def infer(self, obs):
                self.calls += 1
                return {"actions": np.full((10, 7), self.calls, dtype=np.float32)}

        base = Policy()
        provenance = {"service_id": "test", "request_count": 0}
        policy = RecordedPolicy(base, provenance)
        packer = msgpack_numpy.Packer()
        for request in (1, 2):
            response = msgpack_numpy.unpackb(packer.pack(policy.infer({})))
            np.testing.assert_array_equal(response["actions"], np.full((10, 7), request))
            self.assertEqual(response["evaluation"], {"service_id": "test", "request_id": request})
            self.assertEqual(base.calls, request)
        decoded = msgpack_numpy.unpackb(packer.pack({"evaluation": provenance}))
        self.assertEqual(decoded["evaluation"]["request_count"], 2)


if __name__ == "__main__":
    unittest.main()
