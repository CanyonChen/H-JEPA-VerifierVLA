"""Numerical and input-isolation checks for the small Robometer context probe."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.label_verifier_long_pilot import run_pilot
from scripts.prepare_verifier_long_data import sha256_file
from scripts.probe_robometer_context import MODEL_FIELDS, crossed_prefixes, decompose, prepare_probe, run_probe
from tests.test_robometer_pilot import FakeLabeler, make_fixture


class ContextProbeTest(unittest.TestCase):
    def test_crossed_inputs_change_only_the_selected_factor(self):
        self.assertEqual(crossed_prefixes([0, 70, 141, 212], [0, 71, 142, 213]), [
            [0, 70, 141, 212], [0, 70, 141, 213], [0, 71, 142, 212],
            [0, 71, 142, 213], [0, 70, 141, 212], [0, 71, 142, 213]])
        with self.assertRaisesRegex(ValueError, "future"):
            crossed_prefixes([0, 0, 1, 2], [0, 1, 3, 3])

    def test_components_sum_to_delta_and_expose_interaction(self):
        result = decompose(np.array([0.7, 0.68, 0.52, 0.49, 0.7, 0.49]))
        self.assertAlmostEqual(result["current_component"], -0.025)
        self.assertAlmostEqual(result["history_component"], -0.185)
        self.assertAlmostEqual(result["interaction"], -0.01)
        self.assertAlmostEqual(result["current_component"] + result["history_component"], result["rerun_delta"])
        self.assertEqual(result["within_batch_repeat_max_abs_error"], 0)

    def test_four_events_reuse_inputs_without_changing_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            split, _, _, frames = make_fixture(base)
            labeler = FakeLabeler()
            labeler.provenance = {key: "fixture" for key in MODEL_FIELDS}
            run = base / "pilot"
            with patch("scripts.label_verifier_long_pilot.read_external_video", return_value=frames), \
                    patch("scripts.label_verifier_long_pilot.plot_progress", return_value=b"fixture"):
                run_pilot(split, run, labeler)
            before = {path: sha256_file(path) for path in run.rglob("*") if path.is_file()}
            with patch("scripts.inspect_verifier_long_labels.read_external_video", return_value=frames):
                prepared = prepare_probe(run)
            labeler.calls = 0
            report = run_probe(prepared, base / "probe", labeler)
            self.assertEqual(labeler.calls, 4)
            self.assertEqual(report["evaluated_prefixes"], 24)
            for row in report["per_episode"]:
                # This fake model uses only the endpoint, so history has no effect.
                self.assertEqual(row["decomposition"]["history_component"], 0)
                self.assertEqual(row["saved_endpoint_max_abs_error"], 0)
                self.assertAlmostEqual(row["decomposition"]["rerun_delta"], row["source_event"]["delta"])
            self.assertEqual(before, {path: sha256_file(path) for path in before})
            labeler.provenance = dict(labeler.provenance, model_revision="changed")
            with self.assertRaisesRegex(ValueError, "model_revision"):
                run_probe(prepared, base / "mismatch", labeler)
            self.assertEqual(labeler.calls, 4)


if __name__ == "__main__":
    unittest.main()
