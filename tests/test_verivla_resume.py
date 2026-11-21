"""Validation-only checks for full-batch ranking and an already finished checkpoint."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import Dataset

from scripts import b5_train


class ValidationWindows(Dataset):
    by_task = {0: [0, 1], 1: [2, 3]}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        state = torch.zeros(8)
        state[0] = [2, 0, 11, 21][index]
        target = torch.full((5,), float([0, 1, 10, 20][index]))
        mask = torch.ones(5, dtype=torch.bool)
        if index == 0:
            mask[-1] = False
        return torch.zeros(6, 1, 1), torch.zeros(5, 7), state, target, mask, index // 2


class FixedScores(torch.nn.Module):
    def rollout_kv_cached(self, visual, actions, states, task_id):
        return None, states[:, 0].reshape(1, -1, 1).expand(5, -1, -1)


class VerifierValidationTests(unittest.TestCase):
    def test_pairwise_validation_respects_tasks_masks_and_cross_microbatch_pairs(self):
        model = FixedScores()
        args = model, ValidationWindows()
        for micro in [1, 4]:
            with self.subTest(micro=micro):
                metrics = b5_train.validate(*args, micro, torch.device("cpu"), with_rank=True)
                self.assertEqual(metrics["valid_targets"], 19)
                self.assertEqual(metrics["ranking_pairs"], 9)
                self.assertEqual(metrics["validation_rank"], 6.0)
                self.assertAlmostEqual(metrics["validation_huber"], 13.5 / 19)

    def test_final_checkpoint_republishes_missing_report_without_model_or_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            directory = job / "joint40"
            directory.mkdir()
            episodes = [dict(task=i, count=1) for i in range(40)]
            index = dict(train=episodes, validation=episodes, source_sha256="example", suites=["example"])
            (directory / "features.json").write_text(json.dumps(index), encoding="utf-8")
            (directory / "last.pt").touch()
            (directory / "best.pt").touch()
            result = dict(status="COMPLETE", updates=2, validation_history=[dict(update=2, validation_rank=.1)])
            saved = dict(step=2, config=mock.ANY, result=result, best=.2)
            with mock.patch.object(torch, "load", return_value=saved), mock.patch.object(b5_train, "predictor") as model:
                b5_train.train("joint40", 1, 32, 0, job=job)
            model.assert_not_called()
            self.assertEqual(b5_train.read_json(directory / "training.json"), result)

    def test_partial_resume_restores_parameters_and_optimizer_at_next_batch(self):
        class ReachedNextBatch(Exception):
            pass

        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            directory = job / "joint40"
            directory.mkdir()
            episodes = [dict(task=i, count=1) for i in range(40)]
            index = dict(train=episodes, validation=episodes, source_sha256="example", suites=["example"])
            (directory / "features.json").write_text(json.dumps(index), encoding="utf-8")
            (directory / "last.pt").touch()
            saved = dict(config=mock.ANY, step=1, micro=2, best=None, changed=.01,
                         train_seconds=1., warm_seconds=0., warm_count=0,
                         model={"example": 1}, optimizer={"example": 2})
            expected = b5_train.epoch_batches(b5_train.Windows(episodes), 0)[1:]

            def inspect_next_batch(loader):
                self.assertEqual(list(loader.batch_sampler), expected)
                raise ReachedNextBatch

            with (mock.patch.object(torch, "load", return_value=saved),
                  mock.patch.object(b5_train, "predictor") as predictor,
                  mock.patch.object(torch.optim, "AdamW") as optimizer,
                  mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
                  mock.patch.object(b5_train, "prefetched", side_effect=inspect_next_batch),
                  self.assertRaises(ReachedNextBatch)):
                b5_train.train("joint40", 2, 32, 0, job=job)
            predictor.assert_called_once_with(40, pretrained=False, job=job.resolve())
            model = predictor.return_value.to.return_value.train.return_value
            model.load_state_dict.assert_called_once_with(saved["model"])
            optimizer.return_value.load_state_dict.assert_called_once_with(saved["optimizer"])
            optimizer.return_value.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
