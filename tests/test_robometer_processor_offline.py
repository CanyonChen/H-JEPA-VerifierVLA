"""Exercise the real pinned processor in a fresh offline process, without a model."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.robometer_backend import PROCESSOR_ID, PROCESSOR_REVISION


class OfflineProcessorTest(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                            ("transformers", "huggingface_hub", "qwen_vl_utils", "torch")),
                         "Requires the Robometer processor dependencies")
    def test_cached_processor_and_image_token_alignment_offline(self):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot_download(PROCESSOR_ID, revision=PROCESSOR_REVISION, local_files_only=True)
        except LocalEntryNotFoundError:
            self.skipTest("Requires the pinned Qwen processor already in the local cache; no downloads in this test")
        program = """
import json
from pathlib import Path
from huggingface_hub import snapshot_download
import numpy as np
from scripts.robometer_backend import PROCESSOR_ID, PROCESSOR_REVISION, load_processor, prepare_inputs
processor = load_processor()
assert len(processor.tokenizer) == 151674
assert processor.tokenizer.padding_side == 'right'
snapshot = Path(snapshot_download(PROCESSOR_ID, revision=PROCESSOR_REVISION, local_files_only=True))
original = json.loads((snapshot / 'tokenizer.json').read_text(encoding='utf-8'))
loaded = json.loads(processor.tokenizer.backend_tokenizer.to_str())
assert loaded['pre_tokenizer'] == original['pre_tokenizer']
frames = np.zeros((2, 4, 224, 224, 3), dtype=np.uint8)
inputs = prepare_inputs(processor, frames, 'pick the bowl')
progress_token = processor.tokenizer.convert_tokens_to_ids('<|prog_token|>')
assert inputs['input_ids'].shape[0] == 2
assert ((inputs['input_ids'] == progress_token).sum(dim=1) == 4).all()
assert inputs['image_grid_thw'].shape[0] == 8
assert inputs['pixel_values'].isfinite().all()
print('OFFLINE_PROCESSOR_OK', tuple(inputs['input_ids'].shape), len(processor.tokenizer))
"""
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-B", "-c", program],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OFFLINE_PROCESSOR_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
