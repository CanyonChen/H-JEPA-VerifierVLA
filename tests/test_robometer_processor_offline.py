"""Exercise the real pinned processor in a fresh offline process, without a model."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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
            source = Path(snapshot_download(PROCESSOR_ID, revision=PROCESSOR_REVISION, local_files_only=True))
        except LocalEntryNotFoundError:
            self.skipTest("Requires the pinned Qwen processor already in the local cache; no downloads in this test")
        program = """
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
import numpy as np
from scripts.robometer_backend import PROCESSOR_ID, PROCESSOR_REVISION, SPECIAL_TOKENS, load_processor, prepare_inputs
processor = load_processor()
assert len(processor.tokenizer) == 151674
assert processor.tokenizer.padding_side == 'right'
snapshot = Path(snapshot_download(PROCESSOR_ID, revision=PROCESSOR_REVISION, local_files_only=True))
original = json.loads((snapshot / 'tokenizer.json').read_text(encoding='utf-8'))
loaded = json.loads(processor.tokenizer.backend_tokenizer.to_str())
assert loaded['pre_tokenizer'] == original['pre_tokenizer']
assert processor.chat_template == json.loads((snapshot / 'chat_template.json').read_text(encoding='utf-8'))['chat_template']
assert processor.tokenizer.convert_tokens_to_ids(list(SPECIAL_TOKENS)) == list(range(151669, 151674))
frames = np.zeros((2, 4, 224, 224, 3), dtype=np.uint8)
frames[1] = 125
inputs = prepare_inputs(processor, frames, 'pick the bowl')
progress_token = processor.tokenizer.convert_tokens_to_ids('<|prog_token|>')
assert inputs['input_ids'].shape[0] == 2
assert ((inputs['input_ids'] == progress_token).sum(dim=1) == 4).all()
assert inputs['image_grid_thw'].shape[0] == 8
assert inputs['pixel_values'].isfinite().all()
digest = hashlib.sha256()
for name in sorted(inputs):
    digest.update(inputs[name].numpy().tobytes())
print('OFFLINE_PROCESSOR_OK', tuple(inputs['input_ids'].shape), len(processor.tokenizer), digest.hexdigest())
"""
        with tempfile.TemporaryDirectory() as cache:
            snapshot = Path(cache) / source.parent.parent.name / "snapshots" / PROCESSOR_REVISION
            snapshot.mkdir(parents=True)
            for file in source.iterdir():
                if file.suffix in {".json", ".txt"} and file.name != "config.json":
                    shutil.copyfile(file, snapshot / file.name)
            environment = dict(os.environ, HF_HUB_CACHE=cache, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            environment.pop("PYTHONPATH", None)
            results = []
            for has_config in (False, True):
                if has_config:
                    # These two fields in the pinned Qwen config trigger the 4.57.2 regression.
                    config = {"model_type": "qwen3_vl", "transformers_version": "4.57.0.dev0"}
                    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
                result = subprocess.run([sys.executable, "-B", "-c", program],
                                        cwd=Path(__file__).resolve().parents[1], env=environment,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, f"has_model_config={has_config}\n{result.stdout}{result.stderr}")
                self.assertIn("OFFLINE_PROCESSOR_OK", result.stdout)
                results.append(result.stdout.strip())
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
