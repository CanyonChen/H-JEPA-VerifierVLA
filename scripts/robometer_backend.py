"""Pinned Robometer-4B inference using the official RBM model class.

Preprocessing mirrors robometer/data/collators/rbm_heads.py and progress conversion
uses robometer/models/utils.py at ROBOMETER_COMMIT. The adapter loads the complete
checkpoint directly with Transformers and uses SDPA for attention.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time

ROBOMETER_COMMIT = "352d160389daa964788de1ec933d1925f3a6de4f"
MODEL_ID = "robometer/Robometer-4B"
MODEL_REVISION = "beef63bc914c5c189329d49c6d712d96d632aa34"
PROCESSOR_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROCESSOR_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROCESSOR_FILES = ("chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
                   "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt")
SPECIAL_TOKENS = ("<|split_token|>", "<|reward_token|>", "<|pref_token|>", "<|sim_token|>", "<|prog_token|>")


def load_processor():
    from transformers import AutoProcessor
    from transformers.utils import is_offline_mode

    options = dict(padding_side="right", do_sample_frames=False, fix_mistral_regex=False)
    if is_offline_mode():
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(PROCESSOR_ID, revision=PROCESSOR_REVISION, local_files_only=True))
        # Transformers 4.57.2 needs a directory offline, but its tokenizer crashes if
        # it sees Qwen's model config.json. Load only the original processor assets.
        with tempfile.TemporaryDirectory(prefix="robometer-qwen-") as processor_dir:
            for name in PROCESSOR_FILES:
                asset = snapshot / name
                if asset.is_file():
                    shutil.copyfile(asset, Path(processor_dir) / name)
            processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True, **options)
    else:
        processor = AutoProcessor.from_pretrained(PROCESSOR_ID, revision=PROCESSOR_REVISION, **options)
    for token in SPECIAL_TOKENS:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def prepare_inputs(processor, frame_sequences, task):
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Match the official checkpoint's progress prompt and per-image progress tokens.
    prompt = (
        f"The task for the robot is '{task}'. Given the trajectory video, predict the task progress at each frame, "
        "how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting "
        "state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
    )
    conversations = []
    for sequence in frame_sequences:
        content = [{"type": "text", "text": prompt}]
        for frame in sequence:
            content.append({"type": "image", "image": Image.fromarray(frame)})
            content.append({"type": "text", "text": "<|prog_token|>"})
        conversations.append([{"role": "user", "content": content}])
    texts = [processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False, add_vision_id=True,
        enable_thinking=False, fps=1) for message in conversations]
    images, _, _ = process_vision_info(
        conversations, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True)
    return processor(text=texts, images=images, padding=True, truncation=False,
                     return_tensors="pt", do_resize=False)


def validate_loading(loading):
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError("Robometer checkpoint did not load completely: " + str(loading))
    # The released checkpoint contains an older auxiliary similarity head. The
    # pinned official RBM class uses the progress, success and preference heads.
    if any(not key.startswith("similarity_head.") for key in loading["unexpected_keys"]):
        raise ValueError("Unexpected Robometer checkpoint keys: " + str(loading["unexpected_keys"]))


class RobometerLabeler:
    def __init__(self):
        started = time.monotonic()
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig
        import yaml
        from robometer.configs.experiment_configs import ModelConfig
        from robometer.models import RBM

        if not torch.cuda.is_available():
            raise RuntimeError("Robometer pilot requires the selected CUDA GPU")
        torch.cuda.reset_peak_memory_stats()
        source = json.loads(importlib.metadata.distribution("robometer").read_text("direct_url.json"))
        if ROBOMETER_COMMIT not in source["url"]:
            raise ValueError("Install the pinned Robometer source with scripts/setup_robometer_env.sh")
        checkpoint = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["config.json", "config.yaml", "*.safetensors", "model.safetensors.index.json"]))
        configuration = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
        self.processor = load_processor()
        model_config = ModelConfig(**configuration["model"])
        model_config.use_unsloth = False
        config = AutoConfig.from_pretrained(str(checkpoint))
        if len(self.processor.tokenizer) != config.text_config.vocab_size:
            raise ValueError("Robometer tokenizer vocabulary does not match the checkpoint")
        self.model, loading = RBM.from_pretrained(
            str(checkpoint), config=config, processor=self.processor, tokenizer=self.processor.tokenizer,
            base_model_id=PROCESSOR_ID, model_config=model_config, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True)
        validate_loading(loading)
        self.model = self.model.to("cuda:0").eval()
        self.model.requires_grad_(False)
        self.provenance = dict(
            model_id=MODEL_ID, model_revision=MODEL_REVISION, source_commit=ROBOMETER_COMMIT,
            source_archive=source,
            processor_id=PROCESSOR_ID, processor_revision=PROCESSOR_REVISION,
            loading=loading, dtype="bfloat16", attention="sdpa", device_name=torch.cuda.get_device_name(0),
            config_sha256=hashlib.sha256((checkpoint / "config.yaml").read_bytes()).hexdigest(),
            weight_index_sha256=hashlib.sha256((checkpoint / "model.safetensors.index.json").read_bytes()).hexdigest(),
            package_versions={name: importlib.metadata.version(name) for name in (
                "torch", "torchvision", "transformers", "robometer", "qwen-vl-utils", "huggingface-hub")})
        self.load_seconds = time.monotonic() - started

    def predict(self, sequences, task):
        import torch
        from robometer.models.utils import convert_bins_to_continuous

        inputs = prepare_inputs(self.processor, sequences, task).to("cuda:0")
        with torch.inference_mode():
            output, _ = self.model(**inputs, sample_type="progress", timing_raw=None)
            progress = convert_bins_to_continuous(output.progress_logits["A"].float())[:, -1]
            success = torch.sigmoid(output.success_logits["A"].float())[:, -1]
        return progress.cpu().numpy(), success.cpu().numpy().reshape(-1)

    def metrics(self):
        import torch

        return dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    memory_scope="PyTorch memory in this labeler process, including model loading")
