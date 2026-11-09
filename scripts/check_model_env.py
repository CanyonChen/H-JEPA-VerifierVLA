"""Check the shared model environment with small, randomly initialized models.

No checkpoint, dataset, simulator or task evaluation is used. --core-only is
for the local Windows CPU check; the server check also imports policy setup.
"""

import argparse
import importlib.metadata
import json
import os
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from torch import nn
    import transformers
    import torchvision

    from openpi.models import gemma
    from h_jepa.models import HJEPA, JEPA
    from h_jepa.models.jepa import ProjectedEncoder
    from h_jepa.models.predictors import ARPredictor
    from h_jepa.models.encoders.build_encoder import build_encoder
    from h_jepa.models.encoders.seq_encoder import SequenceEncoder
    from h_jepa.models.probers import build_prober
    from verifier import LatentWorldModelVerifier, VisionTransformer, VisionTransformerPredictorAC

    expected = {"torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.53.2",
                "numpy": "1.26.4", "jax": "0.5.3", "flax": "0.10.2"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, version in versions.items():
        assert version.split("+")[0] == expected[name], (name, version, expected[name])
    assert platform.python_version_tuple()[:2] == ("3", "11")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
        assert jax.default_backend() == "gpu", "JAX cannot access CUDA"

    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = torch.device(args.device)

    # openpi's actual two-expert Gemma backbone; the upstream dummy config
    # reduces parameter count while preserving attention and AdaRMSNorm code.
    backbone = gemma.Module(configs=[gemma.get_config("dummy")] * 2, embed_dtype="float32")
    embedded = [jnp.ones((1, 3, 64)), jnp.ones((1, 2, 64))]
    positions = jnp.arange(5)[None, :]
    mask = jnp.tril(jnp.ones((1, 5, 5), dtype=bool))
    (gemma_outputs, _), _ = backbone.init_with_output(
        jax.random.key(7), embedded, positions, mask,
        adarms_cond=[None, jnp.zeros((1, 64))],
    )
    assert [tuple(x.shape) for x in gemma_outputs] == [(1, 3, 64), (1, 2, 64)]
    assert all(np.isfinite(np.asarray(x)).all() for x in gemma_outputs)

    # Two H-JEPA levels: encode at distinct strides and predict at each level.
    levels = []
    for level, input_dim, action_dim, frames, stride in [(1, 8, 7, 4, 1), (2, 16, 16, 2, 2)]:
        levels.append(JEPA(
            encoder=ProjectedEncoder(nn.Linear(input_dim, 16)),
            predictor=ARPredictor(num_frames=frames, depth=1, heads=2, mlp_dim=32,
                                  input_dim=16, hidden_dim=16, dim_head=8),
            action_embed=nn.Linear(action_dim, 16), level=level, temporal_stride=stride,
            action_encoder=(SequenceEncoder(input_dim=16, output_dim=16, d_model=16,
                                            nhead=2, max_chunk=2, dropout=0.0)
                            if level == 2 else None),
        ))
    hjepa = HJEPA(levels).to(device)
    encoded = hjepa.encode_hierarchical({
        "pixels_level1": torch.randn(1, 4, 8, device=device),
        "action_level1": torch.randn(1, 4, 7, device=device),
    }, sparse_level1_encode=False)
    predictions = [hjepa.get_level(i).predict(encoded[f"embed_{i}"], encoded[f"action_{i}"])
                   for i in (1, 2)]
    assert [tuple(p.shape) for p in predictions] == [(1, 4, 16), (1, 2, 16)]
    sum(p.square().mean() for p in predictions).backward()
    assert all(torch.isfinite(p).all().item() for p in predictions)
    gradients = [p.grad for p in hjepa.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all().item() for g in gradients)

    # VeriVLA's image encoder, action-conditioned predictor, cached rollout
    # and candidate ranking, with two candidates and two sub-chunks each.
    encoder = VisionTransformer(img_size=(32, 32), patch_size=16, embed_dim=32,
                                depth=1, num_heads=4)
    predictor = VisionTransformerPredictorAC(
        img_size=(32, 32), patch_size=16, embed_dim=32, predictor_embed_dim=64,
        depth=1, num_heads=4, action_embed_dim=7, action_chunk_size=2,
        max_frames=4, num_tasks=1,
    )
    verifier = LatentWorldModelVerifier(encoder, predictor).to(device).eval()
    latent = verifier.encode_observation(torch.randn(1, 3, 32, 32, device=device))
    best, scores = verifier.rank_candidates(
        latent, torch.randn(2, 2, 2, 7, device=device),
        torch.zeros(7, device=device), use_kv_cache=True,
    )
    assert tuple(scores.shape) == (2,) and best in (0, 1)
    assert torch.isfinite(scores).all().item()

    policy_setup = "not checked (core-only)"
    if not args.core_only:
        from openpi.policies import policy_config
        from openpi.training import config

        baseline = config.get_config("pi05_libero")
        assert baseline.model.pi05 and callable(policy_config.create_trained_policy)
        policy_setup = "PASS"

    print(json.dumps({
        "python": platform.python_version(), "platform": platform.system(),
        "device": args.device, "versions": versions,
        "jax_devices": [str(d) for d in jax.devices()],
        "openpi_gemma_forward": "PASS", "openpi_policy_setup_import": policy_setup,
        "hjepa_two_level_forward_backward": "PASS", "verivla_cached_candidate_ranking": "PASS",
        "scope": "random-weight model compatibility; no checkpoint inference or benchmark",
    }, indent=2))


if __name__ == "__main__":
    main()
