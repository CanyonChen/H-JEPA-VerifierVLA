"""Seeded batches using pi0.5's existing JAX sampler and LIBERO input/output transforms."""
def infer_candidates(policy, observation, seeds):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.model import Observation

    if not 1 <= len(seeds) <= 10 or any(not 0 <= int(seed) < 2**32 for seed in seeds):
        raise ValueError("Invalid candidate seeds")
    inputs = policy._input_transform(jax.tree.map(lambda x: x, observation))
    batched = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x), (len(seeds), *np.shape(x))), inputs)
    shape = (policy._model.action_horizon, policy._model.action_dim)
    noise = jax.vmap(lambda seed: jax.random.normal(jax.random.PRNGKey(seed), shape))(jnp.asarray(seeds, dtype=jnp.uint32))
    raw = policy._sample_actions(jax.random.PRNGKey(0), Observation.from_dict(batched), noise=noise, **policy._sample_kwargs)
    raw = np.asarray(raw)
    outputs = [policy._output_transform({"state": np.asarray(inputs["state"]).copy(), "actions": sample.copy()})["actions"]
               for sample in raw]
    actions = np.stack(outputs)
    if actions.shape != (len(seeds), 10, 7) or not np.isfinite(actions).all():
        raise ValueError("Invalid batched pi0.5 output")
    return actions
