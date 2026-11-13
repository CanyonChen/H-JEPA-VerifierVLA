# LiLo-VLA Ultra-Long benchmark subset

Source: https://github.com/YY-GX/LiLo-VLA/tree/64aa29d77f5457afb3d1642cfc84534dc1e81fad

The Python bootstrap, LIBERO compatibility patches, suite registration, task/skill
JSON and three Ultra-Long BDDL files are copied without code changes from this
revision. LICENSE is the upstream MIT license. The benchmark registry also defines
Long++ suites; their assets and the LiLo-VLA policy, motion planner, checkpoints,
training code and deterministic init-state files are not included or used.

Dependencies for our client are locked in ../../envs/lilo. The upstream benchmark
pins Python >=3.10, NumPy 1.26.4, SciPy 1.15.3, PyYAML 6.0.2, PyTorch 2.2.0,
robosuite 1.4.1, MuJoCo 3.3.0, BDDL 1.0.1, robomimic 0.2.0 and easydict 1.9.
Our simulation client uses the CPU PyTorch wheel; JAX pi05 inference stays in the
project's existing model environment.

Our end-to-end adapter is in ../../scripts/lilo_protocol.py and eval_lilo.py.
Its prompt construction, total control budget, temporal ordering monitor and
episode-local height-baseline reset are explicit project choices. It is not the
LiLo-VLA method or a claim of bit-identical reproduction of its paper baselines.
