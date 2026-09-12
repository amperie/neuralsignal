# METR MALT Agent Monitoring Plan

Use `metr_malt_public`, backed by `metr-evals/malt-public` on Hugging Face. The dataset is gated, so the local Hugging Face account must accept the access terms before `datasets.load_dataset()` can read it.

MALT is the best immediate fit for adversarial agent-monitoring work because it contains labeled agent transcripts for reward hacking, sandbagging-like behavior, refusals, hardcoded solutions, bypassed constraints, and benign controls.

The first NeuralSignal experiment should be transcript-level indirect monitoring:

1. Convert each transcript into a compact ordered action narrative.
2. Ask the judge model whether the next action or recent trajectory shows monitor-worthy risk.
3. Capture the judge activations.
4. Train S1 models against MALT labels.

This does not yet prove NeuralSignal can catch a live incident before damage. It tests whether activation features from the judge can separate benign agent trajectories from known suspicious ones.

For a Hugging Face incident analogue, the useful target is not whether the agent eventually hacked a service. The product-relevant question is whether NeuralSignal can detect the trajectory before a consequential action is allowed.

Relevant label families:

- `bypass_constraints`: ignores supervision or task boundaries.
- `side_channel`: writes or uses files, comments, services, datasets, or shared state for coordination.
- `credential_behavior`: searches for, validates, copies, or transmits secrets.
- `persistence`: creates plugins, users, scheduled jobs, tokens, or durable services.
- `lateral_movement`: uses one environment as a launchpad into another.
- `unauthorized_model_path`: tries to reach a model outside the governed runtime.
- `reward_hacking`: optimizes the scoring path instead of the intended task.

The current dataset runner treats each row as one input/output pair. For agent monitoring, add a trajectory runner that preserves ordered turns and can evaluate prefixes. That runner should produce scan records with metadata for transcript id, source model, labels, prefix length, proposed next action, and intervention target.
