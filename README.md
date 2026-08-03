<div align="right">

English | [简体中文](README.zh-CN.md)

</div>

<div align="center">

# TaskEvolving

**Automatic Task Evolution for Deep Research, and the Agentic RL Training Stack to Train on It**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![verl](https://img.shields.io/badge/RL%20backend-verl-orange.svg)](https://github.com/verl-project/verl)

</div>

TaskEvolving is an end-to-end repository covering the two most critical stages of building deep-research agents:

- **Data Synthesis (Task Evolution)** — starting from a simple QA pair, an iterative multi-agent loop automatically evolves it into a deep-research task with professional depth and a verifiable rubric ([`deep_research/synthesis_v2`](deep_research/synthesis_v2)).
- **Agentic RL Training** — a multi-turn, tool-use reinforcement learning stack built on top of [verl](https://github.com/verl-project/verl), paired with rubric-grounded process rewards to train models to actually solve those tasks ([`deep_research/reward`](deep_research/reward), [`deep_research/dr_*.py`](deep_research)).

Both stages share the same task representation: every task is modeled as a **directed acyclic graph (DAG)** of atomic steps (evidential / analytical), where each node carries independently verifiable checkpoints from which a structure-weighted rubric is derived. This DAG/rubric representation is both the output of data synthesis and the direct input to reward computation during training, so that "constructing a task" and "solving a task" are held to the same verifiable standard.

## Table of Contents

- [How It Works](#how-it-works)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Data Synthesis: Task Evolution](#data-synthesis-task-evolution)
- [Agentic RL Training](#agentic-rl-training)
- [Experiments & Analysis Scripts](#experiments--analysis-scripts)
- [Customizing verl](#customizing-verl)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## How It Works

The data synthesis stage implements an **Explorer–Formalizer–Challenger** loop. Each round makes the task strictly harder while keeping it coherent and solvable:

- **Explorer** rolls out the current query in a real tool environment (search, web browsing, etc.) and returns an exploration trajectory `traj`.
- **Formalizer** organizes the task-relevant information from `traj` into a DAG (nodes are atomic research steps, edges are logical dependencies), then derives a structure-weighted rubric bottom-up from the leaves; every checkpoint attached to a node can be verified independently.
- **Challenger** looks at information that was already explored but not yet absorbed into the task graph, and uses logically-connected clues to draft the next, harder query — either widening the task with new research directions or deepening it with longer reasoning/retrieval chains.

Evolution stops automatically once both the node count and the depth of the DAG stay unchanged for two consecutive rounds, which prevents unbounded growth.

<p align="center">
  <img src="docs/assets/task_evolution_overview.png" alt="Explorer-Formalizer-Challenger task evolution loop" width="820">
</p>

Each round folds newly discovered knowledge (entity relations, timelines, cross-source facts, etc.) back into the task graph, so the final DAG tends to have substantial depth and breadth. Below is an example DAG produced by the evolution loop (nodes are atomic research steps, edges are logical dependencies):

<p align="center">
  <img src="docs/assets/example_task_dag.png" alt="Example DAG produced by task evolution" width="720">
</p>

The Agentic RL training stage reuses the exact same DAG/rubric representation: the reward function ([`deep_research/reward/openjudge_reward_function.py`](deep_research/reward/openjudge_reward_function.py)) uses an [OpenJudge](https://github.com/agentscope-ai/OpenJudge) rubric grader to score a model's multi-turn tool-use trajectory checkpoint-by-checkpoint, aggregates the scores by their DAG-derived weights, and feeds the result to the RL algorithm (PPO/GRPO, etc.) as a process reward signal.

## Repository Layout

```text
TaskEvolving/
├── deep_research/
│   ├── synthesis_v2/            # Data synthesis: the Task Evolution pipeline
│   │   ├── complexity_exploration/  # Explorer-Formalizer-Challenger orchestrator & reviser
│   │   ├── query_generation/        # Seed query generation from a source corpus
│   │   ├── workflow/                # Query -> Trajectory workflows and adapters
│   │   ├── core/                    # Pipeline, batching, LLM scheduling, data structures
│   │   ├── evaluate/                # OpenJudge-based trajectory evaluator
│   │   ├── mermaid/                 # Mermaid-based structured DAG representation
│   │   ├── prompts/                 # Prompts for Explorer / Formalizer / Challenger
│   │   ├── config/                  # synthesis_v2 runtime configs (yaml)
│   │   └── experiments/             # Batch-run entry points for data synthesis
│   ├── reward/                   # Training reward: OpenJudge-based rubric reward function/manager
│   ├── tool/                     # Search / web-scraping tool wrappers and tests
│   ├── config/                   # verl training configs (tool_config, etc.)
│   ├── dr_ray_trainer.py         # Custom trainer extending verl's RayPPOTrainer
│   ├── dr_completion_callback.py # Multi-turn tool-use completion callback
│   ├── dr_chat_scheduler_no_thinking.py  # Custom async chat scheduler
│   └── run_qwen3_async_mt_dr.sh  # Example training launch script
├── experiment/                   # Analysis, plotting, and evaluation-reproduction scripts
├── patches/verl/                 # verl customization patch (never committed inside the submodule)
├── natural-questions/            # [submodule] Seed QA corpus (NQ-Open)
├── OpenJudge/                    # [submodule] Rubric grading / evaluation framework
├── verl/                         # [submodule] Agentic RL training framework
└── requirements.txt
```

## Getting Started

### Clone and initialize submodules

This repository vendors three external projects as Git submodules; each is pinned to an exact commit via the gitlink stored in this repo:

| Path | Upstream | Pinned version |
| --- | --- | --- |
| `natural-questions` | `google-research-datasets/natural-questions` | `fb26a307` |
| `OpenJudge` | `agentscope-ai/OpenJudge` | `v0.2.2` (`33db7c4a`) |
| `verl` | `verl-project/verl` | `v0.4.1` (`8d9e350e`) |

Fresh clone (replace `$TASK_EVOLVING_REPO` with the actual Git URL):

```bash
git clone --recurse-submodules "$TASK_EVOLVING_REPO"
cd TaskEvolving
```

Initializing an existing checkout:

```bash
git submodule update --init --recursive --depth 1
```

### Apply the verl customization patch

After initializing submodules, from the repo root run:

```bash
git -C verl apply --ignore-space-change --check ../patches/verl/task-evolving-customizations.patch
git -C verl apply --ignore-space-change ../patches/verl/task-evolving-customizations.patch
```

After applying the patch, the `verl` submodule will show as dirty (uncommitted changes) — this is expected. Do not commit these changes inside the submodule.

<details>
<summary>Updating submodules / updating verl (patch reverse + reapply)</summary>

**Updating `natural-questions` / `OpenJudge`** (by tag; replace `<new-tag>` with the target tag, e.g. `v0.2.3`):

```bash
cd OpenJudge  # or natural-questions
git fetch origin tag <new-tag>
git checkout <new-tag>
cd ..
git add OpenJudge
git commit -m "chore: bump OpenJudge to <new-tag>"
```

`git fetch origin tag <new-tag>` only fetches the commit the tag points to. If you need to pin an untagged commit, fetch the explicit ref instead (e.g. `git fetch origin refs/heads/<branch>` or the full SHA), then `git checkout <full-commit-sha>`.

**Updating `verl`** (since `verl` is dirty after the customization patch is applied, you must reverse the patch first, check out the clean upstream tag, then re-check and reapply the patch. Run all commands from the repo root; replace `<new-tag>` with the target tag, e.g. `v0.4.2`):

```bash
PATCH=patches/verl/task-evolving-customizations.patch

# 1. Confirm the patch is applied, then reverse it (restore verl to the clean pinned commit)
git -C verl apply --ignore-space-change --reverse --check "../${PATCH}"
git -C verl apply --ignore-space-change --reverse "../${PATCH}"

# 2. Fetch and check out the new tag
git -C verl fetch origin tag <new-tag>
git -C verl checkout <new-tag>

# 3. Verify the patch still applies to the new version; if --check fails, fix the patch first — never force-apply
git -C verl apply --ignore-space-change --check "../${PATCH}"
git -C verl apply --ignore-space-change "../${PATCH}"

# 4. Commit the new gitlink in the main repo
git add verl
git commit -m "chore: bump verl to <new-tag>"
```

If step 3's `git apply --check` fails, the patch is incompatible with the new version. Revise `patches/verl/task-evolving-customizations.patch` in the main repo until it passes `--check`, then reapply it. Do not force it through with `git apply --reject`, `git checkout -- .`, or `git reset --hard`.

**Long-term customizations**: never commit long-lived customizations directly inside a submodule. If you need to maintain ongoing changes to an upstream project, either fork it and point `.gitmodules` at your fork, or keep the customization as a patch in the main repo (e.g. under `patches/verl/`).

</details>

### Install dependencies

```bash
pip install -r requirements.txt
```

`verl` dependencies (pick one option):

```bash
# Option 1
pip install -r verl/requirements_sglang.txt && \
pip install vllm==0.8.4 && \
cd verl && pip install -e . && cd ..

# Option 2 (avoids a series of torch/torchvision version mismatches
# introduced by vllm 0.8.5; after editing verl/setup.py to pin vllm==0.8.4)
cd verl && pip install -e .[vllm] && cd ..
```

If you run into `flash-attn` build issues:

```bash
conda install -c conda-forge flash-attn
# if conda can't resolve it:
pip install flash-attn --no-build-isolation
```

`atomic_eval` and `berkeley-function-call-leaderboard` (Bfcl) are not vendored in this repo. Clone them separately and point the following environment variables at your local checkouts:

```bash
export GORILLA_BFCLEADERBOARD_PATH=/path/to/gorilla/berkeley-function-call-leaderboard
export ATOMIC_EVAL_PATH=/path/to/atomic_eval

pip install -e "$GORILLA_BFCLEADERBOARD_PATH"
pip install -e "$ATOMIC_EVAL_PATH"
```

## Data Synthesis: Task Evolution

The data synthesis pipeline lives under [`deep_research/synthesis_v2`](deep_research/synthesis_v2); its runtime configuration ([`deep_research/synthesis_v2/config/synthesis_v2.yaml`](deep_research/synthesis_v2/config/synthesis_v2.yaml)) controls the model, concurrency, rubric grading, the max number of evolution rounds, and more.

Run task evolution over a batch of seed queries:

```bash
python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
  --language en

# Cap concurrent seed exploration and only process the first 20 seeds
python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
  --max-concurrent-seeds 3 \
  --limit 20

# Explicitly point at a seed file (overrides the language-based default dataset)
python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
  --input deep_research/synthesis_v2/data/nq_ch.jsonl \
  --language cn

# Resume from a checkpoint after an abnormal stop (hit max_iterations, error,
# process interruption, etc.) instead of restarting from scratch
python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
  --resume \
  --max-concurrent-seeds 2
```

Each seed's evolution is persisted under `outputs/complexity_exploration/<query_id>/`:

- `iterations.json` — the query, DAG (Mermaid representation), and rubric actually used in every round; used for both inspection and resuming.
- `traj_<iteration_id>.jsonl` — the full exploration trajectory (tool calls and responses) the Explorer produced that round.
- `reviser.json` — the Formalizer/Challenger inputs and outputs for every round, useful for debugging how the DAG evolved.

API keys are injected via environment variables: rubric grading prefers `DASHSCOPE_API_KEY` (falling back to `OPENAI_API_KEY`); everything else (exploration, trajectory generation, etc.) prefers `OPENAI_API_KEY` (falling back to `DASHSCOPE_API_KEY`).

## Agentic RL Training

Training jobs should be run with the repo root as the working directory, with `verl` installed as an editable package (`pip install -e .`). Training integrates with verl through the following custom components:

| Component | Role |
| --- | --- |
| [`deep_research/dr_ray_trainer.py`](deep_research/dr_ray_trainer.py) | Custom trainer extending `verl.trainer.ppo.ray_trainer.RayPPOTrainer` |
| [`deep_research/dr_completion_callback.py`](deep_research/dr_completion_callback.py) | Multi-turn `CompletionCallback` that parses tool calls and dispatches tool execution |
| [`deep_research/dr_chat_scheduler_no_thinking.py`](deep_research/dr_chat_scheduler_no_thinking.py) | Custom async chat scheduler |
| [`deep_research/reward/openjudge_reward_function.py`](deep_research/reward/openjudge_reward_function.py) | OpenJudge-grader-based rubric reward function (`compute_score_async`) |
| [`deep_research/reward/openjudge_reward_manager.py`](deep_research/reward/openjudge_reward_manager.py) | Reward manager bridging into verl's `DataProto` |

Example launch script ([`deep_research/run_qwen3_async_mt_dr.sh`](deep_research/run_qwen3_async_mt_dr.sh), GRPO + multi-turn tool use + async vLLM rollout):

```bash
bash deep_research/run_qwen3_async_mt_dr.sh
```

Key config knobs in the script: `custom_reward_function.path/name` points at the rubric reward function above; `actor_rollout_ref.rollout.multi_turn.completion_callback` / `chat_scheduler_class_name` point at the custom multi-turn tool-use components; `+trainer.custom_class_path` points at the custom trainer. Tool configs (search engines, web scraping, etc.) live under [`deep_research/config/tool_config/`](deep_research/config/tool_config).

## Experiments & Analysis Scripts

[`experiment/`](experiment) contains scripts for analyzing synthesis quality and model evaluation results — evolution-round/DAG-size statistics, per-round score curves across models, rubric score distributions, batch export and translation of evaluation results, and more. Use these as a reference for reproducing the analysis or extending it with new evaluation dimensions.

## Customizing verl

We keep changes to verl's source as small as possible, to avoid painful merge conflicts when pulling in future upstream releases:

- Prefer wiring new features in as customize classes (custom Trainer / RewardManager / CompletionCallback, all via subclassing).
- Only touch verl's source when it doesn't support customization at all, and keep the change minimal (e.g. `reward.py` only gained a few lines to support injecting a custom reward manager).
- If top-level entry points like `main_ppo.py` or `ray_trainer.py` need changes, prefer subclassing and calling the original class rather than editing the source file directly.
- Long-lived customizations are managed exclusively via a fork or a patch file under `patches/verl/` — never committed inside the submodule itself.

## Acknowledgements

TaskEvolving builds on top of these excellent open-source projects:

- [verl](https://github.com/verl-project/verl) — the agentic RL training framework this repo trains on.
- [OpenJudge](https://github.com/agentscope-ai/OpenJudge) — the rubric grading / LLM-as-judge framework used for both reward computation and synthesis-time evaluation.
- [Natural Questions (NQ-Open)](https://github.com/google-research-datasets/natural-questions) — the seed QA corpus that task evolution starts from.

## License

This repository is licensed under the [Apache License 2.0](LICENSE). Each submodule (`natural-questions`, `OpenJudge`, `verl`) is licensed under its own upstream terms.
