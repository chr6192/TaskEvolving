THIS REPO IS FOR AGENTIC RL TRAINING

# 使用说明

## 外部依赖

| 路径 | 来源 | 锁定版本 |
| --- | --- | --- |
| `natural-questions` | `google-research-datasets/natural-questions` | `fb26a307` |
| `OpenJudge` | `agentscope-ai/OpenJudge` | `v0.2.2` (`33db7c4a`) |
| `verl` | `verl-project/verl` | `v0.4.1` (`8d9e350e`) |

三个外部仓库以 Git submodule 引入，版本由主仓库中的 gitlink commit 精确锁定。

### 新克隆

`$TASK_EVOLVING_REPO` 代表本主仓库的 Git URL（请替换为实际地址）：

```bash
git clone --recurse-submodules "$TASK_EVOLVING_REPO"
cd TaskEvolving  # 进入克隆后的主仓库根目录
```

### 已有工作区初始化

```bash
git submodule update --init --recursive --depth 1
```

### 应用 verl 定制补丁

初始化 submodule 后，在主仓库根目录执行：

```bash
git -C verl apply --ignore-space-change --check ../patches/verl/task-evolving-customizations.patch
git -C verl apply --ignore-space-change ../patches/verl/task-evolving-customizations.patch
```

应用补丁后，`verl` submodule 会显示为包含未提交修改（dirty）——这是预期状态，无需在 submodule 内部提交这些改动。

### 更新 submodule（natural-questions、OpenJudge）

以下示例以 tag 更新；`<new-tag>` 替换为目标 tag 名（如 `v0.2.3`）：

```bash
cd OpenJudge  # 或 natural-questions
git fetch origin tag <new-tag>
git checkout <new-tag>
cd ..
git add OpenJudge
git commit -m "chore: bump OpenJudge to <new-tag>"
```

`git fetch origin tag <new-tag>` 仅获取该 tag 指向的 commit，不会拉取任意 SHA；若需锁定未打 tag 的 commit，请显式 fetch 对应 ref（如 `git fetch origin refs/heads/<branch>` 或 `git fetch origin <full-commit-sha>`），再 `git checkout <full-commit-sha>`。

### 更新 verl（含补丁逆向与重应用）

`verl` 在应用定制补丁后为 dirty 状态，更新前必须先逆向移除补丁，checkout 干净上游后再检查并重应用补丁。以下命令均从主仓库根目录执行；`<new-tag>` 替换为目标 tag（如 `v0.4.2`）：

```bash
PATCH=patches/verl/task-evolving-customizations.patch

# 1. 确认补丁已应用，并逆向移除（使 verl 回到锁定 commit 的干净树）
git -C verl apply --ignore-space-change --reverse --check "../${PATCH}"
git -C verl apply --ignore-space-change --reverse "../${PATCH}"

# 2. 获取并 checkout 新 tag
git -C verl fetch origin tag <new-tag>
git -C verl checkout <new-tag>

# 3. 检查补丁是否仍适用于新版本；--check 失败则须先调整补丁，禁止强制 apply
git -C verl apply --ignore-space-change --check "../${PATCH}"
git -C verl apply --ignore-space-change "../${PATCH}"

# 4. 在主仓库提交新的 gitlink
git add verl
git commit -m "chore: bump verl to <new-tag>"
```

若第 3 步 `git apply --check` 失败，说明 `patches/verl/task-evolving-customizations.patch` 与新版本不兼容，须先在主仓库修订补丁使其通过 `--check`，再重新 apply；不要用 `git apply --reject`、`git checkout -- .` 或 `git reset --hard` 等方式强行覆盖。

### 长期定制

不要在 submodule 目录内直接提交长期定制代码。如需持续修改上游项目，应 fork 对应仓库并在 `.gitmodules` 中指向你的 fork，或将定制逻辑保留在主仓库（如 `patches/verl/` 下的补丁文件）。

# 关于训练

训练任务 workdir 应为主仓库根目录，`verl` 以 `pip install -e .` 安装为 editable package。

# 对于 verl 的二次开发原则

尽可能最小改动 verl 源码，避免以后合并最新 verl 代码时要频繁解决冲突。因此尽可能将新特性以 customize class 的方式接入；如果 verl 源码不支持 customize，则进行相应的更改，如 customized reward manager 仅对 `reward.py` 源码有几行代码的改动；`main_ppo.py` 和 `ray_trainer.py` 作为上层代码，如需修改可以考虑继承原 class 并直接调用。长期维护的定制应通过 fork 或补丁文件管理，而非在 submodule 内提交。

# 依赖安装

## verl 所需依赖

### 方案 1

```bash
pip install -r verl/requirements_sglang.txt && \
pip install vllm==0.8.4 && \
cd verl && pip install -e . && \
cd ..
```

### 方案 2

Vllm 0.8.5 会有一系列的 bugs，主要是跟 torch、torchvision 版本不一致的问题，因此修改 `setup.py` 中 `vllm==0.8.4` 之后：

```bash
cd verl && pip install -e .[vllm] && cd ..
```

（通过上述命令安装后，就不需要再单独 `cd verl && pip install -e .`）

## 如果遇到 flash-attn 的问题

```bash
conda install -c conda-forge flash-attn
```

如果 conda 装不上的话，用 `pip install flash-attn --no-build-isolation`

## reward、tool、log、trace 相关的依赖

在主仓库根目录执行：

```bash
pip install -r requirements.txt
```

## 需要通过 pip install -e . 安装 atomic_eval 和 Bfcl

`atomic_eval` 与 `berkeley-function-call-leaderboard`（Bfcl）不在本主仓库内，需另行 clone 后设置路径环境变量（请替换为实际绝对路径）：

```bash
export GORILLA_BFCLEADERBOARD_PATH=/path/to/gorilla/berkeley-function-call-leaderboard
export ATOMIC_EVAL_PATH=/path/to/atomic_eval

pip install -e "$GORILLA_BFCLEADERBOARD_PATH"
pip install -e "$ATOMIC_EVAL_PATH"
```

```bash
python -m experiment.export_complexity_exploration_review
python -m experiment.translate_review_csv --concurrency 32
```
