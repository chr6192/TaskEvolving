THIS REPO IS FOR AGENTIC RL TRAINING

# 使用说明

## 外部依赖
- `verl`: 
  - 来源: git@github.com:volcengine/verl.git
  - 版本: v0.4.0 (commit: xxx)

## 由subtree方式引入第三方RL训练框架verl

### 添加verl-origin
git remote add verl-origin git@github.com:volcengine/verl.git

### 设置稳定版v0.4.0为subtree
git subtree add --prefix=verl verl-origin main
（不使用--squash）

### 更新subtree
git subtree pull --prefix=verl git@github.com:volcengine/verl.git main 
git commit -m "chore: update verl to version-"

# 关于训练

训练任务workdir应为agentic-rl 根目录，verl以pip install -e .安装为editable package

# 对于subtree verl的二次开发原则
尽可能最小改动verl源码，避免以后merge最新verl代码时要频繁解决冲突。因此尽可能将新特性以customize class的方式接入，如果verl源码不支持customize，则进行相应的更改，如customized reward manger仅对reward.py源码有几行代码的改动；main_ppo.py和ray_trainer.py作为上层代码，如需修改可以考虑继承原class并直接调用。


# 依赖安装：
## verl所需依赖：
### 方案1:
pip install -r {PATH_TO_VERL}/requirements_sglang.txt && \
pip install vllm==0.8.4 && \
cd {PATH_TO_VERL} && pip install -e . && \

### 方案2:
Vllm 0.8.5会有一系列的bugs，主要是跟torch、torchvision版本不一致的问题,因此修改setup.py中vllm==0.8.4之后：
pip install -e .[vllm] && \
(通过上述命令安装后，就不需要cd verl && pip install -e . )

## 如果遇到flash-attn的问题：
conda install -c conda-forge flash-attn 
如果conda装不上的话，用pip install flash-attn --no-build-isolation

## reward、tool、log、trace相关的依赖：
cd agentic-rl && pip install -r requirements.txt

## 需要通过pip install -e .安装atomic_eval和Bfcl：
cd../../gorilla/berkeley-function-call-leaderboard && pip install -e . && \
cd ../../atomic_eval && pip install -e . && \


python -m experiment.export_complexity_exploration_review
python -m experiment.translate_review_csv --concurrency 32