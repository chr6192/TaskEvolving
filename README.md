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