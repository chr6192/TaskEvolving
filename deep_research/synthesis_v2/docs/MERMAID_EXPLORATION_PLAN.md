# Mermaid 驱动的开放任务合成闭环（改造规划）

本文件描述如何将当前 `synthesis_v2` 改造成如下闭环（不涉及 Skeleton），并预留 **mermaid_k 传给 agent** 的接口（第一阶段生成 trajectory 时先不强制按 mermaid_k 执行）：

- (gen) `query_k (+ mermaid_k 可选) → traj_k`
- (eval) `traj_k → score_k (+ details_k)`
- (reviser) `mermaid_k, traj_k (+ metadata 可选，含 evaluation_output) → mermaid_{k+1}, query_{k+1}, rubric_{k+1}`
- (converge) 使用 `score_k` 等判定收敛

> 约束：
> 1) Mermaid 必须可直接传给 agent（prompt 友好、可读、可控长度）。
> 2) reviser 输入：`mermaid_k, traj_k`；`evaluation_output` 作为 `metadata` 的一部分透传（未来可扩展 suggestion）。
> 3) 全流程不要涉及 Skeleton。
> 4) 第一阶段：生成 trajectory **先不按 Mermaid 约束执行**，但必须 **保留接口与数据流**，后续可一键打开。

---

## 一、现状（当前代码链路）

当前 `synthesis_v2` 单条样本默认 workflow 为 `SkeletonTrajectoryWorkflow`：

- `WorkflowInput(query)` → `SkeletonGenerator.generate_skeleton` → `TrajectoryGenerator.generate_trajectories_from_skeleton` → `WorkflowOutput` → `OpenJudgeEvaluator/ORMEvaluator` 打分

这与目标冲突点：

- **冲突点 A**：目标要求 **不走 Skeleton**。
- **冲突点 B**：目前没有 `mermaid_k` 作为 agent 执行约束，也没有 “traj → mermaid” 的抽取与迭代闭环。
- **冲突点 C**：`ComplexityExplorationV2` 仍是占位（reviser TODO），且没有 `info_k` 结构。

---

## 二、目标架构（模块与职责）

### 2.1 约定（命名与接口占位）

- **task_mermaid（mermaid_k）**：用于刻画 task 的“业务逻辑/流程结构”
  - 载体：Mermaid 文本（建议 flowchart）
  - 语义：表示一个**业务流程的有向无环图（DAG）**，节点是业务关键步骤，边是步骤之间的因果/依赖关系
  - 约束：需要 prompt 友好、可控长度、节点命名稳定、保证 DAG
- **query_k**：针对 `mermaid_k` 所描述的业务流程，从**起点到终点**提出的自然语言问题
- **rubric_k**：对 `mermaid_k` 中**重要节点**（关键步骤/关键里程碑）的验证规则集合，用于判断轨迹在这些节点上的完成质量（需符合 OpenJudge 的 rubric 接口）

> 备注：接口里可以预留一个可选参数 `metadata: dict = None` 用于透传上下文，但不需要把它当成独立概念/类来设计。

> 备注：rubric 为必选输出，且需符合 OpenJudge 的 rubric 接口。

### 2.2 三个核心模块

1) **Generator（gen）**
- 输入：`query_k` + `mermaid_k`（可空，表示初轮无约束）
- 输出：`traj_k`
- 第一阶段要求：**仅透传/存储 mermaid_k**，不强制注入 system prompt；后续阶段再打开“注入为执行约束/流程模板”。

2) **Evaluator（eval）**
- 输入：`traj_k`（未来可扩展 `rubric_k`）
- 输出：`score_k` + `details_k`
- 现阶段直接复用 OpenJudge 的 `scores/details` 作为评估产出；reviser 需要的信息一律从“现有评估产出”透传（不再额外提取/压缩 reason）

3) **Reviser（revise）**
- 输入：`mermaid_k` + `traj_k` + `metadata（可选）`（后续可加 rubric/suggestion）
- 输出：`mermaid_{k+1}` + `query_{k+1}` + `rubric_{k+1}`（必选，且需符合 OpenJudge rubric 接口）
- 要求：生成的 `mermaid_{k+1}` 必须可直接塞进 agent prompt（控制长度、节点数量、命名稳定）

---

## 三、需要修改/新增哪些文件

### 3.1 新增：不含 Skeleton 的 Workflow

新增文件（建议）：

- `deep_research/synthesis_v2/workflow/native/query_trajectory_workflow.py`
  - 名称建议：`QueryTrajectoryWorkflow`
  - 行为：从 `WorkflowInput.messages` 提取 query → 构造 `QueryData(metadata)` → 调用
    - `deep_research.synthesis.core.trajectory_generator.TrajectoryGenerator.generate_from_query(query_data, n_samples=1)`
  - 将 `TrajectoryData` 转为 `WorkflowOutput`

这样可以做到：**workflow 层完全不触碰 Skeleton**。

需要调整的引用点：
- `example_usage.py` 默认 workflow 由 `SkeletonTrajectoryWorkflow` 切换为 `QueryTrajectoryWorkflow`
- exploration（见 3.3）也使用该 workflow

### 3.2 修改：TrajectoryGenerator 预留接收 mermaid_k 作为 agent 约束的接口（当前开发阶段先完全不用 mermaid）

修改文件（建议）：

- `deep_research/synthesis/core/trajectory_generator.py`

改造方式（不涉及 Skeleton）：

- **当前开发阶段（Stage 0）**：trajectory 生成完全不使用 mermaid（不透传、不注入 prompt、不参与策略），先把闭环跑通：`query → traj → eval → revise`。

> 目标：现在先不用 mermaid；接口与落点保留，后续启用时 Mermaid 可确实传给 agent，成为 traj 生成约束来源。

### 3.3 改造：ComplexityExplorationV2 串起 gen/eval/revise

修改文件：

- `deep_research/synthesis_v2/complexity_exploration/orchestrator.py`

需要做的结构性改造：

- 将 “每轮仅 pipeline.process + score 判断” 改成明确三段：
  1) (gen) 用 `query_k + mermaid_k` 构造 `WorkflowInput(metadata)` → `pipeline.process()` → `traj_k`（来自 `result.output`）
  2) (eval) 直接复用 `result.evaluation`（包含 `scores` 与 `details`），不再区分“是否异常”，也不再从 `details` 提取 reason
  3) (revise) 调用 reviser：输入 `mermaid_k, traj_k`；将 `evaluation_output(=scores+details)` 作为 `metadata` 透传 → 输出 `mermaid_{k+1}, query_{k+1}, rubric_{k+1}`（rubric 为必选输出，且需符合 OpenJudge 的 rubric 接口约定）
- 收敛条件：使用 `score_k`（以及可选的 “mermaid diff 不再变化”）

> 注意：此处 reviser 不再仅仅“改 query”，而是 “改 Mermaid + 产出下一轮 query”。

### 3.4 新增：Reviser（Mermaid + Query 统一产出）

新增文件（建议）：

- `deep_research/synthesis_v2/complexity_exploration/mermaid_reviser.py`

接口建议：

- `async revise(mermaid_k: str | None, traj_k: TrajectoryData, metadata: dict | None = None) -> dict`
  - 返回 `{"mermaid": str, "query": str, "rubric": dict}`（rubric 为必选，结构需与 OpenJudge rubric 接口兼容）

实现建议：

- 使用结构化输出（Pydantic schema），避免 Mermaid/Query 产生无法解析的格式
- Mermaid 输出强约束：DAG、节点数上限、节点命名规则

### 3.5 评估产出的透传（InfoK 构造）

修改文件（建议）：

- `deep_research/synthesis_v2/evaluate/openjudge_evaluator.py`（可选）

最小改造：

- 不改 evaluator；在 orchestrator 内将 `result.evaluation` 的产出（`scores` + `details`）整体透传给 reviser 即可（不再新增/引入 reason 抽取逻辑）

---

## 四、运行时数据如何流动（无 Skeleton 版本）

### 4.1 每轮迭代的数据

- 输入：
  - `query_k`（字符串或 QueryData.content）
  - `mermaid_k`（可选；首次可为空）
- 生成：
  - `traj_k`（TrajectoryData / WorkflowOutput.messages + tool_calls + tool_responses）
- 评估：
  - `score_k`（来自 OpenJudge）
  - `details_k`（OpenJudge 原始 details，整体透传；不区分是否异常）
  - `evaluation_output`（建议直接使用 `result.evaluation`，包含 `scores` + `details`；作为 `metadata` 透传给 reviser）
- 修订：
- 输出 `mermaid_{k+1}`, `query_{k+1}`, `rubric_{k+1}`（rubric 必选，结构需符合 OpenJudge rubric 接口；`mermaid` 可作为下一轮 `plan` 的载体）

### 4.2 mermaid_k 的来源与作用（先不考虑传给 agent）

在当前阶段，我们只要求：

- `mermaid_k` **由上一轮的 trajectory/traj_k 抽取或修订得到**（即：Mermaid 始终从实际轨迹中产生或增量修改，而不是凭空写一份流程）。
- reviser 在接收到 `traj_k` 与评估产出后，产出新的 `mermaid_{k+1}`，并与 `query_{k+1}`, `rubric_{k+1}` 保持语义一致。
- 新的 `mermaid_{k+1}` 仅用于指导下一轮的 `query_{k+1}` 与 `rubric_{k+1}` 设计与生成，**暂不要求作为 agent 推理的硬约束 prompt 注入**。

---

## 五、收敛条件（最小可用）

建议最小收敛条件：

- `score_k >= quality_threshold`（阈值可沿用 config）
- 或达到 `max_iterations`

建议增强收敛条件（后续）：

- `mermaid_{k+1}` 与 `mermaid_k` diff 足够小（流程稳定）
- 最近 N 轮 score 提升 < epsilon（收益递减）

---

## 六、Mermaid 作为 prompt 约束的注意事项（必须考虑）

因为 Mermaid 要传给 agent，需考虑：

- **长度控制**：节点数、文本总字数上限（例如 15 节点 / 2000 字符）
  - reviser 端强约束输出
  - generator 端必要时做摘要/截断（保留关键节点 + 依赖）

- **命名稳定**：节点 id/name 需可复用，便于跨轮对齐（否则 reviser 难以“增量修改”）

- **可执行性**：每个节点应映射到“可观察的产物/证据”，否则 rubrics 难验证

- **DAG 保证**：reviser 输出必须无环（可在输出后做一次程序校验）

---

## 七、落盘与可追溯性（强建议）

在 `outputs/complexity_exploration/`（或 v2 的 writer output_dir）按迭代保存：

- `iteration_k.json`：{query_k, mermaid_k, rubric_k, score_k, details_k, traj_k(引用或路径)}
- `traj_k.jsonl`：完整 messages/tool_calls/tool_responses

这样可以回放每轮“流程约束 → 轨迹 → 评估 → 修订”的链路。

---

## 八、基于现有代码的具体改造点（最小可用落地）

这一节只讲“基于现有代码，改哪里、怎么改”，并以 **最小可用闭环** 为目标：`traj_k → mermaid_{k+1} → (query_{k+1}, rubric_{k+1})`，同时保留落盘可追溯性。

### 8.1 `deep_research/synthesis_v2/complexity_exploration/orchestrator.py`（核心编排：补齐 revise + 落盘）

现状：

- 已有：每轮构造 `WorkflowInput` → `pipeline.process()` → 得到 `PipelineResult`（含 `result.output` 与 `result.evaluation`）。
- 缺失：从本轮 `traj_k` 生成/修订 `mermaid_{k+1}`，并用它指导生成下一轮 `query_{k+1}` 与 `rubric_{k+1}`；同时缺少按迭代落盘。

建议改法（最小闭环）：

- **拿到 traj_k**：
  - 直接使用 `result.output`（`WorkflowOutput`）在 orchestrator 内重建 `TrajectoryData`（做法可复用 `OpenJudgeEvaluator._convert_to_trajectory(...)` 的逻辑），保证后续抽取 Mermaid 时有统一的 `TrajectoryData` 视图。
- **拿到 evaluation_output**：
  - 直接使用 `result.evaluation`（包含 `scores` + `details`），不做任何 reason 抽取、不区分异常与否；
  - 将其作为 `metadata['evaluation_output']` 透传给 reviser（当前阶段 reviser 的 prompt 不强依赖它）。
- **调用 reviser**：
  - 输入：`mermaid_k`（上一轮）、`traj_k`（本轮）、`metadata`（其中包含 `evaluation_output`）。
  - 输出：`mermaid_{k+1}`, `query_{k+1}`, `rubric_{k+1}`（rubric需符合 OpenJudge rubric 接口）。
- **把 rubric 带入下一轮评估链路**：
  - 现有评估链路（`OpenJudgeEvaluator` → `ORMEvaluator`）会从 `trajectory.metadata['rubrics']` 读取 rubric 并透传给 OpenJudge reward；因此在 orchestrator 构造下一轮 `WorkflowInput` 时，需继续将 `rubric_{k+1}` 写入 `workflow_input.metadata['rubrics']`（保持当前链路不改）。
- **落盘**（对应第七节）：
  - 每轮写 `iteration_k.json`：至少包含 `{query_k, mermaid_k, rubric_k, score_k, details_k, traj_k_path}`。
  - 写 `traj_k.jsonl`：保存本轮完整 messages/tool_calls/tool_responses（建议以 `TrajectoryData.to_dict()` 为主）。

### 8.2 `deep_research/synthesis_v2/evaluate/openjudge_evaluator.py`（rubric 透传与数据对齐点）

现状：

- `OpenJudgeEvaluator._convert_to_trajectory(...)` 会把 `context`（即 `WorkflowInput.metadata`）合并进 `TrajectoryData.metadata`。
- `deep_research/synthesis/core/orm_evaluator.py` 会读取 `trajectory.metadata['rubrics']` 并放进 `RewardSample.extra['rubrics']` 供 OpenJudge 使用。

因此要求：

- orchestrator 必须把 **OpenJudge 兼容的** `rubric_k` 放到 `WorkflowInput.metadata['rubrics']`（dict 形态），无需改 evaluator 代码即可让 rubric 进入 OpenJudge 评估。

### 8.3 `deep_research/synthesis_v2/complexity_exploration/mermaid_reviser.py`（新：统一产出 mermaid/query/rubric）

新增一个 reviser，职责是“从轨迹中产出/修订 DAG Mermaid，并生成下一轮 query 与 rubric”：

- **输入**：`mermaid_k`（可空）、`traj_k: TrajectoryData`、`metadata（可选；其中可包含 evaluation_output=scores+details）`。
- **输出**：`{"mermaid": str, "query": str, "rubric": dict}`。
- **约束**：
  - `mermaid` 必须是 DAG（flowchart），节点命名稳定、长度可控。
  - `query` 必须对应该 DAG 的起点到终点发问。
  - `rubric` 必须覆盖 DAG 的关键节点验证规则，且 **结构与 OpenJudge rubric 接口兼容**（能被现有 `trajectory.metadata['rubrics']` 透传并被 OpenJudge grader 正常消费）。

实现落地建议（在不大改现有 core 的前提下）：

- **Mermaid 抽取/修订策略**：
  - 第一版可以是“基于 traj_k 的工具调用/工具响应 + 最终回答”生成关键步骤节点与依赖；
  - 或先做启发式抽取，再交给 LLM 结构化输出（强 schema + DAG 校验）。
- **rubric 生成**：
  - 可复用 `deep_research.synthesis.core.query_reviser.QueryReviserAgent` 的 rubric 结构化产出能力（它已经能生成 `RubricsData`），再在 v2 侧将 `RubricsData.to_dict()` 作为 OpenJudge 接口兼容的 rubric dict 透传。

### 8.4 `deep_research/synthesis_v2/io/`（新：复杂度探索专用落盘 writer，可复用现有 writer 风格）

当前 `StreamingWriter` 主要面向 pipeline/batch 结果；复杂度探索建议新增一个轻量 writer（或在 orchestrator 内直接写文件）：

- `outputs/complexity_exploration/<evolution_id>/iteration_k.json`
- `outputs/complexity_exploration/<evolution_id>/traj_k.jsonl`

重点是：**每轮都能复盘** `traj_k → evaluation_output(作为metadata) → reviser → (mermaid_{k+1}, query_{k+1}, rubric_{k+1})`。

---

## 九、实施顺序（建议）

1) 先新增 `QueryTrajectoryWorkflow`，跑通 **无 Skeleton 的 query→traj→OpenJudge eval**。
2) 在 `TrajectoryGenerator` 加入 `business_process_mermaid` 的 system prompt 注入（先写死一个 Mermaid 测试）。
3) 在 `ComplexityExplorationV2` 串起：mermaid_k（初始为空）→ gen → eval → 调一个简易 reviser（先固定输出/或用 LLM 结构化输出）。
4) 再逐步增强：rubric 节点验证、mermaid diff 收敛。
