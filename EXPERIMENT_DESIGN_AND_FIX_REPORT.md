# LLM Benchmark Reliability — 实验设计与问题修复报告

---

## 第一部分：实验设计总览

### 1. 项目目标

本项目研究：**LLM benchmark 评测结果中，有多少来自真实能力差异，有多少来自评测噪声。** 项目将噪声拆为三个来源，分别用三个实验刻画。

### 2. 实验框架

| 实验 | 研究问题 | 核心设计 | 问题数 | 模型数 | 变体数 | API 调用量 |
|------|----------|----------|--------|--------|--------|-----------|
| Exp1 | Prompt wording noise | 4 维度 prompt 扰动 (OFAT + 随机全因子) | 300 | 4 | 18 | ~43,200 |
| Exp2 | Test-set wording noise | 语义等价 paraphrase 重采样 | 150 | 4 | 4 (原始+3改写) | ~4,800/source |
| Exp3 | 题目级噪声识别与清洗 | 合并 Exp1+Exp2 计算 noise score，过滤后重算指标 | 300/150 | — | — | 无 API 调用 |

### 3. 共同设置

- **数据集**：ARC-Challenge 300 题、MMLU-Pro 300 题
- **模型**：LLaMA-3.1-8B、Qwen2.5-7B、Qwen3-32B、Qwen2.5-72B
- **推理接口**：OpenRouter API，temperature=0.0
- **Qwen3 特殊处理**：追加 `/no_think` 关闭 thinking 模式

### 4. Exp1：Prompt Perturbation

**操纵变量**（4 维度，全因子空间 3x3x3x2=54）：
1. Instruction（3 level）：`Choose the correct answer.` / `Select the best answer.` / `Which is correct?`
2. Answer Format（3 level）：只输出字母 / `Answer: [X]` / 先解释再答
3. Option Format（3 level）：`A. text` / `(A) text` / `A) text`
4. Framing（2 level）：无前缀 / `You are a knowledgeable assistant.`

**实际变体**：1 base + 7 OFAT + 10 随机 factorial = 18 个 variant

**两阶段运行**：Phase 1 max_tokens=200 → Phase 2 对 parse failure 重跑 max_tokens=1024

**分析指标**：准确率 mean/std/range、item flip rate、OFAT 主效应、OLS 回归交互分析、pairwise gap bootstrap CI、reversal frequency、rank distribution、noise score、规模效应分析

### 5. Exp2：Test-Set Resampling via Paraphrasing

**设计原则**：固定 prompt 为 base 版本（v00），只改变题目表述

**Paraphrase 生成**：
- 双 source：GPT-4o（外部模型，无评测偏差）+ Qwen2.5-72B（对照源）
- 单次 prompt 生成 3 个不同改写，temperature=0.7
- 只改写 question stem，选项和答案标签不变

**分析指标**：parse failure 统计、full-set 主分析（parse failure 计为错误）、clean subset 补充分析、准确率稳定性、item flip rate、pairwise gap（含 BH 校正）、reversal frequency、rank distribution、cross-source 比较、cross-experiment 方差分解

### 6. Exp3：High-Noise Item Analysis

**噪声公式**：`Noise(q) = 1 - |2c(q) - N(q)| / N(q)`
- c(q)=答对次数，N(q)=总评测次数
- Noise=0 表示完全一致（全对或全错），Noise=1 表示最大不一致

**过滤阈值**：删除 top 10% / 20% / 30% noisiest items

**分析内容**：过滤后重算 Exp1/Exp2 全部指标、noise correlation across models、noise vs difficulty、三源方差分解（prompt / sampling / test-set）、scale analysis

---

## 第二部分：发现的问题

### 问题总表

| # | 优先级 | 位置 | 问题描述 | 状态 |
|---|--------|------|----------|------|
| **P1** | **高** | `exp2/run_experiment2_async.py` | Qwen3 thinking 开关指令错误 `/nothink`（应为 `/no_think`） | **已修复** |
| **P2** | **高** | `exp2/run_experiment2_async.py` | max_tokens=128 过小，无 Phase 2 fallback，导致 MMLU 上 Qwen3 和 LLaMA parse failure 率 16-24% | **已修复** |
| **P3** | **高** | `exp3/run_experiment3.py` | 300 题 Exp1 与 150 题 Exp2 混合，后 150 题缺 Exp2 数据导致 noise score 系统性偏低 | **已修复** |
| **P4** | **高** | `exp3/run_experiment3.py` | Exp3 只取一个 Exp2 source（优先 gpt4o），不支持双源合并 | **已修复** |
| P5 | 中 | `exp3/analyze_experiment3.py` | Exp3 分析脚本也只取一个 source，不加载双源 | **已修复** |
| P6 | 中 | `exp1/run_experiment1_async.py:164` | answer parsing fallback 对 `with_explanation` 取最后字母、其他取第一个字母——解析策略随被测变体变化 | 需在论文说明 |
| P7 | 中 | `exp1/prompt_variants.py:81` | 18 个变体非正交设计，OLS 主效应可能混入交互效应 | 需在论文说明 |
| P8 | 中 | Exp1 vs Exp2 | Exp1 无多重比较校正，Exp2 有 BH 校正，标准不一致 | 需在论文说明 |
| P9 | 中 | 模型选择 | Qwen3-32B 与 Qwen2.5 系列不属于同一代际，影响 scaling 分析的参数量单调性假设 | 需在论文说明 |
| P10 | 中 | `exp3/run_experiment3.py` | Exp1 贡献 72 obs/题、Exp2 仅 16 obs/题，combined noise 主要反映 prompt 扰动 | 需在论文说明 |
| P11 | 中 | Exp2 | paraphrase 语义保真未验证（无人工审查或自动语义一致性检查） | 建议补充 |
| P12 | 中 | Exp2 | 取前 150 题（顺序采样，无随机化） | **已修复** |
| P13 | 低 | Exp3 | `Noise=0` 同时涵盖"稳定全对"和"稳定全错"——衡量的是稳定性而非质量 | 已在代码注释中说明 |
| P14 | 低 | Exp3 | "Var(sampling)" 实为题目池抽样方差而非模型随机性方差 | 建议改名 |
| P15 | 中 | `exp2/analyze_experiment2.py` | 旧版主分析依赖 clean subset，可能引入 selection bias | **已修复** |
| P16 | 低-中 | `exp3/visualize_experiment3.py` | 可视化脚本硬编码文件名，无法读取 shared150 等 tagged 结果 | **已修复** |

---

## 第三部分：已执行的修复

### 修复 1：Exp2 Qwen3 thinking 开关 + max_tokens + Phase 2 重试

**文件**：`exp2/run_experiment2_async.py`

**变更内容**：

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| Qwen3 soft switch | `/nothink`（无效指令） | `/no_think`（官方指令） |
| max_tokens | 128（单阶段） | Phase 1: 200, Phase 2: 1024（两阶段） |
| Parse failure 处理 | 无 fallback | Phase 2 自动重跑所有 parse failure |

**修复原理**：
- `/nothink` 不是 Qwen3 认可的 soft switch，模型可能仍处于 thinking 模式
- 128 tokens 在模型输出解释性内容时会被截断，导致答案丢失
- 新增的 Phase 2 与 Exp1 的两阶段设计对齐，确保 parse failure 经过大 token budget 重试

**验证**：实际检查 parse failure 样本，确认根因是模型生成长解释（step-by-step reasoning）后被 128 tokens 截断，而非 thinking mode `<think>` 标签问题（OpenRouter 已在服务端剥离）。

### 修复 2：Exp3 支持 shared_150 模式

**文件**：`exp3/run_experiment3.py`

**变更内容**：
- `compute_noise_scores()` 新增 `shared_only` 参数
- 当 `shared_only=True` 时，只保留同时拥有 Exp1 和 Exp2 数据的题目
- CLI 新增 `--shared-only` 选项

**修复原理**：
原设计中后 150 题只有 Exp1 数据（72 次观测），缺少 Exp2 的 test-set 扰动维度。在相同真实噪声水平下，观测维度更少的题倾向于获得更低的 noise score，系统性地偏向被保留在 removal set 中。`shared_only` 模式确保所有题在相同的信息条件下被比较。

### 修复 3：Exp3 支持双 source 策略

**文件**：`exp3/run_experiment3.py`、`exp3/analyze_experiment3.py`

**变更内容**：
- `load_exp2_results()` 新增 `exp2_source` 参数，支持 `gpt4o` / `qwen` / `both`
- 当 `exp2_source="both"` 时，合并两个 source 的所有 Exp2 观测
- CLI 新增 `--exp2-source` 选项
- `analyze_experiment3.py` 的 `load_exp2_dataframe()` 改为加载所有可用 source

**修复原理**：
原设计采用 fallback 策略（优先 gpt4o，找不到才用 qwen），实际只使用一个 source 的数据。这与 Exp2 本身已支持双源分析的设计不匹配。新的 `both` 模式将两个 source 的数据合并，最大化信息利用。

### 修复 4：Exp2 主分析改为 full set，clean subset 降级为补充分析

**文件**：`exp2/analyze_experiment2.py`、`exp2/visualize_experiment2.py`

**变更内容**：
- 新增 full-set 分析视图：保留全部题目，把 `parse_failure` 视为该次作答错误
- 原 `clean subset` 不删除，但移入 `clean_subset` 字段，作为敏感性分析结果保留
- `accuracy_summary`、`item_flip_rate`、`pairwise_gaps`、`reversals`、`rank_distribution` 现在默认基于 full set
- `two_layer_accuracy` 继续保留，用于并排展示 `full set` 与 `clean subset`
- `visualize_experiment2.py` 的主 accuracy 图改为读取 full-set 视图

**修复原理**：
- 旧设计把 parse failure 转化成整题删除，主分析对象实际上变成了“所有模型都能顺利解析的题”
- 新设计与 Exp1 的两阶段思路更一致：先尽量修复 parse failure，再在主分析中保留题目本身
- 这样可以把“知识错误”和“输出格式不稳”统一视为鲁棒性的一部分，同时仍保留 clean subset 作为补充检验

**结果解释变化**：
- 现在 Exp2 的主结果表示：模型在完整 paraphrase 题集上的稳定性
- `clean_subset` 结果表示：剔除所有 parse failure 后的敏感性分析
- 若两套结果一致，说明结论对解析问题不敏感；若不一致，则可明确定位偏差来自 parse failure

### 修复 5：Exp2 采样改为分层随机抽样

**文件**：`exp2/generate_paraphrases_gpt4o.py`

**变更内容**：
- `load_questions()` 从 `items[:150]`（顺序截取）改为分层随机抽样（seed=42）
- MMLU-Pro 按 `category` 分层，每类按比例抽取，保证学科分布与全集一致
- ARC 结构均匀（全是 4 选项），使用简单随机抽样

**修复原理**：
旧版顺序截取导致 MMLU 前 150 题的学科分布与全集显著偏离（biology 11 vs 应为 8，math 12 vs 应为 16，philosophy 12 vs 应为 8）。分层采样后各类别命中率与全集比例一致。

**注意**：此修复改变了 Exp2 的题目集合。已有的 paraphrase 数据和 Exp2 结果基于旧的前 150 题生成，重新运行 `generate_paraphrases_gpt4o.py` 后会得到不同的 150 题，需要重跑整个 Exp2 流水线。

### 修复 6：Exp3 可视化脚本支持 --noise-tag

**文件**：`exp3/visualize_experiment3.py`

**变更内容**：
- `load_noise()` 和 `load_analysis()` 使用模块级 `NOISE_TAG` 变量拼接文件名
- `main()` 新增 `--noise-tag` CLI 参数
- tagged 模式的图片输出到独立目录（如 `figures_exp3_shared150/`）

### 删除的受影响结果

以下文件因基于有缺陷的代码（max_tokens=128）生成，已被删除：

**Exp2 原始结果**（parse failure 率过高）：
| 文件 | parse failure 率 | 删除原因 |
|------|-----------------|----------|
| `exp2/exp2_mmlu_pro_qwen3-32b_gpt4o.json` | 23.7% (142/600) | max_tokens=128 截断 |
| `exp2/exp2_mmlu_pro_qwen3-32b_qwen.json` | 21.2% (127/600) | 同上 |
| `exp2/exp2_mmlu_pro_llama-3.1-8b_gpt4o.json` | 16.0% (96/600) | 同上 |
| `exp2/exp2_mmlu_pro_llama-3.1-8b_qwen.json` | 15.5% (93/600) | 同上 |

**Exp2 分析结果**（依赖上述有缺陷数据）：
- `exp2/analysis_exp2/analysis_arc.json`
- `exp2/analysis_exp2/analysis_mmlu.json`

**Exp3 全部结果**（依赖 Exp2 数据 + 使用旧的单源 fallback 逻辑）：
- `exp3/noise_data/noise_arc.json`
- `exp3/noise_data/noise_mmlu.json`
- `exp3/analysis_exp3/analysis_arc.json`
- `exp3/analysis_exp3/analysis_mmlu.json`
- `exp3/analysis_exp3/summary_table_exp3.csv`

**未删除的结果**（parse failure 率 <= 0.2%，数据可靠）：
- Exp1 全部结果（8 个文件，None rate 0-3.1%）
- Exp2 ARC 全部 8 个结果文件（parse failure 0-0.2%）
- Exp2 MMLU Qwen2.5-7B 和 Qwen2.5-72B 的 4 个结果文件（parse failure 0%）

---

## 第四部分：重新运行指南

修复代码后，需要重新采集删除的数据并重跑分析。

### 步骤 1：重跑 Exp2 缺失的 MMLU 数据（需要 API 调用）

```bash
cd exp2
python run_experiment2_async.py --model qwen32b --dataset mmlu --source both
python run_experiment2_async.py --model llama --dataset mmlu --source both
```

### 步骤 2：重跑 Exp2 分析

```bash
python analyze_experiment2.py
```

说明：
- 该脚本现在会同时产出 `full-set` 主分析和 `clean subset` 补充分析
- `analysis_exp2/analysis_*.json` 中顶层字段默认对应 full set，`clean_subset` 字段保存旧口径结果

### 步骤 3：重跑 Exp3（建议使用新的 shared-only + 双源模式）

```bash
cd ../exp3
python run_experiment3.py --shared-only --exp2-source both
python analyze_experiment3.py
```

如需同时生成全量 300 题版本（作为对照）：

```bash
python run_experiment3.py --exp2-source both
```

---

## 第五部分：仍需在论文中说明的限制

以下问题属于实验设计层面的固有限制，已通过代码修复无法完全消除，需要在论文 Limitations 节中明确说明：

1. **Exp1 设计矩阵非正交**（P7）：18 个变体不构成标准正交分数因子设计，OLS 回归的主效应估计可能混入交互效应。可通过计算 VIF 量化混淆程度。

2. **Exp1 answer parsing 策略随变体变化**（P6）：fallback 规则对 `with_explanation` 取最后匹配字母、对其他格式取第一个字母。解析策略本身与被测变量存在耦合。

3. **多重比较校正不一致**（P8）：Exp1 未做 p 值校正，Exp2 使用了 BH 校正，两组"显著性"结论基于不同标准。

4. **Qwen3-32B 代际差异**（P9）：Qwen3-32B 与 Qwen2.5-7B/72B 属于不同代际架构，以参数量为唯一横轴的 scaling 分析存在混淆。

5. **Combined noise score 中 Exp1 权重占优**（P10）：Exp1 贡献 72 obs/题 (82%)，Exp2 仅贡献 16 obs/题 (18%)。"高噪声题"的识别主要由 prompt 敏感性驱动。

6. **Paraphrase 语义保真未验证**（P11）：无人工审查或自动语义一致性检查，观测到的波动中可能混入 semantic drift。

7. **Exp2 前 150 题顺序采样**（P12）：`items[:150]` 无随机化，若原始文件按难度或学科排列则存在子样本偏差。

8. **Noise=0 的二义性**（P13）：全对和全错均得 noise=0，Exp3 筛除的是"不稳定题"而非"低质量题"，二者需严格区分。

9. **"Var(sampling)" 命名**（P14）：实际估计的是题目池重采样方差（bootstrap of question pool），非模型 stochastic variance（temperature=0 无重复调用）。建议改名为 Var(question_pool)。
