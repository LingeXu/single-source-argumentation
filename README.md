# Meta CRAG-MM：多模态 RAG 基准挑战赛

> **上海大学 计算机工程与科学学院 · 人工智能专业 · 智能技术实训课程项目**  
> 赛题：[KDD Cup 2025 Meta CRAG-MM Challenge](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)

本项目是《智能技术实训》课程的 Task 1（单源增强）实践代码，基于竞赛官方提供的入门套件进行开发和调试。

---

## 目录

1. [比赛背景](#-比赛背景)
2. [比赛任务](#-比赛任务)
3. [项目结构](#-项目结构)
4. [快速开始](#-快速开始)
5. [Task 1 详解](#-task-1-详解)
6. [评估指标](#-评估指标)
7. [优化方向](#-优化方向)
8. [踩坑记录](#-踩坑记录)

---

## 📖 比赛背景

KDD Cup 是由 ACM 组织的年度国际数据挖掘和知识发现大赛（CCF-A 类），素有"大数据领域世界杯"之称。2025 年赛题 **Meta CRAG-MM Challenge** 聚焦于**多模态检索增强生成（MM-RAG）**，旨在解决视觉大语言模型（VLLM）在实际可穿戴设备应用中的**幻觉问题**。

CRAG-MM 数据集包含：
- **5K+ 图像**（含 3K 第一人称视角图像，由 RayBan Meta 智能眼镜采集）
- **14 个领域**：书籍、食品、车辆、购物、观光、宠物、时尚、自然等
- **4 种问题类型**：简单识别、多跳推理、比较汇总、逻辑推理

---

## 👨‍💻👩‍💻 比赛任务

| 任务 | 名称 | 目标 | 检索源 | 难度 |
|------|------|------|--------|------|
| Task 1 | 单源增强 | 测试 MM-RAG 基础回答生成能力 | 仅图像 KG API | ⭐ 1.0 |
| Task 2 | 多源增强 | 测试多信息源融合与噪声鲁棒性 | 图像 KG + 网页搜索 API | ⭐⭐ 1.2 |
| Task 3 | 多轮问答 | 测试上下文理解与对话连贯性 | 图像 KG + 网页搜索 API | ⭐⭐⭐ 1.5 |

---

## 🗂️ 项目结构

```
.
├── agents/
│   ├── base_agent.py                  # Agent 基类（所有 Agent 必须继承）
│   ├── task1_agent.py                 # ★ Task 1 Agent（MLX 版，适配 Mac Apple Silicon）
│   ├── mllm_rag_agent.py              # 多阶段验证 RAG Agent（需 CUDA GPU）
│   ├── rag_agent.py                   # 简单 RAG Agent 示例
│   ├── vanilla_llama_vision_agent.py  # 纯视觉语言模型 Agent
│   ├── random_agent.py                # 随机回答 Agent（测试评估流程用）
│   └── user_config.py                 # 选择使用哪个 Agent
├── local_evaluation.py                # 本地评估脚本（主入口）
├── crag_batch_iterator.py             # 批量数据迭代器
├── crag_image_loader.py               # 图片加载器
├── crag_web_result_fetcher.py         # 网页结果抓取器
├── utils.py                           # 工具函数
├── docs/                              # 文档
│   ├── search_api.md                  # 模拟搜索 API 文档
│   ├── dataset.md                     # 数据集结构说明
│   ├── baselines.md                   # 基线模型说明
│   └── submission.md                  # 提交指南
├── requirements.txt                   # Python 依赖
├── aicrowd.json                       # AIcrowd 竞赛配置
└── Dockerfile                         # Docker 构建文件
```

---

## 🏁 快速开始

### 环境要求

| 平台 | 硬件 | 推理框架 | 推荐模型 |
|------|------|---------|---------|
| Mac (Apple Silicon) | M1/M2/M3, 16GB+ 内存 | MLX | Qwen2-VL-2B（默认）/ 7B（推荐）|
| Linux | NVIDIA L40s 48GB | vLLM | Llama-3.2-11B-Vision |

> **Mac 用户**：2B 模型轻量快速，适合调试和跑通流程。**追求分数请切换到 7B 模型**（见下方"切换更大模型"一节）。7B 模型首次需下载 ~5GB，耗时取决于网络。

### 1. 安装依赖

```bash
# 基础依赖（所有平台）
pip install -r requirements.txt

# Mac 用户 — Apple Silicon 原生加速
pip install mlx-vlm

# Linux/CUDA 用户
# pip install vllm>=0.6.2
```

### 2. 配置 Agent

编辑 `agents/user_config.py`：

```python
# Mac 用户
UserAgent = Task1SingleSourceAgent

# Linux/CUDA 用户（需要 vLLM）
# UserAgent = MLLMRAGAgent
```

### 3. 运行评估

```bash
# === Task 1（单源增强）===
python local_evaluation.py \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --suppress-web-search-api \
    --eval-model None

# === Task 2（多源增强）===
python local_evaluation.py \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --eval-model gpt-4o-mini

# === Task 3（多轮问答）===
python local_evaluation.py \
    --dataset-type multi-turn \
    --split validation \
    --num-conversations 100 \
    --eval-model gpt-4o-mini
```

### 4. 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset-type` | `single-turn` / `multi-turn` | `single-turn` |
| `--split` | `validation` / `public_test` | `validation` |
| `--num-conversations` | 评估对话数（`-1` = 全部） | `-1` |
| `--suppress-web-search-api` | **Task 1 必须加此参数** | `False` |
| `--eval-model` | 语义评估模型名，`None` = 仅精确匹配 | `gpt-4o-mini` |
| `--display-conversations` | 显示的示例数 | `10` |
| `--no_progress` | 隐藏进度条 | `False` |
| `--num-workers` | 并行评估线程数 | `8` |

> **注意**：`--eval-model None` 仅做**精确字符串匹配**，无法识别语义等价答案，分数会偏低。需要语义评估时传入 OpenAI 模型名（需设置 `OPENAI_API_KEY` 环境变量）。

---

## 📌 Task 1 详解

### 任务目标

仅使用 **图像检索模拟 API** 访问基于图像的结构化知识图谱（KG），利用 KG 返回的相似图像及其结构化数据，辅助完成回答生成。**答案可能不存在于 KG 中**，模型需要判断知识边界。

### Agent 工作流程

```
图片 + 问题
    │
    ▼
┌──────────────────────────┐
│ Step 1: 查询类型分析       │  关键词规则分类：
│                          │  visual / knowledge / reasoning / comparison
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 2: 图像 → KG 检索    │  CLIP 相似度匹配，召回 Top-K 实体
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 3: 分数过滤          │  基于 CLIP Cosine 距离阈值过滤低质量结果
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 4: KG 文本提取       │  查询感知的字段优先级 + 去重 + 截断
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 5: Prompt 构建       │  System Prompt + KG 上下文 + 问题
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 6: VLM 生成答案      │  Qwen2-VL-2B / Llama-3.2-Vision
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 7: 后处理 + 置信度评估 │  清理输出 / 不确定性 → "I don't know"
└──────────────────────────┘
```

### 核心可调参数

在 `agents/task1_agent.py` 顶部可直接修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VLM_MODEL_NAME` | `Qwen2-VL-2B-Instruct-4bit` | VLM 模型，取消注释可切换到 7B 或 11B |
| `NUM_KG_RECALL` | 10 | KG 检索召回数量 |
| `NUM_KG_FINAL` | 5 | 过滤后保留数量 |
| `SIMILARITY_THRESHOLD_STRICT` | 0.65 | 知识型问题严格阈值 |
| `SIMILARITY_THRESHOLD_LOOSE` | 0.85 | 视觉型问题宽松阈值 |
| `TEMPERATURE` | 0.0 | 生成温度（越低越确定） |
| `MAX_GENERATION_TOKENS` | 128 | 最大生成长度 |
| `BATCH_SIZE` | 4 | 批处理大小（Mac 建议 1-4） |

### 切换更大模型

```python
# agents/task1_agent.py 第 52-56 行
# 2B 模型（默认，轻量快速）
VLM_MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

# 7B 模型（推荐，M3 Pro 18GB 可运行，首次需下载 ~5GB）
# VLM_MODEL_NAME = "mlx-community/Qwen2-VL-7B-Instruct-4bit"

# 11B 模型（需要 32GB+）
# VLM_MODEL_NAME = "mlx-community/Llama-3.2-Vision-Instruct-4bit"
```

切换后首次运行会自动下载模型。如果下载慢，可设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Mac 平台性能参考

| 模型 | 下载大小 | 每查询耗时 | 内存占用 |
|------|---------|-----------|---------|
| Qwen2-VL-2B-4bit | ~2 GB | ~1.0-1.5s | ~3-4 GB |
| Qwen2-VL-7B-4bit | ~5 GB | ~3-5s | ~8-10 GB |
| Llama-3.2-Vision-11B-4bit | ~8 GB | ~5-8s | ~12-14 GB |

---

## 📏 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match** | 回答与标准答案字符串完全一致的比例 |
| **Accuracy**（准确率） | 语义上正确的回答比例（需 `--eval-model` 启用） |
| **Missing Rate**（缺失率） | "I don't know" 的比例 |
| **Hallucination Rate**（幻觉率） | 给出错误回答的比例 |
| **Truthfulness Score**（真实性得分） | 综合得分 = (2×正确 + 缺失) / 总数 − 1，范围 [-1, 1] |

> Truthfulness Score 是排行榜最终排名依据。正确回答得 +1，IDK 得 0，错误回答得 -1。

---

## 📊 当前基准数据

以下为 Mac (M3 Pro, 18GB) + Qwen2-VL-2B-4bit + 无语义评估（精确匹配）的实测结果：

| 指标 | 初始版本 (bug 未修复) | 修复 + 优化版本 |
|------|:---------------------:|:---------------:|
| Missing Rate | 59.62% | **71.15%** |
| Hallucination Rate | 40.38% | **28.85%** ↓ |
| Truthfulness Score | -0.4038 | **-0.2885** ↑ |

> ⚠️ 以上为 **2B 模型基准线**，仅代表功能跑通。要参与排行竞争需：① 切换到 7B+ 模型 ② 启用语义评估 (`--eval-model gpt-4o-mini`)。

---

## 🚀 优化方向

根据实训指导建议，以下是提升 Task 1 性能的主要途径：

| 方向 | 具体做法 | 预期收益 |
|------|---------|---------|
| **使用更强模型** | Qwen2-VL-7B、Llama-3.2-Vision-11B 等 | ⭐⭐⭐ |
| **优化 System Prompt** | 根据问题类型设计分层指令，加入 Few-shot 示例 | ⭐⭐ |
| **改进检索质量** | 调整 CLIP 阈值、增加召回量、尝试不同 embedding 模型 | ⭐⭐ |
| **引入重排序** | 在检索和生成间加入 Cross-Encoder Reranker | ⭐⭐ |
| **置信度校准** | 基于检索分数和 KG 匹配度动态决定是否回复 IDK | ⭐⭐ |
| **错误案例分析** | 按领域和问题类型分类诊断失败原因，针对性优化 | ⭐⭐ |
| **答案后处理** | 统一 IDK 格式、截断过长输出、清理模型前缀 | ⭐ |
| **参数调优** | 温度、Top-P、生成长度、检索 K 值等 | ⭐ |

---

## 🔧 踩坑记录

在 Mac (Apple Silicon, Python 3.13) 环境下运行时遇到的兼容性问题及解决方案：

### 1. MLX VLM 生成返回值变更

**现象**：`'GenerationResult' object has no attribute 'strip'`

**原因**：新版本 `mlx_vlm.generate()` 返回 `GenerationResult` 对象而非字符串，文本在 `.text` 属性中。

**修复**（`agents/task1_agent.py`）：
```python
# 错误
answer = response.strip()

# 正确
answer = response.text.strip()
```

### 2. CLIP 模型输出格式变化

**现象**：`'BaseModelOutputWithPooling' object has no attribute 'norm'`

**原因**：新版 `transformers` 中 `model.get_image_features()` 返回 `BaseModelOutputWithPooling` 对象，需要手动取 `.pooler_output`。

**修复**（`cragmm_search/image_search_mock_api/image_search.py` 第 54 行附近）：
```python
features = model.get_image_features(**inputs)
# 新版 transformers 返回 BaseModelOutputWithPooling
if hasattr(features, 'pooler_output'):
    features = features.pooler_output
features = features / features.norm(dim=-1, keepdim=True)
```

### 3. 相似度阈值与实际数据不匹配

**现象**：KG 检索返回了结果，但全部被分数阈值过滤（KG=0）。

**原因**：默认 `STRICT=0.3, LOOSE=0.6`，但实际 CLIP Cosine 距离的最佳匹配集中在 0.55-0.75 范围。

**修复**：调整为 `STRICT=0.65, LOOSE=0.85`，允许合理范围内的 KG 结果通过。

### 4. Prompt 格式不符合模型 Chat Template

**现象**：模型输出混淆（输出原始 KG 数据、回答其他问题）。

**原因**：手动拼接 `System: ... User: ...` 前缀不符合 Qwen2-VL 的 ChatML 格式。

**修复**（`agents/task1_agent.py`）：
```python
# 错误
prompt = f"System: {system}\n\nUser: {question}"

# 正确
prompt = self.processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

### 5. 大模型下载慢（国内网络）

**现象**：7B 模型从 HuggingFace 下载极慢（~5GB 需数小时）。

**解决**：
```bash
# 方法 1：使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "from mlx_vlm import load; load('mlx-community/Qwen2-VL-7B-Instruct-4bit')"

# 方法 2：用 hf CLI 单独下载
hf download mlx-community/Qwen2-VL-7B-Instruct-4bit
```

---

## 📎 相关链接

- 💪 [比赛页面](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)
- 📊 [单轮数据集 (HuggingFace)](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public)
- 📊 [多轮数据集 (HuggingFace)](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public)
- 🛠️ [入门套件 (GitLab)](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit)
- 🗣 [讨论论坛](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025/discussion)
