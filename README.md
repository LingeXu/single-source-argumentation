# Meta CRAG-MM：多模态 RAG 基准挑战赛

> **上海大学 计算机工程与科学学院 · 人工智能专业 · 智能技术实训课程项目**  
> 赛题：[KDD Cup 2025 Meta CRAG-MM Challenge](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)  
> 第 9 组 | 组长：徐麟阁（KG 检索优化）| 组员：薛沣时（VLM+Prompt）| 组员：张桂银（实验+输出）

本项目基于竞赛官方入门套件，完成了 **Task 1（单源增强）和 Task 2（多源增强）** 两套系统的开发与本地评估。

---

## 目录

1. [比赛背景](#-比赛背景)
2. [比赛任务](#-比赛任务)
3. [项目结构](#-项目结构)
4. [快速开始](#-快速开始)
5. [Task 1 详解](#-task-1-详解)
6. [Task 2 详解](#-task-2-详解)
7. [评估指标](#-评估指标)
8. [当前基准数据](#-当前基准数据)
9. [后续规划](#-后续规划)
10. [踩坑记录](#-踩坑记录)

---

## 📖 比赛背景

KDD Cup 是由 ACM 组织的年度国际数据挖掘和知识发现大赛（CCF-A 类），素有"大数据领域世界杯"之称。2025 年赛题 **Meta CRAG-MM Challenge** 聚焦于**多模态检索增强生成（MM-RAG）**，旨在解决视觉大语言模型（VLLM）在实际可穿戴设备应用中的**幻觉问题**。

CRAG-MM 数据集包含：
- **5K+ 图像**（含 3K 第一人称视角图像，由 RayBan Meta 智能眼镜采集）
- **14 个领域**：书籍、食品、车辆、购物、观光、宠物、时尚、自然等
- **4 种问题类型**：简单识别、多跳推理、比较汇总、逻辑推理

---

## 👨‍💻 比赛任务

| 任务 | 名称 | 目标 | 检索源 | 难度 |
|------|------|------|--------|------|
| Task 1 | 单源增强 | 测试 MM-RAG 基础回答生成能力 | 仅图像 KG API | ⭐ 1.0 |
| Task 2 | 多源增强 | 测试多信息源融合与噪声鲁棒性 | 图像 KG + 网页搜索 API | ⭐⭐ 1.2 |
| Task 3 | 多轮问答 | 测试上下文理解与对话连贯性 | 图像 KG + 网页搜索 API | ⭐⭐⭐ 1.5 |

---

## 🗂️ 项目结构

```
CRAG_MM_Proj/
├── shared/                             # 公共代码
│   ├── base_agent.py                   # Agent 基类（所有 Agent 必须继承）
│   ├── crag_batch_iterator.py          # 批量数据迭代器
│   ├── crag_image_loader.py            # 图片加载与缓存
│   ├── crag_web_result_fetcher.py      # 网页结果获取与缓存
│   └── utils.py                        # 工具函数（缓存配置、结果展示等）
│
├── task1/                              # Task 1：单源增强（⭐ 1.0）
│   ├── task1_agent.py                  # ★ KG 检索 + VLM 生成 + 置信度评估
│   └── user_config.py                  # 配置：UserAgent = Task1SingleSourceAgent
│
├── task2/                              # Task 2：多源增强（⭐⭐ 1.2）
│   ├── task2_agent.py                  # ★ KG + Web 双源融合 + 噪声过滤
│   ├── rag_agent.py                    # 官方简单 RAG 示例（需 CUDA GPU）
│   ├── mllm_rag_agent.py               # 获奖方案：多阶段验证 RAG（需 CUDA GPU）
│   ├── vanilla_llama_vision_agent.py   # 纯视觉 VLM Agent（需 CUDA GPU）
│   ├── random_agent.py                 # 随机回答 Agent（测试评估流程用）
│   └── user_config.py                  # 配置：UserAgent = Task2MultiSourceAgent
│
├── local_evaluation.py                 # 本地评估脚本（主入口，支持 --task 参数）
├── docs/                               # 官方文档
│   ├── search_api.md                   # 模拟搜索 API 文档
│   ├── dataset.md                      # 数据集结构说明
│   ├── baselines.md                    # 基线模型说明
│   └── submission.md                   # 提交指南
├── requirements.txt                    # Python 依赖
├── aicrowd.json                        # AIcrowd 竞赛配置
└── Dockerfile                          # Docker 构建文件
```

> **设计理念**：Task 1 和 Task 2 各自拥有独立的代码目录和配置文件，公共逻辑统一收归 `shared/`，便于快速开展对照实验。

---

## 🏁 快速开始

### 环境要求

| 平台 | 硬件 | 推理框架 | 推荐模型 |
|------|------|---------|---------|
| Mac (Apple Silicon) | M1/M2/M3, 16GB+ 内存 | MLX | Qwen2-VL-2B（默认）/ 7B（推荐）|
| Linux | NVIDIA L40s 48GB | vLLM | Llama-3.2-11B-Vision |

### 1. 安装依赖

```bash
# 基础依赖（所有平台）
pip install -r requirements.txt

# Mac 用户 — Apple Silicon 原生加速
pip install mlx-vlm

# Linux/CUDA 用户
# pip install vllm>=0.6.2
```

### 2. 运行评估

```bash
# === Task 1（单源增强）===
python local_evaluation.py \
    --task task1 \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --suppress-web-search-api \
    --eval-model None

# === Task 2（多源增强）===
python local_evaluation.py \
    --task task2 \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --eval-model None

# === 启用 GPT 语义评估（需配置 OPENAI_API_KEY）===
python local_evaluation.py \
    --task task2 \
    --eval-model gpt-4o-mini
```

### 3. 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 选择任务：`task1` 或 `task2` | `task2` |
| `--dataset-type` | `single-turn` / `multi-turn` | `single-turn` |
| `--split` | `validation` / `public_test` | `validation` |
| `--num-conversations` | 评估对话数（`-1` = 全部） | `-1` |
| `--suppress-web-search-api` | **Task 1 必须加此参数** | `False` |
| `--eval-model` | 语义评估模型名，`None` = 仅本地语义相似度 | `gpt-4o-mini` |
| `--display-conversations` | 显示的示例数 | `10` |
| `--no_progress` | 隐藏进度条 | `False` |
| `--num-workers` | 并行评估线程数 | `8` |

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
│ + 规则快速提取（旁路）     │  确定性匹配：price→msrp, brand→manufacturer...
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 5: Prompt 构建       │  System Prompt + Few-shot 示例 + KG 上下文 + 问题
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 6: VLM 生成答案      │  Qwen2-VL-2B / 7B，MLX 原生加速
└──────┬───────────────────┘
       ▼
┌──────────────────────────┐
│ Step 7: 后处理 + 置信度    │  答案清洗 / 不确定性 → "I don't know"
│                          │  规则结果兜底：VLM 无效时回退到规则提取
└──────────────────────────┘
```

### 核心技术特点

| 特点 | 说明 |
|------|------|
| **双阈值自适应过滤** | 知识型问题用严格阈值 (0.65)，视觉型问题放宽到 (0.85) |
| **规则 + VLM 双通道** | 规则通道毫秒级确定性提取，VLM 通道深度理解，VLM 失效时规则兜底 |
| **Few-shot Prompt** | 5 组示例教模型"KG 有值→直接取、无值→看图、都无→说 IDK" |
| **双防线防幻觉** | 后处理层（统一不确定性表述）+ 置信度层（检索质量不足→强制 IDK） |

### 核心可调参数

在 `task1/task1_agent.py` 顶部可直接修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VLM_MODEL_NAME` | `Qwen2-VL-2B-Instruct-4bit` | VLM 模型，可切换到 7B 或 11B |
| `NUM_KG_RECALL` | 10 | KG 检索召回数量 |
| `NUM_KG_FINAL` | 5 | 过滤后保留数量 |
| `SIMILARITY_THRESHOLD_STRICT` | 0.65 | 知识型问题严格阈值 |
| `SIMILARITY_THRESHOLD_LOOSE` | 0.85 | 视觉型问题宽松阈值 |
| `TEMPERATURE` | 0.0 | 生成温度（越低越确定） |
| `MAX_GENERATION_TOKENS` | 128 | 最大生成长度 |
| `BATCH_SIZE` | 4 | 批处理大小（Mac 建议 1-4） |

---

## 📌 Task 2 详解

### 任务目标

在 Task 1 基础上新增**网页搜索 API** 作为第二信息源。系统需同时访问图像 KG 和 Web 搜索两个 API，具备**信息筛选、跨源融合与噪声鲁棒性**。

### 与 Task 1 的核心差异

```
Task 1: 图片 → KG 检索 ────────────→ 答案
Task 2: 图片 → KG 检索 + Web 搜索 → 融合 → 答案
                        ↑ 新增
```

### 新增能力

| 能力 | 实现 |
|------|------|
| **查询感知 Web 搜索** | 用 KG 实体名 + 用户问题拼接搜索词，提升 Web 召回精度 |
| **三级噪声过滤** | 格式清洗（去 HTML）→ 内容过滤（去 Cookie/广告）→ 相关度重排（取 Top-3） |
| **分层优先级策略** | KG > Web > Image，冲突时以 KG 为准 |
| **双源联合置信度** | KG 和 Web 都无结果时才强制 IDK，有任一源匹配则放行 |

---

## 📏 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match** | 回答与标准答案字符串完全一致的比例 |
| **Accuracy**（准确率） | 语义上正确的回答比例 |
| **Missing Rate**（缺失率） | "I don't know" 的比例 |
| **Hallucination Rate**（幻觉率） | 给出错误回答的比例 |
| **Truthfulness Score**（真实性得分） | 综合得分 = (2×正确 + 缺失) / 总数 − 1，范围 [-1, 1] |

> Truthfulness Score 是排行榜最终排名依据。正确回答得 +1，IDK 得 0，错误回答得 -1。

---

## 📊 当前基准数据

以下为 Mac (M3 Pro, 18GB) + Qwen2-VL-2B-4bit + 本地语义相似度评估（`all-MiniLM-L6-v2`）的实测结果：

### Task 1 vs Task 2 对比

| 指标 | Task 1（50 条） | Task 2（100 条） |
|------|:--------------:|:---------------:|
| Accuracy | 0.00% | **31.73%** |
| Missing Rate | 71.15% | **32.69%** |
| Hallucination Rate | **28.85%** | 35.58% |
| Truthfulness Score | -0.29 | **-0.04** |

> ⚠️ **评估说明**：
> - Task 1 用 2B 模型，缺失率偏高（过于保守）；Truthfulness Score 为负说明幻觉率 > 0。
> - Task 2 引入 Web 搜索后，准确率从 0 提升至 31.73%，缺失率从 71% 降至 33%，验证了多源检索的有效性。
> - 以上为 **2B 模型基准线**。要参与排行竞争需：① 切换到 7B+ 模型 ② 启用 GPT-4o-mini 语义评估。

### Task 1 错误分析

从指标逆推，Task 1 的错误分为三类：

| 错误类型 | 占比 | 根因 | 优化方向 |
|---------|:---:|------|---------|
| 过度保守（假 IDK） | ~50% | 固定阈值 (0.65/0.85) 将 VLM 正确回答强行覆盖 | 参数网格搜索，动态阈值 |
| KG 检索漏配 | ~20% | CLIP 未能匹配到正确 KG 实体 | 增大召回量 + Re-ranker 精排 |
| VLM 事实错误 | ~30% | 2B 小模型提取/推理能力不足 | 升级 7B 模型 + Few-shot |

---

## 🚀 后续规划

| 阶段 | 优化项 | 说明 | 预期收益 |
|------|--------|------|---------|
| **短期** | 模型升级 2B→7B | Qwen2-VL-7B-Instruct-4bit | Accuracy +8~12% |
| （第 4-5 周） | 参数精调脚本 | 网格搜索最优阈值组合，减少假 IDK | Hallu -5% |
| | Prompt 重构 | 针对错误 case 补充 Few-shot 示例 | Hallu -3~5% |
| | 嵌入语义分类 | 用 embedding 替换关键词规则做查询分类 | 分类精度 +10% |
| **中期** | Re-ranker 精排 | BGE-reranker 对 KG 召回结果重排序 | 检索精度 +10~15% |
| （第 5-6 周） | 闭环错误分析 | 自动统计错误类型分布，反哺优化策略 | 持续迭代 |
| | 评估体系升级 | 接入 GPT-4o-mini 语义评估，建立可信基线 | 量化基准 |

---

## 🔧 踩坑记录

在 Mac (Apple Silicon, Python 3.13) 环境下运行时遇到的兼容性问题及解决方案：

### 1. MLX VLM 生成返回值变更

**现象**：`'GenerationResult' object has no attribute 'strip'`

**原因**：新版本 `mlx_vlm.generate()` 返回 `GenerationResult` 对象而非字符串，文本在 `.text` 属性中。

**修复**（`task1/task1_agent.py`）：
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
if hasattr(features, 'pooler_output'):
    features = features.pooler_output
features = features / features.norm(dim=-1, keepdim=True)
```

### 3. 相似度阈值与实际数据不匹配

**现象**：KG 检索返回了结果，但全部被分数阈值过滤（KG=0）。

**原因**：默认 `STRICT=0.3, LOOSE=0.6`，但实际 CLIP Cosine 距离的最佳匹配集中在 0.55-0.75 范围。

**修复**：调整为 `STRICT=0.65, LOOSE=0.85`。

### 4. Prompt 格式不符合模型 Chat Template

**现象**：模型输出混淆（输出原始 KG 数据、回答其他问题）。

**原因**：手动拼接 `System: ... User: ...` 前缀不符合 Qwen2-VL 的 ChatML 格式。

**修复**（`task1/task1_agent.py`）：
```python
# 正确
prompt = self.processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

### 5. ChromaDB 分页 Bug（Web 搜索）

**现象**：Web 搜索加载 90 万条数据时报 `chromadb.errors.InternalError: too many SQL variables`。

**原因**：`cragmm_search/web_search_mock_api/api/web_index.py` 中 `vector_db.get()` 一次性加载全部数据，超出 SQLite 变量上限。

**修复**：改为分页加载（`limit=10000`，分批拼接）。详见 `cragmm_search/web_search_mock_api/api/web_index.py`。

---

## 📎 相关链接

- 💪 [比赛页面](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)
- 📊 [单轮数据集 (HuggingFace)](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public)
- 📊 [多轮数据集 (HuggingFace)](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public)
- 🛠️ [入门套件 (GitLab)](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit)
- 🗣 [讨论论坛](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025/discussion)
- 📂 [项目仓库](https://github.com/LingeXu/single-source-argumentation)
