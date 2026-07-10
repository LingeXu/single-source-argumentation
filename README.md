[![banner image](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/1155/3d44411079169ec5776a.jpg)](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)

# [Meta CRAG-MM：多模态多轮对话综合 RAG 基准挑战赛](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)

本仓库是 **CRUISE 团队** 在 **KDD Cup 2025 Meta CRAG-MM 挑战赛** 中的官方实现。

我们的方法在 **Task 1：单源增强 中获得第 3 名**。

<div align="center">

### [Baiyu Chen](https://baiyuchen.com/)<sup>1,2</sup>, [Wilson Wongso](https://wilsonwongso.dev/)<sup>1,2</sup>, [Xiaoqian Hu](https://scholar.google.com/citations?user=1RUZG-IAAAAJ)<sup>1</sup>, [Yue Tan](https://yuetan031.github.io/)<sup>1</sup>, and [Flora Salim](https://fsalim.github.io/)<sup>1,2</sup>

<sup>1</sup> 新南威尔士大学计算机科学与工程学院，澳大利亚悉尼<br/>
<sup>2</sup> ARC 自动化决策与社会卓越研究中心

[![arXiv](https://img.shields.io/badge/arXiv-2507.20136-b31b1b.svg)](https://arxiv.org/abs/2507.20136)
[![Challenge](https://img.shields.io/badge/KDD%20Cup%202025-Meta%20CRAG--MM-blue.svg)](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)

</div>

# 目录

1. [我们的方法](#-我们的方法)
2. [比赛概述](#-比赛概述)
3. [数据集](#-数据集)
4. [比赛任务](#-比赛任务)
5. [评估指标](#-评估指标)
6. [快速开始](#-快速开始)
   - [如何编写自己的 Agent？](#️-如何编写自己的-agent)
   - [如何参与比赛？](#-如何参与比赛)
      - [环境配置](#环境配置)
      - [如何提交？](#-如何提交)
      - [代码运行在什么硬件上？](#-代码运行在什么硬件上)
      - [基线模型](#基线模型)
7. [常见问题](#-常见问题)
8. [重要链接](#-重要链接)

# 🧠 我们的方法

我们的框架由四个主要阶段组成：

## 🔀 轻量级查询路由
首先使用轻量级路由模块判断查询是否需要外部知识或实时检索。

## 🔎 查询感知检索与摘要
从可用来源检索相关证据，并进行查询感知的摘要，仅保留最有用的支撑上下文。

## 🧩 双路径生成
通过两条互补路径生成答案：

- **基于 RAG 的增强答案路径**
- **非 RAG / 模型先验知识答案路径**

用于比较增强证据与模型内部知识的一致性。

## ✅ 验证与最终裁决
采用**以验证为中心的答案选择流程**，包括：

- 自洽性检查 (Self-Consistency)
- 结构化验证链 (Chain-of-Verification, CoV)

这种保守的设计旨在减少幻觉，提高可信度。

## 🏆 比赛成绩

我们的方法取得了：

- **Task 1：单源增强 第 3 名**
- **无需训练**的框架，无需额外训练或微调
- 通过以验证为中心的设计，实现了强大的事实可靠性
- 在答案覆盖率和幻觉抑制之间实现了有效平衡

## 🗂️ 代码结构

主要比赛提交文件：

```bash
agents/mllm_rag_agent.py       # 获奖方案：多阶段验证 RAG Agent (需要 CUDA GPU)
agents/task1_agent.py           # Task 1 Agent：MLX 版，适配 Apple Silicon / Mac
agents/base_agent.py            # Agent 基类
agents/rag_agent.py             # 简单 RAG Agent 示例
agents/random_agent.py          # 随机回答 Agent（测试用）
agents/vanilla_llama_vision_agent.py  # 纯视觉语言模型 Agent
agents/user_config.py           # Agent 配置文件
local_evaluation.py             # 本地评估脚本
```

# 📖 比赛概述

你是否曾尝试在出国旅行时用智能眼镜查询地标的历史？是否曾用可穿戴设备实时翻译外语来点餐？是否曾因忘记停车位置，幸好在眼镜里的图像提醒中找到了位置？可穿戴设备正在彻底改变人们交流、工作和娱乐的方式。为了让可穿戴设备在日常生活中真正发挥作用，它们必须提供与用户需求相关的准确信息。

视觉大语言模型（VLLMs）近年来取得了显著进展，为智能眼镜背后的多模态理解和视觉问答（VQA）功能提供了支持。尽管取得了这些进步，VLLMs 仍面临一个重大挑战：**生成幻觉答案**。研究表明，VLLMs 在处理涉及长尾实体的查询时面临巨大困难；这些模型在处理需要整合识别、OCR、知识和生成等多种能力的复杂查询时同样困难重重。

检索增强生成（RAG）范式已扩展以适应多模态（MM）输入，并在解决 VLLM 的知识局限方面展现出了潜力。给定一张图片和一个问题，MM RAG 系统通过综合图片和问题中的信息构建搜索查询，搜索外部来源以检索相关信息，然后提供有据可依的答案来回答用户问题。

# 📊 数据集

CRAG-MM 包含三部分数据：图像集、问答对和检索内容。

数据集可通过以下链接访问：
- **单轮对话：** [https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public)
- **多轮对话：** [https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public)

## 🖼️ 图像集
CRAG-MM 包含两种类型的图像：**第一人称视角图像**和**普通图像**。第一人称视角图像是使用 RayBan Meta 智能眼镜从第一视角采集的。普通图像则来自互联网上的公开图片。

## 📝 问答对
CRAG-MM 涵盖 **14 个领域**：书籍、食品、通用物体识别、数学与科学、自然、宠物、植物与园艺、购物、观光、体育与游戏、时尚与风格、文字理解、车辆及其他。代表了可穿戴设备用户日常使用的主要场景。同时包含 **4 种问题类型**，从可直接基于图像回答的简单问题，到需要检索多个来源并综合答案的复杂问题。

## 📁 检索内容
数据集包含模拟图像搜索 API 和模拟网页搜索 API，用于模拟 RAG 方案检索的真实知识来源。

安装模拟 API：
```bash
pip install -U cragmm-search-pipeline
```

- [docs/search_api.md](docs/search_api.md) 包含模拟 API 的文档
- [agents/rag_agent.py](agents/rag_agent.py) 展示了 API 的示例用法

# 👨‍💻👩‍💻 比赛任务

比赛设有三个任务：

## Task #1：单源增强 (Single-Source Augmentation) ⭐
Task #1 提供图像模拟 API，访问底层的基于图像的结构化知识图谱（mock KG）。该 KG 以图像为索引，存储与图像相关的结构化数据；问题的答案可能存在，也可能不存在于 KG 中。模拟 API 以图像为输入，返回 KG 中的相似图像及其结构化数据，以辅助答案生成。此任务旨在**测试 MM-RAG 系统的基本答案生成能力**。

> 📌 **难度系数：1.0**

## Task #2：多源增强 (Multi-Source Augmentation)
Task #2 额外提供**网页搜索模拟 API** 作为第二个检索来源。网页内容可能包含回答问题的有用信息，但同时也包含大量噪声。此任务旨在测试 **MM-RAG 系统如何综合不同来源的信息**。

> 📌 **难度系数：1.2**

## Task #3：多轮问答 (Multi-turn QA)
Task #3 测试系统进行多轮对话的能力。每次对话包含 **2–6 轮**。除首轮外，后续问题可能需要也可能不需要图像来回答。Task #3 测试 **上下文理解能力**，以实现流畅的多轮对话。

> 📌 **难度系数：1.5**

# 📏 评估指标

对于 Task #1 和 #2，我们采用与 CRAG 竞赛（KDD Cup 2024）完全相同的指标和方法来评估 MM RAG 系统的性能。

## 单轮问答（Task #1 和 #2）
对评测集中的每个问题，答案评分为：

| 等级 | 分数 | 说明 |
|------|------|------|
| 完美 (Perfect) | **1** | 完全正确 |
| 可接受 (Acceptable) | **0.5** | 有用但存在轻微无害的错误 |
| 缺失 (Missing) | **0** | 如 "I don't know" |
| 错误 (Incorrect) | **-1** | 错误或不相关的答案 |

**真实性得分 (Truthfulness Score)** = 所有样本的平均分，同时也是排行榜上的**最终排名依据**。对每个领域分别计算平均分，再按所有领域的加权平均得出最终分数。

## 多轮问答
当连续两轮的答案都错误时，对话终止，该对话中剩余问题的答案均视为缺失（模拟真实用户失去信任后放弃对话的行为）。最终取所有多轮对话的平均分。

# 🏁 快速开始

1. **注册报名** 参加 [AIcrowd 官网](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)的比赛。
2. **Fork** 本入门套件仓库。
3. **克隆** fork 后的仓库，开始开发你的 Agent。
4. **开发** 你的 Agent，参考 [如何编写自己的 Agent](#️-如何编写自己的-agent) 部分的模板。
5. [**提交**](#-如何提交) 到 [AIcrowd Gitlab](https://gitlab.aicrowd.com) 进行评估。

# ✍️ 如何编写自己的 Agent？

请参考 [agents/README.md](agents/README.md) 中的说明和示例，了解如何为此比赛编写自己的 Agent。

## 🎯 Task 1 快速上手（Mac / Apple Silicon）

本项目提供了适配 Mac 的 Task 1 Agent (`agents/task1_agent.py`)，使用 MLX + Qwen2-VL-2B 实现 Apple Silicon 原生加速：

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install mlx-vlm

# 2. 确认使用 Task 1 Agent
# 编辑 agents/user_config.py，确保：
# UserAgent = Task1SingleSourceAgent

# 3. 运行评估 (不启用语义评估，Mac 上无需 OpenAI API)
python local_evaluation.py \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --suppress-web-search-api \
    --eval-model None
```

## 🚀 优化方向 (来自 PPTX 实训指导)

| 优化方向 | 说明 |
|----------|------|
| **优化 System Prompt** | 根据问题类型（视觉/知识/推理/比较）设计分层指令 |
| **模型生成后处理** | 清理输出格式、统一 IDK 表达、截断过长答案 |
| **参数调优** | 调整温度、检索数量、相似度阈值等参数 |
| **设计更好的 Agent** | 查询路由、分数过滤、置信度评估、双路径生成 |
| **不确定性处理** | 基于检索质量和置信度分数判断是否回复 "I don't know" |
| **检索优化** | 根据问题困难程度调整检索策略，提高匹配精度 |

# 🚴 如何参与比赛？

## 环境配置

1. **添加 SSH Key** 到 AIcrowd GitLab

在 [GitLab 个人设置](https://gitlab.aicrowd.com/-/user_settings/ssh_keys)中添加 SSH Key。如果没有，需要先生成一个。

2. **Fork 仓库**

3. **克隆仓库**
    ```bash
    git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USERNAME>/meta-crag-submission.git
    cd meta-crag-submission
    ```

4. **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
**注意**：vLLM 的安装可能依赖特定的 CUDA 或 PyTorch 版本，`pip install -r requirements.txt` 可能会失败。如果失败，请在 [vLLM 官网](https://docs.vllm.ai/en/latest/) 寻找合适版本。运行 LLaMA-3.2-Vision 至少需要 `vllm>=0.6.2`。

    - **Mac 用户**：只需安装基础依赖 + `mlx-vlm`，无需 vLLM

5. 按照 [agents/README.md](agents/README.md) 编写自己的 Agent。

6. 使用 `python local_evaluation.py` 在本地测试 Agent。

7. 在[比赛页面](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)点击 **Participate** 按钮接受比赛规则。

8. 按照[提交指南](#-如何提交)提交。

## 📮 如何提交？

请参考 [docs/submission.md](docs/submission.md) 进行首次提交。

## 💻 代码运行在什么硬件上？

所有提交将在 AWS 上的单台 **`g6e.2xlarge`** 实例上运行，配备 **NVIDIA L40s GPU（48GB 显存）**。请注意：

- `LLaMA 3.2 11B-Vision` 和 `Pixtral 12B` 可以全精度直接运行
- `Llama 3.2 90B-Vision` 全精度无法直接运行，需要量化等技术手段才能运行

此外，还有以下限制：
- **网络连接将被禁用**
- 每轮对话有 **10 秒超时**限制，N 轮对话的批次有 `N × 10 秒` 超时
- 为鼓励简洁回答，每个答案将在自动评估中被截断为 **75 个 BPE token**

## 基线模型

我们提供了三个基线模型：

| Agent | 说明 |
|-------|------|
| **RandomAgent** | 生成随机回答的简单 Agent |
| **LlamaVisionModel** | 基于 Meta LLaMA 3.2 11B Vision Instruct 的视觉语言 Agent |
| **SimpleRAGAgent** | 使用统一搜索管道检索相关信息的 RAG Agent |

详见 [docs/baselines.md](docs/baselines.md)。

# ❓ 常见问题

**Q: 哪里可以了解更多关于数据集结构的信息？**
数据集结构描述在 [docs/dataset.md](docs/dataset.md)。

**Q: 为什么提交失败？**
常见原因：
1. 未在 `aicrowd.json` 中指定所需的 HuggingFace 模型 → 评测时无网络，无法下载
2. 私有模型未授权给 `aicrowd` HuggingFace 账号
3. 代码超时（单轮 10s、批次 N×10s）
4. Agent 初始化超过 10 分钟

**祝你好运！** 🎉 🎉

如果你觉得这个仓库有用，请引用我们的论文：
```bibtex
@article{chen2025multi,
  title={Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG},
  author={Chen, Baiyu and Wongso, Wilson and Hu, Xiaoqian and Tan, Yue and Salim, Flora},
  journal={arXiv preprint arXiv:2507.20136},
  year={2025}
}
```

# 📎 重要链接

- 💪 [比赛页面](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)
- 🗣 [讨论论坛](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025/discussion)
- 🏆 [获奖公告](https://discourse.aicrowd.com/t/meta-crag-challenge-2025-winners-announcement/17308)
- 🛠️ [Workshop](https://kddcup25.github.io/)
- 📧 联系邮箱：`breeze.chen(at)unsw(dot)edu(dot)au`
