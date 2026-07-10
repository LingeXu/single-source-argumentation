# 基线模型实现 🚀

本文档介绍仓库 `agents` 目录中提供的**基线 Agent**。这些 Agent 展示了解决 **Meta CRAG-MM** 基准测试的不同方法。希望它们能成为你创建自己方案的**有趣**且**有启发性的**起点！🤖

---

## 🎯 为什么需要基线？

基线演示了如何实现符合[提交指南](../docs/submission.md)和 [Agent 接口](../agents/README.md)的 Agent。它们也为使用 `local_evaluation.py` 进行**本地评估**提供了参考基准。你可以尝试运行它们、修改它们的逻辑，或构建全新的解决方案！

---

## 1. RandomAgent 🎲

**文件**：[`agents/random_agent.py`](../agents/random_agent.py)  
**类名**：`RandomAgent`

### 主要特点

- **随机字符串生成器**：生成可变长度（2-16 字符）的随机字母串
- **参考实现**：以最小开销展示 Agent 接口
- **批量处理**：演示如何高效处理批量查询
- **无真实智能**：仅用于测试评估流程

### 使用示例

```python
from agents.random_agent import RandomAgent
from cragmm_search.search import UnifiedSearchPipeline

# 初始化 Agent
search_pipeline = UnifiedSearchPipeline(...)  # Task 1 可传 None
agent = RandomAgent(search_pipeline)

# 获取评估用的批次大小
batch_size = agent.get_batch_size()  # 返回 16

# 处理一批查询
queries = ["这是什么？", "描述这张图片。"]
images = [image1, image2]  # PIL Image 对象
message_histories = [[], []]  # 单轮对话为空列表
responses = agent.batch_generate_response(queries, images, message_histories)
print(responses)  # 输出随机字符串，如 ["abDUq hf", "xYz pQr"]
```

---

## 2. LlamaVisionModel 🦙

**文件**：[`agents/vanilla_llama_vision_agent.py`](../agents/vanilla_llama_vision_agent.py)  
**类名**：`LlamaVisionModel`

### 主要特点

- **视觉语言模型**：使用 `meta-llama/Llama-3.2-11B-Vision-Instruct` + vLLM 进行高效推理
- **图像 + 文本处理**：同时处理 PIL Image 对象和文本查询
- **对话历史**：支持带对话历史的多轮对话
- **批量处理**：高效并行处理多个查询和图像
- **优化配置**：包含在单张 NVIDIA L40s GPU 上运行的优化设置

### 使用示例

```python
from PIL import Image
from agents.vanilla_llama_vision_agent import LlamaVisionModel
from cragmm_search.search import UnifiedSearchPipeline

# 初始化 Agent
search_pipeline = UnifiedSearchPipeline(...)  # Task 1 可传 None
agent = LlamaVisionModel(search_pipeline)

# 处理一批查询
queries = ["这张照片是在哪里拍的？", "你能在图片中看到什么？"]
images = [Image.open("path/to/image1.jpg"), Image.open("path/to/image2.jpg")]
message_histories = [[], []]  # 单轮对话为空列表

responses = agent.batch_generate_response(queries, images, message_histories)
print(responses)
```

> **注意**：此 Agent 演示了如何使用 vLLM 库结合 Llama Vision 模型实现更快的推理速度。

---

## 3. SimpleRAGAgent 🔎

**文件**：[`agents/rag_agent.py`](../agents/rag_agent.py)  
**类名**：`SimpleRAGAgent`

### 主要特点

- **检索增强生成**：使用 `UnifiedSearchPipeline` 根据图像内容和查询获取外部文本片段
- **批量处理**：高效处理单批次中的多个查询
- **两步方法**：
  1. 首先为图像生成摘要，构建有效的搜索词
  2. 然后检索相关信息并将其融入回答
- **增强 Prompt**：用检索到的上下文构建更优质的 Prompt

### 使用示例

```python
from PIL import Image
from agents.rag_agent import SimpleRAGAgent
from cragmm_search.search import UnifiedSearchPipeline

# 初始化 Agent（RAG Agent 需要搜索管道）
search_pipeline = UnifiedSearchPipeline(...)
agent = SimpleRAGAgent(search_pipeline)

# 处理一批查询
queries = ["这是什么类型的车？", "这张照片中显示的是什么地标？"]
images = [Image.open("path/to/car.jpg"), Image.open("path/to/landmark.jpg")]
message_histories = [[], []]  # 单轮对话为空列表

responses = agent.batch_generate_response(queries, images, message_histories)
print(responses)
```

> **提示**：RAG Agent 演示了如何将视觉语言模型与外部知识检索相结合。对于需要图像中不可见的客观事实信息的查询特别有用。

---

## 🧰 其他资源

1. **[提交指南](../docs/submission.md)** – 解释如何组织仓库并推送 Agent 进行评估
2. **[Agent 开发指南](../agents/README.md)** – 详细说明如何创建或修改 Agent，包括 `BaseAgent` 接口
3. **[本地评估脚本](../local_evaluation.py)** – 在 CRAG-MM 数据集上测试 Agent，支持快速迭代

---

## 🔧 如何在基线之间切换

1. 打开 [`agents/user_config.py`](../agents/user_config.py)
2. 更新导入和赋值：
   ```python
   from agents.rag_agent import SimpleRAGAgent
   UserAgent = SimpleRAGAgent
   ```
3. 运行本地评估：
   ```bash
   python local_evaluation.py --dataset-type single-turn --split validation --num-conversations 10 --display-conversations 3
   ```

---

## 🚀 准备构建你自己的 Agent？

1. **选择一个基线**作为起点
2. **克隆**它，或从 [`BaseAgent`](../agents/base_agent.py) **从头开始**
3. **实现**你自己的方法 — 无论是增强 Prompt、自定义检索方法，还是专用视觉模型
4. **本地测试**，使用评估脚本确保一切正常
5. **提交**你的作品，按 [submission.md](../docs/submission.md) 的说明操作！

---

**享受在这些基线上构建的乐趣，愿你的回答永远有据可依！** 🏆
