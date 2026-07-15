# 🤖 CRAG-MM Agent 开发指南

欢迎来到 CRAG-MM Agent 开发实践！本目录用于参赛者实现自己的视觉语言模型，以参加 CRAG-MM 基准测试。

建议将所有 Agent 运行所需的内容都放在本仓库中，以便更顺畅地完成提交和评估。不过，你也可以按照自己的偏好自由组织代码。

## 🎯 本目录结构

```
agents/
├── base_agent.py                 # 所有 Agent 必须继承的基类
├── random_agent.py               # 随机回答 Agent 示例（参考实现）
├── vanilla_llama_vision_agent.py # 基于 Llama-Vision 的视觉语言模型 Agent
├── rag_agent.py                  # 基于 RAG 的带搜索能力 Agent 示例
├── mllm_rag_agent.py             # 获奖方案：多阶段验证 RAG Agent
├── task1_agent.py                # Task 1 Agent：MLX 版，适配 Mac/Apple Silicon
└── user_config.py                # 配置文件，指定使用哪个 Agent
```

## 🧪 示例 Agent

我们提供了以下示例 Agent 帮助你入门：

1. **RandomAgent** 🎲
   - 生成随机回答的简单 Agent
   - 适合测试评估流程
   - 文件：`random_agent.py`

2. **LlamaVisionModel** 🦙
   - 基于 Meta Llama 3.2 11B Vision Instruct 的视觉语言模型
   - 支持单轮和多轮对话
   - 文件：`vanilla_llama_vision_agent.py`
   - 需要 CUDA GPU + vLLM

3. **SimpleRAGAgent** 🔍
   - 基于 RAG（检索增强生成）的 Agent
   - 使用统一搜索管道检索相关信息
   - 展示如何结合视觉搜索和文本搜索
   - 文件：`rag_agent.py`
   - 需要 CUDA GPU + vLLM

4. **Task1SingleSourceAgent** 🍎
   - 专为 Task 1（单源增强）设计的 Agent
   - 使用 MLX + Qwen2-VL-2B（4-bit 量化），适配 Apple Silicon Mac
   - 支持查询分析、分数阈值过滤、分层 Prompt、置信度评估
   - 文件：`task1_agent.py`
   - 需要 Mac (Apple Silicon) + mlx-vlm

## 🛠️ 创建自己的 Agent

按以下步骤创建自己的 Agent：

1. 在 `agents` 目录中创建新的 Python 文件
2. 导入并继承 `BaseAgent`：
   ```python
   from typing import Dict, List, Any
   from PIL import Image
   from agents.base_agent import BaseAgent
   from cragmm_search.search import UnifiedSearchPipeline

   class YourAgent(BaseAgent):
       def __init__(self, search_pipeline: UnifiedSearchPipeline):
           # 在此初始化你的模型
           # 初始化时间限制为 10 分钟
           super().__init__(search_pipeline)
           # 注意：Task 1（单源增强）中，网页搜索将被禁用，
           # 只能使用图像搜索。

       def get_batch_size(self) -> int:
           # 返回你偏好的批次大小 (1-16)
           return 8

       def batch_generate_response(
           self,
           queries: List[str],
           images: List[Image.Image],
           message_histories: List[List[Dict[str, Any]]],
       ) -> List[str]:
           # 在此实现批量回答生成的逻辑
           # 处理批次中的所有查询，返回回答列表
           # 此函数应在 at most: 10s * self.get_batch_size() 内返回
           responses = []
           for query, image, message_history in zip(queries, images, message_histories):
               # 你的处理逻辑
               responses.append("你对这个查询的回答")
           return responses
   ```

### ⚡ 性能约束

- **初始化时间**：最多 10 分钟
- **批量处理**：评估器根据你的 `get_batch_size()` 一次处理多个查询
- **批量回答时间**：每次 `agent.batch_generate_response(..)` 调用时限为 `10 s × agent.get_batch_size()`
- **内存使用**：注意 GPU 显存。提交将在单台 NVIDIA L40s GPU（48GB 显存）上运行

### 📝 必须实现的方法

你的 Agent 必须实现以下方法：

```python
def get_batch_size(self) -> int:
    # 返回 1-16 之间的值
```

```python
def batch_generate_response(
    self,
    queries: List[str],
    images: List[Image.Image],
    message_histories: List[List[Dict[str, Any]]],
) -> List[str]:
```

参数说明：
- `queries`：用户的问题列表
- `images`：对应的 PIL Image 对象列表（每个查询一个图片）
- `message_histories`：对话历史列表（每个查询一个历史记录）

message_histories 的格式：
- 单轮对话：空列表 `[]`
- 多轮对话：之前轮次的对话记录，格式如下：
  ```json
  [
    {"role": "user", "content": "第一个用户消息"},
    {"role": "assistant", "content": "第一个助手回答"},
    {"role": "user", "content": "追问"},
    {"role": "assistant", "content": "追问回答"},
    ...
  ]
  ```

## 🔧 配置

使用你的 Agent：

1. 编辑 `user_config.py`
2. 导入你的 Agent 类
3. 将其赋值给 `UserAgent`：
   ```python
   from agents.your_agent_file import YourAgent
   UserAgent = YourAgent
   ```

### Mac 用户注意事项
由于 vLLM 仅支持 CUDA/Linux，Mac 用户应使用 `task1_agent.py`（基于 MLX）。`user_config.py` 已配置 try/except 保护，确保缺少 vLLM 时不会导致导入失败。

## 📦 使用 HuggingFace 🤗 上的模型

评估期间，互联网将被禁用，环境变量 `HF_HUB_OFFLINE=1` 将被设置。如果你想使用 HuggingFace 上的模型，请在 `aicrowd.json` 中包含其模型仓库引用：
```json
{
    "challenge_id": "single-source-augmentation",
    "gpu": true,
    "hf_models": [
        {
            "repo_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "revision": "main"
        },
        {
            "repo_id": "your-org/your-model",
            "revision": "your-custom-revision",
            "ignore_patterns": "*.md"
        },
        ...
    ]
}
```

评测方将确保在评估开始前，这些模型已在评测容器的本地 Hugging Face 缓存中可用。

`model_spec` 字典的键可以包含 [`huggingface_hub.snapshot_download`](https://huggingface.co/docs/huggingface_hub/v0.30.2/en/package_reference/file_download#huggingface_hub.snapshot_download) 函数支持的任何参数。

**重要提醒：**
- 指定的模型必须是**公开的**，或者 `aicrowd` Hugging Face 账号已被明确授权访问
- 如果模型仓库是**私有的**，你必须向 [`aicrowd` 用户](https://huggingface.co/aicrowd) 授权。否则，提交将失败。

**如何授权私有仓库访问：**
为参加此比赛，建议在 Hugging Face 上创建一个专门的组织。在该组织内创建私有仓库，并将 `aicrowd` 用户添加为成员，以确保顺利访问。

### ⚠️ 重要注意事项

1. **模型访问权限**：确保 `aicrowd` HuggingFace 账号有权访问 `hf_models` 中指定的所有模型
2. **离线模式**：未在 `aicrowd.json` 中指定的任何模型在评估期间都将无法加载
3. **模型下载**：我们将在评估开始前下载所有指定的模型
4. **访问控制**：如果你的模型是私有的，请确保：
   - 已向 `aicrowd` HF 账号授权
   - 在 `hf_models` 中包含该模型
   - 使用正确的模型 ID（org/model-name）

## 🧪 本地评估

使用 `local_evaluation.py` 脚本评估你的 Agent：

```bash
# Task 1（单源增强）— 禁用 Web 搜索
python local_evaluation.py \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --suppress-web-search-api \
    --display-conversations 3 \
    --eval-model None

# Task 2（多源增强）— 同时使用图像和 Web 搜索
python local_evaluation.py \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --eval-model gpt-4o-mini

# Task 3（多轮问答）
python local_evaluation.py \
    --dataset-type multi-turn \
    --split validation \
    --num-conversations 100 \
    --eval-model gpt-4o-mini
```

参数说明：
| 参数 | 说明 |
|------|------|
| `--dataset-type` | 数据集类型："single-turn" 或 "multi-turn" |
| `--split` | 数据集分片："validation"、"public_test" |
| `--num-conversations` | 评估的对话数量（-1 表示全部） |
| `--suppress-web-search-api` | 禁用 Web 搜索 API（**Task 1 必须使用**） |
| `--display-conversations` | 显示的示例对话数量 |
| `--eval-model` | 语义评估的 OpenAI 模型（设为 'None' 禁用） |
| `--output-dir` | 评估结果保存目录 |
| `--no_progress` | 禁用进度条 |
| `--revision` | 数据集版本 |
| `--num-workers` | 并行评估线程数 |

## 🎯 评估指标

评估脚本计算以下指标：

| 指标 | 说明 |
|------|------|
| **Exact Match (精确匹配率)** | 回答与标准答案完全一致的比例 |
| **Accuracy (准确率)** | 语义上正确的回答比例 |
| **Missing Rate (缺失率)** | "I don't know" 回答的比例 |
| **Hallucination Rate (幻觉率)** | 错误回答的比例 |
| **Truthfulness Score (真实性得分)** | 综合评分 = (2×正确 + 缺失) / 总数 − 1 |
| **Multi-turn Score (多轮对话得分)** | 多轮对话中的综合表现评分 |

## 🔍 使用搜索管道

对于基于 RAG 的 Agent，可以使用提供的 `UnifiedSearchPipeline`：

```python
# 使用搜索管道
search_results = self.search_pipeline(query, k=3)

# 每个结果包含：
for result in search_results:
    snippet = result.get('page_snippet', '')
    # 在 Prompt 中使用 snippet
```

对于 Task 1（仅图像搜索），返回的每个结果还包含：
```python
{
    "index": int,         # 索引
    "score": float,       # CLIP 相似度距离（越小越相似）
    "url": str,           # 图像 URL
    "entities": [         # 关联的 KG 实体
        {
            "entity_name": str,
            "entity_attributes": {
                "description": "...",
                "caption": "...",
                # 其他自定义属性...
            }
        }
    ]
}
```

## 🚀 快速开始

1. 克隆仓库
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # Mac 用户还需安装：
   pip install mlx-vlm
   ```
3. 继承 `BaseAgent` 实现自己的 Agent
4. 更新 `user_config.py` 使用你的 Agent
5. 运行本地评估
6. 迭代改进！

祝编码愉快！🚀
