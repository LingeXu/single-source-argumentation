# 提交指南 🚀

准备好将你的 **CRAG-MM** 方案展示在排行榜上了吗？按以下步骤将你的模型**提交**到 **Meta CRAG-MM** 挑战赛！🎉

---

## 1. 仓库与配置

### 1.1 克隆入门套件 🏁

1. **Fork** 官方的 **[Meta CRAG-MM Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit/-/tree/main)** 仓库
2. **克隆**你 fork 的仓库到本地：
   ```bash
   git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USERNAME>/<YOUR-FORK>.git
   cd <YOUR-FORK>
   ```

### 1.2 添加你的 Agent 代码 🧩

1. 进入克隆仓库中的 `agents/` 目录
2. 创建或修改文件（如 `my_agent.py`），实现 [BaseAgent](../agents/base_agent.py) 接口
3. 在 `agents/user_config.py` 中，**导入**你的新 Agent 类并**赋值**给 `UserAgent`

> **注意**：评估期间，你的代码将在**离线环境**（无互联网！）中运行。这意味着你必须**预下载**或以其他方式引用 **Hugging Face** 模型，使其在离线状态下可用。参见[下方说明](#link-to-hf)了解如何指定 HF 模型。

---

## 2. 指定模型与依赖

### 2.1 aicrowd.json 🗒️

在仓库根目录创建或更新 `aicrowd.json`，指定提交的关键信息：

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
        }
    ]
}
```

- **`challenge_id`**：选择以下之一：
  - `"single-source-augmentation"` — Task 1: 单源增强
  - `"multi-source-augmentation"` — Task 2: 多源增强
  - `"multi-turn-qa"` — Task 3: 多轮问答

- **`gpu`**：需要 GPU 加速设为 `true`，否则设为 `false`

- **`hf_models`**：列出你的 Agent 使用的所有 Hugging Face 模型。这些模型**必须**是公开的，或者已明确授权给 `aicrowd` Hugging Face 账号。评估前，这些模型会被预下载并缓存在一个没有互联网的容器中（`HF_HUB_OFFLINE=1`）

  > `hf_models` 条目支持与 [`huggingface_hub.snapshot_download`](https://huggingface.co/docs/huggingface_hub/v0.30.2/en/package_reference/file_download#huggingface_hub.snapshot_download) 兼容的参数

关于如何安全地将私有 Hugging Face 模型用于提交的详细说明，请参考[使用受限 Hugging Face 模型提交 🔒](huggingface-gated-models.md)。

### 2.3 requirements.txt 🗒️

所有 Python 依赖必须在 `requirements.txt` 中声明。例如：

```
torch>=2.0.0
transformers>=4.36.0
pillow>=10.0.0
numpy>=1.24.0
# 包含你的 Agent 需要的任何其他库
```

### 2.4 Dockerfile 🐳（可选）

如果你想进一步自定义运行环境，可以编辑或创建仓库根目录中的 `Dockerfile`。例如：

```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# (可选) 复制代码
COPY . /app

# (可选) 指定环境变量
ENV HF_HUB_OFFLINE=1
```

---

## 3. 提交流程

### 3.1 提交并推送代码 🌐

当你完成以下步骤后：
1. 在 `agents/` 中实现或更新了你的 Agent
2. 在 `aicrowd.json` 中指定了你的模型
3. 在 `requirements.txt` 中列出了依赖
4. （可选）更新了 `Dockerfile`

提交并推送你的更改：
```bash
git add .
git commit -m "添加我的自定义 Agent"
git push origin main
```

### 3.2 标记提交版本 ✨

**创建以 `submission-<version>` 开头的 Git 标签**来触发提交：

```bash
git tag submission-v1.0
git push origin submission-v1.0
```
这个**带标签的提交**将用于构建和评估你的模型，并在排行榜上生成分数。你可以创建任意多个 `submission-*` 标签（如 `submission-v1.1`、`submission-v2.0` 等）。

---

## 4. 硬件与评估环境

1. **硬件**：你的代码将在 **NVIDIA L40s GPU** 上运行，配备 4 vCPU、**32GB RAM**、**48GB 显存**，无互联网（`HF_HUB_OFFLINE=1`）
2. **初始化时间**：有 **10 分钟**下载模型和设置环境
3. **回答时间**：每次调用 Agent 的 `batch_generate_response()` 必须在 **10 秒 × agent.get_batch_size()** 内完成
4. **无网络**：任何试图访问外部 URL 的代码都将失败。确保你的模型和所有依赖通过 `hf_models` 或 Docker 镜像在离线状态下可用

---

## 5. 提示与示例

### 5.1 aicrowd.json 示例

```json
{
    "challenge_id": "multi-source-augmentation",
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
        }
    ]
}
```
- 提交到 **multi-source-augmentation** 赛道
- 申请 GPU 资源
- 两个 HF 模型：Llama 3.2 11B Vision Instruct + 你的自定义视觉模型

### 5.2 requirements.txt 示例

```
torch>=2.0.0
transformers>=4.36.0
pillow>=10.0.0
numpy>=1.24.0
some-retrieval-lib>=0.1.3
```

### 5.3 Dockerfile 示例

```dockerfile
FROM python:3.10-slim-bookworm

RUN pip install --progress-bar off --no-cache-dir -U pip==21.0.1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --progress-bar off --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/aicrowd
COPY . .
```

---

## 6. 选择赛道

在 `aicrowd.json` 中，将 `"challenge_id"` 设为以下之一：

1. **`"single-source-augmentation"`** — Task 1: 单源增强
2. **`"multi-source-augmentation"`** — Task 2: 多源增强
3. **`"multi-turn-qa"`** — Task 3: 多轮问答

选择的 ID 决定了你的提交将参加**哪个任务**的比赛。🏆

---

## 7. 下一步

- 在 `agents/` 中编写或完善你的 Agent 代码
- 用正确的 `challenge_id` 和 `hf_models` 更新 `aicrowd.json`
- 用 `submission-<version>` 标签你的提交并推送：
  ```bash
  git commit -am '你的提交信息'
  git tag -am 'submission-<名称>' submission-<名称>
  git push origin submission-<名称>
  ```
- 🎉 在比赛页面上查看你的排行榜结果！

## 8. 故障排除与最佳实践

1. **模型访问**：确认 **`aicrowd`** Hugging Face 账号有权拉取你的私有模型
2. **本地测试**：在打标签提交前使用 `python local_evaluation.py` 验证基本功能
3. **性能优化**：对于大模型，考虑量化或其他加速方法以满足运行时间限制
4. **多次提交**：为不同版本打标签（如 `submission-v1.0`、`submission-v1.1`）进行多次尝试

> **提示**：关注你的 Docker 构建日志，确保不会超过时间或内存限制。

---

**期待看到你的创意方案！** 🤖🌟 祝编码愉快！
