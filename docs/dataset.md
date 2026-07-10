# CRAG-MM 数据集文档

**CRAG-MM（Comprehensive RAG Benchmark for Multi-modal, Multi-turn）** 是一个面向事实的视觉问答语料库，用于评估和训练**单轮**和**多轮**场景下的检索增强生成（RAG）系统。

最新公开发布版本：**v0.1.2**

Hugging Face 上可用：

| 模态 | URL |
|------|-----|
| 单轮对话 | [https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public) |
| 多轮对话 | [https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public) |

---

## 1. 数据集亮点

1. **图像** – 混合了 Ray-Ban Meta 智能眼镜采集的第一人称照片和公开许可的网络图片
2. **13 个领域** – 如购物、食品、数学与科学等
3. **多样化的问题类型** – 识别、多跳推理、聚合、比较等
4. **单轮 vs 多轮** – 单轮问答或围绕同一张图片的延伸对话
5. **画质变化** – 正常、低光、模糊等
6. **第二阶段评估** – 所有第一人称图像在评分前都会被降采样到 **960 × 1280**；非第一人称图像保留原始分辨率

---

## 2. 数据分片

公开发布 **v0.1.2** 仅包含 **validation** 分片。之前的 *sample* 分片已废弃。

---

## 3. 数据结构（v0.1.2）

### 关于包装器的说明

在 v0.1.2 中，内部列 `turns` 和 `answers` 以**列字典**而非列表字典的形式存储：

```text
# 单轮示例
type(sample["turns"])   # -> <class 'dict'>
# 多轮示例
sample["turns"]["query"]        # -> list[str]  (长度 = 轮次数)
sample["answers"]["ans_full"]   # -> list[str]  (同样长度)
```

### 单轮模式

```jsonc
{
  "session_id": "string",
  "image": Image(),          // PIL.Image 或 None
  "image_url": "string",     // 当 'image' 已提供时为空
  "turns": {
    "interaction_id": ["string"],
    "domain": ["int"],      // 领域编码
    "query_category": ["int"], // 问题类型编码
    "dynamism": ["int"],    // 动态性编码
    "query": ["string"],
    "image_quality": ["int"] // 画质编码
  },
  "answers": {
    "interaction_id": ["string"],
    "ans_full": ["string"]
  }
}
```

### 多轮模式

```jsonc
{
  "session_id": "string",
  "image": Image(),
  "image_url": "string",
  "turns": {
    "interaction_id": ["string", ...],
    "domain": ["int", ...],
    "query_category": ["int", ...],
    "dynamism": ["int", ...],
    "query": ["string", ...],
    "image_quality": ["int", ...]
  },
  "answers": {
    "interaction_id": ["string", ...],
    "ans_full": ["string", ...]
  }
}
```

- `interaction_id` 将每个问题与其答案对齐
- `domain`、`query_category`、`dynamism`、`image_quality` 是整数编码的分类标签
- 保证 `image` 或 `image_url` 中至少有一个存在

---

## 4. 快速访问示例

```python
from datasets import load_dataset

# --- 加载数据集 ---------------------------------------------------------
st = load_dataset("crag-mm-2025/crag-mm-single-turn-public",
                  split="validation", revision="v0.1.2")
mt = load_dataset("crag-mm-2025/crag-mm-multi-turn-public",
                  split="validation", revision="v0.1.2")

# --- 遍历轮次，适配不同模式 ---------------------------------------------
def iter_turns(sample):
    """返回单轮或多轮行中的 (turn_dict, answer_dict) 对"""
    if isinstance(sample["turns"], dict):
        n = len(sample["turns"]["interaction_id"])
        for i in range(n):
            turn =  {k: v[i] for k, v in sample["turns"].items()}
            ans  =  {k: v[i] for k, v in sample["answers"].items()}
            yield turn, ans
    else:  # 仅旧版本
        for turn, ans in zip(sample["turns"], sample["answers"]):
            yield turn, ans

# 查看第一个多轮对话
for t, a in iter_turns(mt[0]):
    print(f"Q: {t['query']}\nA: {a['ans_full']}\n")

# 显示（可能已降采样的）图像
import matplotlib.pyplot as plt
plt.imshow(st[0]["image"])
plt.axis("off")
plt.show()
```

> **第二阶段提示**：如果你将原始像素输入模型，请在预处理前将第一人称输入调整为 `width=960, height=1280`，以确保你的流程与评估条件一致。

---

## 5. 无需编码的用户 — 好消息

如果你使用仓库中提供的 `crag_batch_iterator.py`（已更新），不需要任何代码更改。该迭代器透明地：

- 接受列表字典（v0.1.1）和列字典（v0.1.2）两种布局
- 当只有 `image_url` 存在时自动下载图像
- 将第一人称图片调整为 960 × 1280

拉取最新提交并继续训练即可。

---

## 6. 许可证与引用

- **许可证**：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0)
- **引用**：

```bibtex
@inproceedings{crag-mm-2025,
  title  = {CRAG-MM: A Comprehensive RAG Benchmark for Multi-modal, Multi-turn Question Answering},
  author = {CRAG-MM Team},
  year   = {2025},
  url    = {https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025}
}
```
