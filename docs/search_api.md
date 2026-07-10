# CRAG-MM 搜索 API

CRAG-MM 是一个关注检索增强生成（RAG）事实性的视觉问答基准测试。它提供了独特的图像和问答集合，用于全面评估可穿戴设备的性能。

## CRAG-MM 搜索 API 描述

CRAG-MM 搜索 API 是一个 Python 库，提供统一的图像和文本搜索接口。它支持图像和文本查询，可以从给定的检索内容集合中检索相关信息。

**图像搜索 API** 使用 CLIP 嵌入来编码图像。它以图像或图像 URL 作为输入，返回相似图像列表及其包含实体的相关信息。相似度由嵌入的余弦相似度决定。见下方"图像搜索"示例。

**网页搜索 API** 使用 chromadb 全文搜索为预抓取的网页搜索结果构建索引。它以文本查询作为输入，返回相关网页的 URL 和元数据（如页面标题和页面片段）。你可以根据 URL 下载网页内容，并利用这些信息构建检索增强生成（RAG）系统。网页相关性基于余弦相似度计算。见下方"文本查询搜索"示例。

## 安装

```bash
pip install cragmm-search-pipeline==0.5.0
```

## 使用方法

### Task 1（仅图像搜索）

```python
from cragmm_search.search import UnifiedSearchPipeline

# 仅启用图像搜索 API，Task 1 不启用网页搜索
## validation 分片
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
)
```

### Task 2 & 3（图像 + 网页搜索）

```python
from cragmm_search.search import UnifiedSearchPipeline

# 同时启用图像和网页搜索 API
## validation 分片
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
)

## public_test 分片
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
)

# 可选，指定索引标签。默认为 "main"，推荐始终使用默认值
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
    image_hf_dataset_tag="main",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
    web_hf_dataset_tag="v0.5",
)
```

### 图像搜索

```python
# 使用 PIL 图像作为输入（也可使用 image_url）
import requests
from PIL import Image
from io import BytesIO
from crag_image_loader import ImageLoader

image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg"
image = ImageLoader(image_url).get_image()

results = search_pipeline(image, k=2)
assert results is not None, "未找到结果"
print(f"图像搜索结果: '{image_url}'\n")

for result in results:
    print(result)
    print('\n')
```

#### 输出示例

```
图像搜索结果: 'https://upload.wikimedia.org/...'

{'index': 17030, 'score': 0.9064, 'url': 'https://...', 
 'entities': [{'entity_name': '8 Spruce Street', 
   'entity_attributes': {
     'name': '8 Spruce Street (New York by Gehry)',
     'address': '8 Spruce Street, Manhattan, New York, U.S. 10038',
     'architect': 'Frank Gehry',
     'roof': '870 ft (265 m)',
     'floor_count': '76',
     'completion_date': '2010',
     ...
   }}]}
```

返回结果中：
- `score`：CLIP 余弦距离，**越小越相似**
- `entities`：知识图谱中关联的实体，每个包含 `entity_name` 和 `entity_attributes`（结构化属性字典）

### 网页文本搜索

```python
# 使用文本查询搜索
text_query = 'What to know about Andrew Cuomo?'
results = search_pipeline(text_query, k=2)
assert results is not None, "未找到结果"
print(f"网页搜索结果: '{text_query}'\n")

for result in results:
    print(result)
    print('\n')
```

#### 输出示例

```
网页搜索结果: 'What to know about Andrew Cuomo?'

{'index': 'https://en.wikipedia.org/wiki/Mario_Cuomo_chunk_2', 
 'score': 0.5728, 
 'page_name': 'Mario Cuomo - Wikipedia', 
 'page_snippet': 'He vigorously attacked Ronald Reagan\'s ... brought him to national attention...',
 'page_url': 'https://en.wikipedia.org/wiki/Mario_Cuomo'}
```

**注意**：搜索 API 仅返回图像和网页的 URL，而非完整内容。要获取完整的网页内容和图像，需要自行下载。在比赛期间，参赛者可以假设这些 URL 是可访问的。

#### 获取完整页面内容

我们提供了一个辅助类来获取页面的完整内容：

```python
from crag_web_result_fetcher import WebSearchResult

# 使用文本查询搜索
text_query = 'What to know about Andrew Cuomo?'
results = search_pipeline(text_query, k=2)
assert results is not None, "未找到结果"

for result in results:
    result = WebSearchResult(result)
    print(result["page_content"])  # 打印完整的页面内容
```
