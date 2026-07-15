"""
Task 2: Multi-Source Augmentation Agent
========================================
专为 Mac (Apple Silicon) 设计的 Task 2 Agent。

与 Task 1 的核心区别：
  Task 1: 图像 → KG 检索 → 生成答案  （单源）
  Task 2: 图像 → KG 检索 + Web 搜索 → 融合 → 生成答案  （多源）

工作流程：
  1. 查询分析：判断问题类型（视觉识别 vs 知识型 vs 推理型）
  2. 图片 → search_pipeline(image) → KG 结构化数据 + 分数过滤
  3. 文本查询 → search_pipeline(text) → Web 搜索结果（新增！）
  4. 双源融合：KG 实体属性 + Web 片段，去噪排序
  5. 分层 Prompt：图片 + KG 数据 + Web 数据 + 问题
  6. VLM 生成答案
  7. 后处理：置信度判断，不确定时说 "I don't know"

技术栈：
  - 检索：cragmm-search-pipeline (CLIP 图像KG + BGE 文本Web)
  - 推理：mlx-vlm + Qwen2-VL-2B-Instruct-4bit (Apple Silicon 原生)
"""

from typing import Dict, List, Any, Tuple, Optional
import re
import time

import numpy as np
from PIL import Image
from shared.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline


# ============================================================
# 文本清洗工具 (复用 Task 1)
# ============================================================

def clean_attr_value(text: str) -> str:
    """清洗 KG 属性值中的 Wiki 标记等噪声。"""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[([^\]|]+?)\]\]', r'\1', text)
    text = re.sub(r'\[\[[^\]]+\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r"'''?([^']+)'''?", r'\1', text)
    text = re.sub(r'<br\s*/?>', ', ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text


def is_noisy_attribute(key: str, value: str) -> bool:
    """判断属性值是否纯噪声。"""
    if re.search(r'[\{\}\[\]<>|]', value):
        return True
    if key.lower() in ("logo", "image", "track_map", "coordinates", "website", "image_caption"):
        return True
    if len(value.strip()) < 2:
        return True
    return False


# ============================================================
# 可调参数
# ============================================================

# --- 知识图谱检索 ---
NUM_KG_RECALL = 10
NUM_KG_FINAL = 5
SIMILARITY_THRESHOLD_STRICT = 0.65
SIMILARITY_THRESHOLD_LOOSE = 0.85

# --- Web 搜索 (新增) ---
NUM_WEB_RECALL = 5        # Web 搜索召回数量
NUM_WEB_FINAL = 3          # 最终使用的 Web 结果数量

# --- 生成参数 ---
MAX_GENERATION_TOKENS = 128
BATCH_SIZE = 4
TEMPERATURE = 0.0

# --- 模型 ---
VLM_MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


# ============================================================
# 查询类型分类 (复用 Task 1)
# ============================================================

def classify_query_type(query: str) -> Dict[str, Any]:
    """
    基于关键词规则分类查询类型。

    返回:
        {
            "needs_knowledge": bool,
            "is_comparison": bool,
            "is_reasoning": bool,
            "is_visual_only": bool,
            "query_type": str,  # "visual" | "knowledge" | "reasoning" | "comparison"
        }
    """
    query_lower = query.lower()

    knowledge_keywords = [
        "brand", "price", "model", "year", "made in", "country",
        "author", "director", "artist", "publisher", "manufacturer",
        "ingredient", "nutrition", "calories", "material", "release",
        "species", "breed", "variety", "scientific name",
        "what brand", "what company", "who made", "who wrote",
        "how much", "how many calories", "what year",
    ]
    comparison_keywords = [
        "compare", "difference", "versus", "vs", "better",
        "cheaper", "more expensive", "which one",
    ]
    reasoning_keywords = [
        "why", "how does", "how do", "explain", "reason",
        "cause", "because", "what if", "can i", "should i",
        "is it safe", "is it possible", "would",
    ]
    visual_keywords = [
        "what color", "what is in the", "what is on the",
        "describe", "what do you see", "identify",
        "what object", "what animal", "what plant",
    ]

    needs_knowledge = any(kw in query_lower for kw in knowledge_keywords)
    is_comparison = any(kw in query_lower for kw in comparison_keywords)
    is_reasoning = any(kw in query_lower for kw in reasoning_keywords)
    is_visual_only = any(kw in query_lower for kw in visual_keywords)

    if is_comparison:
        query_type = "comparison"
    elif is_reasoning and needs_knowledge:
        query_type = "reasoning"
    elif needs_knowledge:
        query_type = "knowledge"
    elif is_reasoning:
        query_type = "reasoning"
    elif is_visual_only:
        query_type = "visual"
    else:
        query_type = "knowledge"

    return {
        "needs_knowledge": needs_knowledge or query_type in ("knowledge", "comparison"),
        "is_comparison": is_comparison,
        "is_reasoning": is_reasoning,
        "is_visual_only": is_visual_only,
        "query_type": query_type,
    }


# ============================================================
# Web 搜索结果处理 (新增)
# ============================================================

def _clean_web_snippet(text: str) -> str:
    """清洗 Web 搜索片段中的噪声。"""
    if not text:
        return ""
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 截断过长片段
    if len(text) > 300:
        text = text[:300] + "..."
    return text


def _is_noisy_web_result(snippet: str) -> bool:
    """判断 Web 搜索结果是否为纯噪声。"""
    if not snippet or len(snippet.strip()) < 10:
        return True
    # 纯 URL / 纯数字 / 纯符号
    if re.match(r'^[\d\s\.,;:!?\-_=/\\@#$%^&*()+|\[\]{}""''<>~`]+$', snippet):
        return True
    # 垃圾内容模式
    noise_patterns = [
        r'^(cookie|privacy|advertisement|subscribe|sign up|log in|404|error)',
        r'^[A-Za-z0-9]{20,}$',  # 纯哈希/ID
    ]
    for pat in noise_patterns:
        if re.search(pat, snippet, re.IGNORECASE):
            return True
    return False


def _build_web_search_query(
    query: str,
    kg_entity_names: List[str],
    query_type_info: Dict[str, Any],
) -> str:
    """
    构建高质量的 Web 搜索查询字符串。

    策略：
    - 如果 KG 有匹配实体 → "{entity_name} {query}" 增强精度
    - 如果 KG 无匹配 → 使用原问题
    - 视觉类问题可能不需要 Web 搜索
    """
    parts = []

    # 添加 KG 实体名作为搜索上下文
    if kg_entity_names:
        # 最多取前 2 个实体名
        entity_str = " ".join(kg_entity_names[:2])
        parts.append(entity_str)

    # 添加用户问题
    parts.append(query)

    return " ".join(parts)


# ============================================================
# 主 Agent 类
# ============================================================

class Task2MultiSourceAgent(BaseAgent):
    """
    Task 2: 多源增强 Agent

    核心改进 (相比 Task 1):
    - 双源检索：图像 KG + Web 搜索
    - 智能融合：KG 结构化数据 + Web 文本片段
    - 噪声过滤：Web 结果质量评估与去噪
    - 增强 Prompt：引导 VLM 区分和利用两个信息源
    - 置信度校准：综合考虑双源检索质量
    """

    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        model_name: str = VLM_MODEL_NAME,
        max_gen_len: int = MAX_GENERATION_TOKENS,
    ):
        super().__init__(search_pipeline)

        if search_pipeline is None:
            raise ValueError(
                "Task 2 需要 search_pipeline 来进行图像KG检索和Web搜索！"
            )

        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.model = None
        self.processor = None

        print(f"[Task2 Agent] 正在加载 VLM 模型: {model_name} ...")
        self._initialize_vlm()

    def _initialize_vlm(self):
        """使用 MLX 加载视觉语言模型 (Apple Silicon 原生加速)。"""
        from mlx_vlm import load

        self.model, self.processor = load(self.model_name)
        print(f"[Task2 Agent] ✅ 模型加载完成: {self.model_name}")

    def get_batch_size(self) -> int:
        return BATCH_SIZE

    # ============================================================
    # Step 1: 图像 KG 检索 (复用 Task 1 逻辑)
    # ============================================================

    def _retrieve_and_filter_kg(
        self,
        image: Image.Image,
        query_type_info: Dict[str, Any],
    ) -> Tuple[List[Dict], float]:
        """
        图片 → KG 检索 + 分数过滤。

        Returns:
            (filtered_results, best_score)
        """
        try:
            results = self.search_pipeline(image, k=NUM_KG_RECALL)
            if not results:
                return [], 1.0

            filtered = []
            best_score = 1.0

            for r in results:
                score = r.get("score", 1.0)
                threshold = (
                    SIMILARITY_THRESHOLD_STRICT
                    if query_type_info["needs_knowledge"]
                    else SIMILARITY_THRESHOLD_LOOSE
                )
                if score < threshold:
                    filtered.append(r)
                    if score < best_score:
                        best_score = score

            filtered = filtered[:NUM_KG_FINAL]
            return filtered, best_score

        except Exception as e:
            print(f"[Task2 Agent] ⚠️ KG 检索失败: {e}")
            return [], 1.0

    # ============================================================
    # Step 1.5: KG 实体名提取 (供 Web 搜索使用)
    # ============================================================

    def _extract_entity_names(self, kg_results: List[Dict]) -> List[str]:
        """从 KG 结果中提取实体名称列表。"""
        names = []
        for result in kg_results:
            for entity in result.get("entities", []):
                name = entity.get("entity_name", "")
                if name and name not in names:
                    names.append(name)
        return names

    # ============================================================
    # Step 2: KG 文本提取 (复用 Task 1 逻辑)
    # ============================================================

    QUERY_TO_KG_FIELD_MAP = {
        "price": ["price", "msrp", "list_price", "retail_price", "cost"],
        "cost": ["price", "msrp", "list_price", "retail_price", "cost"],
        "how much": ["price", "msrp", "list_price", "retail_price", "cost"],
        "seat": ["number_of_seats", "seating_capacity", "passenger_capacity", "capacity"],
        "passenger": ["number_of_seats", "seating_capacity", "passenger_capacity"],
        "brand": ["brand", "manufacturer", "name"],
        "manufacturer": ["manufacturer", "brand", "name"],
        "who made": ["manufacturer", "brand", "author", "artist", "publisher"],
        "year": ["production_years", "production_start_year", "release_date", "model_year", "year"],
        "when": ["production_start", "release_date", "opened", "constructed", "built"],
        "release": ["release_date", "production_start_year", "model_year"],
        "construction": ["constructed", "built", "broke_ground", "opened"],
        "where": ["location", "country", "country_of_origin", "assembly_locations", "address"],
        "location": ["location", "country", "address"],
        "country": ["country", "country_of_origin", "location"],
        "color": ["color", "colour"],
        "model": ["model", "name", "body_style", "body_type", "class"],
        "engine": ["engine_type", "engine", "engine_displacement", "drive_type"],
        "length": ["length", "width", "height", "wheelbase"],
        "weight": ["curb_weight", "weight", "mass"],
        "fuel": ["fuel_capacity", "fuel_economy", "mpg", "fuel_type"],
    }

    def _extract_kg_text(
        self,
        kg_results: List[Dict],
        query: str = "",
    ) -> Tuple[str, int, float]:
        """
        从 KG 检索结果中提取结构化信息。

        Returns:
            (context_text, num_valid_entities, avg_score)
        """
        if not kg_results:
            return "", 0, 1.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        all_blocks = []
        total_score = 0.0
        num_valid = 0

        for result in kg_results:
            entities = result.get("entities", [])
            score = result.get("score", 1.0)

            if not entities:
                continue

            for entity in entities:
                attrs = entity.get("entity_attributes", {})
                entity_name = entity.get("entity_name", "")
                if not attrs:
                    continue

                attr_items = []
                for key, value in attrs.items():
                    text = str(value).strip()
                    text = clean_attr_value(text)
                    if is_noisy_attribute(key, text):
                        continue
                    key_words = set(key.lower().replace("_", " ").split())
                    relevance = len(key_words & query_words)
                    attr_items.append((relevance, key, text))

                if not attr_items:
                    continue

                attr_items.sort(key=lambda x: (-x[0], x[1]))
                attr_items = attr_items[:8]

                lines = [f"Entity: {entity_name}"]
                for _, key, text in attr_items:
                    lines.append(f"  {key}: {text}")
                all_blocks.append("\n".join(lines))
                total_score += 1.0 - score
                num_valid += 1

        if not all_blocks:
            return "", 0, 1.0

        seen = set()
        unique_blocks = []
        for block in all_blocks:
            if block not in seen:
                seen.add(block)
                unique_blocks.append(block)

        context = "\n\n".join(unique_blocks[:3])
        avg_score = total_score / num_valid if num_valid > 0 else 0.0
        return context, num_valid, avg_score

    def _extract_candidate_from_kg(
        self, query: str, kg_results: List[Dict]
    ) -> Optional[str]:
        """尝试直接从 KG 属性中提取与问题匹配的值。"""
        query_lower = query.lower()
        target_fields = set()
        for keyword, fields in self.QUERY_TO_KG_FIELD_MAP.items():
            if keyword in query_lower:
                target_fields.update(fields)
        if not target_fields:
            return None

        candidates = []
        for result in kg_results:
            result_score = result.get("score", 1.0)
            for entity in result.get("entities", []):
                attrs = entity.get("entity_attributes") or {}
                for key, value in attrs.items():
                    key_lower = key.lower()
                    for target in target_fields:
                        if target in key_lower:
                            text = clean_attr_value(str(value))
                            if text and len(text) >= 2:
                                match_score = (1.0 - result_score) * (
                                    1.5 if target == key_lower else 1.0
                                )
                                candidates.append((match_score, text, key))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        best_score, best_value, _ = candidates[0]
        return best_value if best_score > 0.25 else None

    # ============================================================
    # Step 3: Web 搜索 (核心新增功能)
    # ============================================================

    def _retrieve_and_filter_web(
        self,
        query: str,
        kg_entity_names: List[str],
        query_type_info: Dict[str, Any],
    ) -> Tuple[List[Dict], str]:
        """
        Step 3: 文本查询 → Web 搜索 → 结果过滤 (新增)

        流程:
        1. 构建搜索查询词（融合 KG 实体名 + 用户问题）
        2. 调用 search_pipeline(text_query) 进行 Web 检索
        3. 基于 snippet 质量过滤噪声结果

        Args:
            query: 用户原始问题
            kg_entity_names: 从 KG 中提取的实体名列表
            query_type_info: 查询类型信息

        Returns:
            (filtered_web_results, web_context_str)
        """
        # 视觉类问题通常不需要 Web 搜索
        if query_type_info["is_visual_only"]:
            return [], ""

        # 构建搜索查询
        search_query = _build_web_search_query(query, kg_entity_names, query_type_info)

        try:
            # 调用 search_pipeline 进行文本搜索（返回 Web 结果）
            results = self.search_pipeline(search_query, k=NUM_WEB_RECALL)
            if not results:
                return [], ""

            # 过滤和清洗 Web 结果
            filtered = []
            for r in results:
                snippet = r.get("page_snippet", "")
                snippet = _clean_web_snippet(snippet)
                if _is_noisy_web_result(snippet):
                    continue
                # 创建清洗后的结果副本
                cleaned = dict(r)
                cleaned["page_snippet"] = snippet
                filtered.append(cleaned)

            # 按相关性分数排序（如果有的话），取 Top-K
            filtered = sorted(
                filtered,
                key=lambda x: x.get("score", 0.0),
                reverse=True,
            )
            filtered = filtered[:NUM_WEB_FINAL]

            return filtered, search_query

        except Exception as e:
            print(f"[Task2 Agent] ⚠️ Web 搜索失败: {e}")
            return [], ""

    def _format_web_context(self, web_results: List[Dict]) -> str:
        """
        将 Web 搜索结果格式化为可嵌入 Prompt 的文本。
        """
        if not web_results:
            return ""

        blocks = []
        for i, r in enumerate(web_results):
            snippet = r.get("page_snippet", "")
            page_name = r.get("page_name", "")
            score = r.get("score", 0.0)

            if not snippet:
                continue

            # 构建格式化的信息块
            header = f"[Web {i + 1}]"
            if page_name:
                header += f" {page_name}"
            header += f" (relevance: {score:.2f})"

            blocks.append(f"{header}\n{snippet}")

        return "\n\n".join(blocks)

    # ============================================================
    # Step 4: 双源融合 + Prompt 构建
    # ============================================================

    def _build_answer_prompt(
        self,
        query: str,
        kg_context: str,
        web_context: str,
        query_type_info: Dict[str, Any],
        retrieval_quality: Dict[str, Any],
    ) -> List[Dict]:
        """
        构建融合双源信息的 Prompt。

        与 Task 1 的区别:
        - 新增 Web 搜索上下文区域
        - 引导 VLM 优先采信 KG 结构化数据，Web 数据用于补充
        - 告知 VLM Web 数据可能含噪声，需谨慎筛选
        """

        # ——— System Prompt ———
        system_instruction = (
            "Answer questions using the image, Knowledge Graph (KG) data, and Web search results. "
            "Answer in ONE short phrase or sentence. Never make up facts.\n"
            "\n"
            "Source priority:\n"
            "- KG data is structured and reliable — use it first for factual answers.\n"
            "- Web data may contain noise or irrelevant info — verify before using.\n"
            "- If sources conflict, trust KG over Web.\n"
            "- If the image shows the answer directly, use it regardless of sources.\n"
            "\n"
            "Examples:\n"
            "KG: Entity: Organic Valley Milk\n  brand: Organic Valley\nWeb: [Web 1] Organic Valley is a popular brand...\n"
            "Q: what brand is this milk?\nA: Organic Valley\n\n"
            "KG: Entity: Toyota Camry\n  number_of_seats: 5\nWeb: [Web 1] Camry review...5 seats...\n"
            "Q: how many seats does this car have?\nA: 5\n\n"
            "KG: (no data)\nWeb: [Web 1] The painting was created in 1889 by Van Gogh...\n"
            "Q: who painted this?\nA: Van Gogh\n\n"
            "KG: Entity: Unknown Item\n  weight: 2kg\nWeb: (no relevant info)\n"
            "Q: what is the price?\nA: I don't know\n"
            "\n"
            "Key rules:\n"
            "- Extract the EXACT value from KG if it answers the question.\n"
            "- Use Web info only to supplement what KG doesn't have.\n"
            '- If neither source nor image can answer, say "I don\'t know".\n'
            "- Never output raw data. Always give a natural answer."
        )

        # ——— User Message ———
        parts = []
        if kg_context:
            parts.append(f"[KG Data]\n{kg_context}")
        if web_context:
            parts.append(f"[Web Search Results]\n{web_context}")
        parts.append(f"Question: {query}")

        user_message = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_message},
        ]

        return messages

    # ============================================================
    # Step 5: VLM 生成 (复用 Task 1 逻辑)
    # ============================================================

    def _generate_single_answer(
        self,
        query: str,
        image: Image.Image,
        messages: List[Dict],
    ) -> str:
        """使用 VLM 生成单条答案。"""
        from mlx_vlm import generate

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            response = generate(
                self.model,
                self.processor,
                prompt=prompt,
                image=image,
                max_tokens=self.max_gen_len,
                temp=TEMPERATURE,
                verbose=False,
            )
            answer = response.text.strip()
            answer = self._post_process_answer(answer)
            return answer

        except Exception as e:
            print(f"[Task2 Agent] ⚠️ 生成失败: {e}")
            return "I don't know"

    def _post_process_answer(self, answer: str) -> str:
        """答案后处理。"""
        if not answer:
            return "I don't know"

        prefixes_to_strip = [
            "answer:", "the answer is:", "response:",
            "assistant:", "answer is:",
        ]
        for prefix in prefixes_to_strip:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()

        if len(answer) > 300:
            truncated = answer[:300]
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")
            cut_point = max(last_period, last_newline, 250)
            answer = answer[:cut_point].strip().rstrip(".").strip()

        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot determine",
            "i can't determine", "i am not sure", "i am unable",
            "i'm unable", "cannot be determined", "unknown",
            "not sure", "unsure", "no information",
            "i cannot answer", "i can't answer",
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            return "I don't know"

        return answer

    # ============================================================
    # Step 6: 置信度评估 (增强版——考虑双源)
    # ============================================================

    def _assess_confidence(
        self,
        answer: str,
        retrieval_quality: Dict[str, Any],
        query_type_info: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """
        置信度评估 (增强版)。

        新增考虑:
        - Web 搜索结果的质量
        - 双源互补情况下的置信度提升
        """
        if answer.lower().strip() == "i don't know":
            return answer, False

        best_score = retrieval_quality.get("best_score", 1.0)
        has_good_match = retrieval_quality.get("has_good_match", False)
        num_entities = retrieval_quality.get("num_entities", 0)
        has_web_results = retrieval_quality.get("has_web_results", False)
        needs_knowledge = query_type_info["needs_knowledge"]

        # 知识型问题 + 无 KG 且无 Web → 高风险幻觉
        if needs_knowledge and not has_good_match and num_entities == 0:
            if not has_web_results:
                print(f"  [Confidence] 知识型问题无双源匹配 → I don't know")
                return "I don't know", True

        # 最佳 KG 匹配分数太差 且 Web 也无结果
        if best_score > SIMILARITY_THRESHOLD_LOOSE and needs_knowledge:
            if not has_web_results:
                print(
                    f"  [Confidence] 最佳KG分数 {best_score:.3f} > 阈值，"
                    f"且无Web结果 → I don't know"
                )
                return "I don't know", True

        return answer, False

    # ============================================================
    # 批处理接口 (评估框架调用的入口)
    # ============================================================

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        批量生成回答 — Task 2 主入口。

        增强版流程 (相比 Task 1):
        1. 查询分析
        2. 图片检索 KG + 分数过滤
        3. 文本检索 Web + 噪声过滤  ← 新增
        4. 提取 KG 文本 + 格式化 Web 结果
        5. 双源融合 Prompt
        6. VLM 生成
        7. 答案后处理 + 置信度评估
        """
        batch_start = time.time()
        print(f"\n[Task2 Agent] ===== 处理 {len(queries)} 条查询 (多源增强) =====")

        responses = []
        idk_count = 0

        for i, (query, image, history) in enumerate(
            zip(queries, images, message_histories)
        ):
            turn_start = time.time()

            # ---- Step 1: 查询类型分析 ----
            query_type_info = classify_query_type(query)

            # ---- Step 2: 图片 → KG 检索 + 分数过滤 ----
            kg_results, best_score = self._retrieve_and_filter_kg(
                image, query_type_info
            )

            # ---- Step 2.5: 提取 KG 实体名 (供 Web 搜索用) ----
            kg_entity_names = self._extract_entity_names(kg_results)

            # ---- Step 3: Web 搜索 (新增) ----
            web_results, search_query_used = self._retrieve_and_filter_web(
                query, kg_entity_names, query_type_info
            )
            web_context = self._format_web_context(web_results)

            # ---- Step 4: 提取 KG 文本 ----
            kg_context, num_entities, avg_score = self._extract_kg_text(
                kg_results, query
            )

            # ---- Step 4.5: 尝试规则快速提取 ----
            candidate = self._extract_candidate_from_kg(query, kg_results)

            # 评估双源检索质量
            max_score_threshold = (
                SIMILARITY_THRESHOLD_STRICT
                if query_type_info["needs_knowledge"]
                else SIMILARITY_THRESHOLD_LOOSE
            )
            retrieval_quality = {
                "best_score": best_score,
                "num_entities": num_entities,
                "avg_score": avg_score,
                "has_good_match": best_score < max_score_threshold and num_entities > 0,
                "num_kg_results": len(kg_results),
                # 新增 Web 质量指标
                "num_web_results": len(web_results),
                "has_web_results": len(web_results) > 0,
                "search_query_used": search_query_used,
            }

            # ---- Step 5: 双源融合 Prompt + VLM 生成 ----
            messages = self._build_answer_prompt(
                query, kg_context, web_context, query_type_info, retrieval_quality
            )

            raw_answer = self._generate_single_answer(query, image, messages)
            cleaned_answer = self._post_process_answer(raw_answer)

            # ---- Step 6: 置信度评估 ----
            final_answer, was_overridden = self._assess_confidence(
                cleaned_answer, retrieval_quality, query_type_info
            )

            # ---- Step 7: 规则提取答案优先 ----
            if candidate and not was_overridden:
                vlm_is_idk = final_answer.strip().lower() == "i don't know"
                vlm_is_junk = len(final_answer) < 3 or final_answer.count(",") > 4
                if vlm_is_idk or vlm_is_junk:
                    final_answer = candidate
                    was_overridden = False

            responses.append(final_answer)

            if final_answer.strip().lower() == "i don't know":
                idk_count += 1

            # 日志
            turn_time = time.time() - turn_start
            kg_count = len(kg_results)
            web_count = len(web_results)
            kg_icon = "✅" if retrieval_quality["has_good_match"] else "⚠️"
            web_icon = "🌐" if web_count > 0 else "  "
            qtype = query_type_info["query_type"][:4]
            override_tag = " [OVERRIDE→IDK]" if was_overridden else ""
            print(
                f"  [{i + 1}/{len(queries)}] {kg_icon}{web_icon} "
                f"type={qtype} | "
                f"KG={kg_count}条/{num_entities}实体 | "
                f"Web={web_count}条 | "
                f"KG_score={best_score:.3f} | "
                f"Q: {query[:40]}... | A: {final_answer[:50]}..."
                f"{override_tag} | ⏱ {turn_time:.1f}s"
            )

        total_time = time.time() - batch_start
        idk_rate = idk_count / len(queries) * 100 if queries else 0
        print(
            f"[Task2 Agent] ===== 完成！总耗时 {total_time:.1f}s, "
            f"平均 {total_time / len(queries):.1f}s/条, "
            f"IDK率: {idk_rate:.1f}% ({idk_count}/{len(queries)}) =====\n"
        )

        return responses
