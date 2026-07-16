"""
Task 1: Single-Source Augmentation Agent (Enhanced)
=====================================================
专为 Mac (Apple Silicon) 设计的 Task 1 Agent。

工作流程（增强版）：
  1. 查询分析：判断问题类型（视觉识别 vs 知识型 vs 推理型）
  2. 图片 → search_pipeline (image search) → 知识图谱结构化数据
  3. 基于相似度分数过滤低质量 KG 结果
  4. 提取实体属性 (description, caption, summary 等)，按分数加权
  5. 构建分层 Prompt: 图片 + KG 数据 + 问题 + 查询类型指导
  6. VLM 生成答案
  7. 后处理：置信度判断，不确定时说 "I don't know"

技术栈：
  - 检索: cragmm-search-pipeline (CLIP + BGE)
  - 推理: mlx-vlm + Qwen2-VL-2B-Instruct-4bit (Apple Silicon 原生)
"""

from typing import Dict, List, Any, Tuple, Optional
import re
import time

import numpy as np
from PIL import Image
from shared.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline


# ============================================================
# 文本清洗工具
# ============================================================

def clean_attr_value(text: str) -> str:
    """清洗 KG 属性值中的 Wiki 标记等噪声。"""
    # 去除 Wiki 标记
    text = re.sub(r'\{\{[^}]*\}\}', '', text)      # {{coord|...}}, {{convert|...}}
    text = re.sub(r'\[\[([^\]|]+?)\]\]', r'\1', text)  # [[Dallara DW12]] → Dallara DW12
    text = re.sub(r'\[\[[^\]]+\|([^\]]+)\]\]', r'\1', text)  # [[link|text]] → text
    # 去除 HTML/Wiki 格式标记
    text = re.sub(r"'''?([^']+)'''?", r'\1', text)  # '''bold'''
    text = re.sub(r'<br\s*/?>', ', ', text)         # <br /> → ", "
    text = re.sub(r'<[^>]+>', '', text)             # 其他 HTML 标签
    text = re.sub(r'&[a-z]+;', ' ', text)           # HTML entities
    # 压缩多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 截断过长值
    if len(text) > 200:
        text = text[:200] + "..."
    return text


def is_noisy_attribute(key: str, value: str) -> bool:
    """判断属性值是否纯噪声（应被跳过）。"""
    # 包含残留的 wiki 标记
    if re.search(r'[\{\}\[\]<>|]', value):
        return True
    # 纯坐标、URL 等无用信息
    if key.lower() in ("logo", "image", "track_map", "coordinates", "website", "image_caption"):
        return True
    # 值太短
    if len(value.strip()) < 2:
        return True
    return False

# ============================================================
# 可调参数 — 在这里调整你的 Agent 行为
# ============================================================

# 知识图谱检索数量 (先多召回，再过滤)
NUM_KG_RECALL = 10
NUM_KG_FINAL = 5

# 相似度分数阈值 — CLIP Cosine 距离: 越小越相似 (0=identical, 1=orthogonal)
# distance < SIMILARITY_THRESHOLD_STRICT: 高置信度匹配
# distance < SIMILARITY_THRESHOLD_LOOSE: 低置信度匹配
# distance >= SIMILARITY_THRESHOLD_LOOSE: 噪声，丢弃
# 注：实际数据最佳匹配通常在 0.55-0.65 范围内
SIMILARITY_THRESHOLD_STRICT = 0.65
SIMILARITY_THRESHOLD_LOOSE = 0.85

# 最大生成长度 (token) — 评估会自动截断为 75 BPE tokens
MAX_GENERATION_TOKENS = 128

# 批处理大小 (Mac 上建议小一些)
BATCH_SIZE = 4

# MLX VLM 模型 (4-bit 量化)
# 按硬件选择：
#   M1/M2/M3 16GB → Qwen2-VL-2B (~2GB, 最快)
#   M2/M3 Pro 18GB+ → Qwen2-VL-7B (~5GB, 推荐，首次需下载)
#   M3 Max 32GB+ → Qwen2-VL-7B 或 Llama-3.2-11B-Vision (~8GB)
VLM_MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
# VLM_MODEL_NAME = "mlx-community/Qwen2-VL-7B-Instruct-4bit"  # 7B，下载后取消注释
# VLM_MODEL_NAME = "mlx-community/Llama-3.2-Vision-Instruct-4bit"  # 需要 32GB+

# 生成温度 (0 = 确定性最高)
TEMPERATURE = 0.0


# ============================================================
# 查询类型分类 (关键词规则 — 轻量级，无需独立路由器模型)
# ============================================================

def classify_query_type(query: str) -> Dict[str, Any]:
    """
    基于关键词规则分类查询类型，用于指导后续的检索和生成策略。

    返回:
        {
            "needs_knowledge": bool,     # 是否需要外部知识
            "is_comparison": bool,       # 是否是比较类问题
            "is_reasoning": bool,        # 是否需要推理
            "is_visual_only": bool,      # 是否纯视觉识别
            "query_type": str,           # "visual" | "knowledge" | "reasoning" | "comparison"
        }
    """
    query_lower = query.lower()

    # 知识型关键词（需要外部知识库）
    knowledge_keywords = [
        "brand", "price", "model", "year", "made in", "country",
        "author", "director", "artist", "publisher", "manufacturer",
        "ingredient", "nutrition", "calories", "material", "release",
        "species", "breed", "variety", "scientific name",
        "what brand", "what company", "who made", "who wrote",
        "how much", "how many calories", "what year",
    ]

    # 比较型关键词
    comparison_keywords = [
        "compare", "difference", "versus", "vs", "better",
        "cheaper", "more expensive", "which one",
    ]

    # 推理型关键词
    reasoning_keywords = [
        "why", "how does", "how do", "explain", "reason",
        "cause", "because", "what if", "can i", "should i",
        "is it safe", "is it possible", "would",
    ]

    # 纯视觉识别关键词
    visual_keywords = [
        "what color", "what is in the", "what is on the",
        "describe", "what do you see", "identify",
        "what object", "what animal", "what plant",
    ]

    needs_knowledge = any(kw in query_lower for kw in knowledge_keywords)
    is_comparison = any(kw in query_lower for kw in comparison_keywords)
    is_reasoning = any(kw in query_lower for kw in reasoning_keywords)
    is_visual_only = any(kw in query_lower for kw in visual_keywords)

    # 确定查询类型
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
        # 默认为知识型（大多数 CRAG 问题需要外部知识）
        query_type = "knowledge"

    return {
        "needs_knowledge": needs_knowledge or query_type in ("knowledge", "comparison"),
        "is_comparison": is_comparison,
        "is_reasoning": is_reasoning,
        "is_visual_only": is_visual_only,
        "query_type": query_type,
    }


# ============================================================
# 主 Agent 类
# ============================================================

class Task1SingleSourceAgent(BaseAgent):
    """
    Task 1: 单源增强 Agent (增强版)

    核心改进:
    - 查询分析：识别问题类型，适配检索策略
    - 分数阈值：基于 CLIP 相似度过滤低质量 KG 数据
    - 加权提取：高相似度的 KG 数据获得更高权重
    - 分层 Prompt：根据查询类型构建最优 Prompt
    - 置信度判断：检索质量不足时主动回复 "I don't know"
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
                "Task 1 需要 search_pipeline 来进行图像知识图谱检索！"
            )

        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.model = None
        self.processor = None

        print(f"[Task1 Agent] 正在加载 VLM 模型: {model_name} ...")
        self._initialize_vlm()

    def _initialize_vlm(self):
        """
        使用 MLX 加载视觉语言模型 (Apple Silicon 原生加速)
        """
        from mlx_vlm import load

        self.model, self.processor = load(self.model_name)
        print(f"[Task1 Agent] ✅ 模型加载完成: {self.model_name}")

    def get_batch_size(self) -> int:
        return BATCH_SIZE

    # ============================================================
    # Step 1: 查询分析 + 图片检索
    # ============================================================

    def _retrieve_and_filter_kg(
        self,
        image: Image.Image,
        query_type_info: Dict[str, Any],
    ) -> Tuple[List[Dict], float]:
        """
        Step 1: 图片 → KG 检索 + 分数过滤

        改进：
        - 多召回 (recall=10)，然后基于相似度分数过滤
        - 根据查询类型调整检索策略
        - 返回最佳相似度分数用于置信度判断

        Returns:
            (filtered_results, best_score)
        """
        try:
            results = self.search_pipeline(image, k=NUM_KG_RECALL)
            if not results:
                return [], 1.0  # 1.0 = 最差分数 (无结果)

            # 解析分数：CLIP 距离，越小越相似
            filtered = []
            best_score = 1.0

            for r in results:
                score = r.get("score", 1.0)

                # 严格模式：知识型问题要求更高质量
                if query_type_info["needs_knowledge"]:
                    threshold = SIMILARITY_THRESHOLD_STRICT
                else:
                    # 视觉识别可以放宽
                    threshold = SIMILARITY_THRESHOLD_LOOSE

                if score < threshold:
                    filtered.append(r)
                    if score < best_score:
                        best_score = score

            # 限制数量
            filtered = filtered[:NUM_KG_FINAL]

            return filtered, best_score

        except Exception as e:
            print(f"[Task1 Agent] ⚠️ 检索失败: {e}")
            return [], 1.0

    # ============================================================
    # Step 2: KG 文本提取（优化版 — 清晰属性列表格式）
    # ============================================================

    def _extract_kg_text(
        self,
        kg_results: List[Dict],
        query: str = "",
    ) -> Tuple[str, int, float]:
        """
        从 KG 检索结果中提取结构化信息，以 "实体名 + 属性列表" 清晰格式输出。

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

                # 过滤出有意义且不重复的属性，按与问题的相关性排序
                attr_items = []
                for key, value in attrs.items():
                    text = str(value).strip()
                    # 清洗噪声 + 跳过无意义属性
                    text = clean_attr_value(text)
                    if is_noisy_attribute(key, text):
                        continue
                    # 计算属性名与问题的相关性（简单的词重叠）
                    key_words = set(key.lower().replace("_", " ").split())
                    relevance = len(key_words & query_words)
                    attr_items.append((relevance, key, text))

                if not attr_items:
                    continue

                # 相关的属性排在前面，最多保留 8 个
                attr_items.sort(key=lambda x: (-x[0], x[1]))
                attr_items = attr_items[:8]

                # 构建该实体的文本块
                lines = [f"Entity: {entity_name}"]
                for _, key, text in attr_items:
                    lines.append(f"  {key}: {text}")
                all_blocks.append("\n".join(lines))
                total_score += 1.0 - score
                num_valid += 1

        if not all_blocks:
            return "", 0, 1.0

        # 去重 + 限制数量
        seen = set()
        unique_blocks = []
        for block in all_blocks:
            if block not in seen:
                seen.add(block)
                unique_blocks.append(block)

        context = "\n\n".join(unique_blocks[:3])  # 最多 3 个实体
        avg_score = total_score / num_valid if num_valid > 0 else 0.0
        return context, num_valid, avg_score

    # ============================================================
    # Step 2.5: 直接从 KG 提取候选答案（Rule-based 快速通道）
    # ============================================================

    # 问题关键词 → KG 属性名的映射表
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
                                match_score = (1.0 - result_score) * (1.5 if target == key_lower else 1.0)
                                candidates.append((match_score, text, key))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        best_score, best_value, _ = candidates[0]
        return best_value if best_score > 0.25 else None

    # ============================================================
    # Step 3: 分层 Prompt 构建（含 Few-shot 示例）
    # ============================================================

    def _build_answer_prompt(
        self,
        query: str,
        kg_context: str,
        query_type_info: Dict[str, Any],
        retrieval_quality: Dict[str, Any],
    ) -> List[Dict]:
        """
        构建 Prompt，包含 Few-shot 示例以指导 2B 模型行为。
        """

        # ——— System Prompt（精简指令 + Few-shot 示例）———
        system_instruction = (
            "Answer questions using the image and Knowledge Graph (KG) data. "
            "Answer in ONE short phrase or sentence. Never make up facts.\n"
            "\n"
            "Examples:\n"
            "KG: Entity: Organic Valley Milk\n  brand: Organic Valley\nQ: what brand is this milk?\nA: Organic Valley\n\n"
            "KG: Entity: Toyota Camry\n  number_of_seats: 5\nQ: how many seats does this car have?\nA: 5\n\n"
            "KG: (no data)\nQ: what color is the flower?\nA: purple\n\n"
            "KG: Entity: Eiffel Tower\n  construction_started: 1887\nQ: when was this built?\nA: 1887\n\n"
            "KG: Entity: Unknown Item\n  weight: 2kg\nQ: what is the price?\nA: I don't know\n"
            "\n"
            "Key rules:\n"
            "- Extract the EXACT value from KG if it answers the question.\n"
            "- If KG has related info but NOT the answer, check the image.\n"
            '- If neither image nor KG can answer, say "I don\'t know".\n'
            "- Never output raw KG data. Always give a natural answer."
        )

        # ——— User Message ———
        if kg_context:
            user_message = (
                f"KG data:\n{kg_context}\n\n"
                f"Question: {query}"
            )
        else:
            user_message = f"Question: {query}"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_message},
        ]

        return messages

    # ============================================================
    # Step 4: VLM 生成
    # ============================================================

    def _generate_single_answer(
        self,
        query: str,
        image: Image.Image,
        messages: List[Dict],
    ) -> str:
        """
        Step 4: 使用 VLM 生成单条答案
        """
        from mlx_vlm import generate

        try:
            # 使用 Qwen2-VL 的 chat template 构建 prompt
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
            # mlx_vlm.generate() 返回 GenerationResult 对象，需用 .text 提取文本
            answer = response.text.strip()

            # 后处理：清理常见问题
            answer = self._post_process_answer(answer)

            return answer

        except Exception as e:
            print(f"[Task1 Agent] ⚠️ 生成失败: {e}")
            return "I don't know"

    def _post_process_answer(self, answer: str) -> str:
        """
        Step 5: 答案后处理

        处理常见问题：
        - 截断过长答案
        - 移除重复内容
        - 清理模型有时会输出的前缀
        """
        if not answer:
            return "I don't know"

        # 移除常见的模型输出前缀（只移除纯元数据前缀，不误删实质性内容）
        prefixes_to_strip = [
            "answer:", "the answer is:", "response:",
            "assistant:", "answer is:",
        ]
        for prefix in prefixes_to_strip:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()

        # 截断到合理长度
        if len(answer) > 300:
            # 尝试在句子边界截断
            truncated = answer[:300]
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")
            cut_point = max(last_period, last_newline, 250)
            answer = answer[:cut_point].strip().rstrip(".").strip()

        # 如果答案本质上说的是不知道，统一格式
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
    # Step 5: 置信度判断 (检索质量 + 答案置信度)
    # ============================================================

    def _assess_confidence(
        self,
        answer: str,
        retrieval_quality: Dict[str, Any],
        query_type_info: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """
        Step 6: 置信度评估

        决定是否使用原始答案还是回复 "I don't know"。

        逻辑：
        1. 如果 VLM 已返回 "I don't know" → 保持不变
        2. 如果检索质量很低 且 问题是知识型 → "I don't know"
        3. 如果最佳匹配分数太差 → "I don't know"
        4. 否则 → 使用原答案

        Returns:
            (final_answer, was_modified)
        """
        # 如果 VLM 已经说了不知道，保持不变
        if answer.lower().strip() == "i don't know":
            return answer, False

        best_score = retrieval_quality.get("best_score", 1.0)
        has_good_match = retrieval_quality.get("has_good_match", False)
        num_entities = retrieval_quality.get("num_entities", 0)
        needs_knowledge = query_type_info["needs_knowledge"]

        # 知识型问题 + 无 KG 匹配 → 高风险幻觉
        if needs_knowledge and not has_good_match and num_entities == 0:
            print(f"  [Confidence] 知识型问题无KG匹配 → I don't know")
            return "I don't know", True

        # 最佳匹配分数太差
        if best_score > SIMILARITY_THRESHOLD_LOOSE and needs_knowledge:
            print(f"  [Confidence] 最佳匹配分数 {best_score:.3f} > 阈值 → I don't know")
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
        批量生成回答 — 评估框架调用的主入口

        增强版流程：
        1. 查询分析 → 判断问题类型
        2. 图片检索 KG → 分数过滤
        3. 提取 KG 文本 → 查询感知的字段优先级
        4. 分层 Prompt → 适配查询类型
        5. VLM 生成
        6. 答案后处理 → 清理噪声
        7. 置信度评估 → 不确定时回复 "I don't know"
        """
        batch_start = time.time()
        print(f"\n[Task1 Agent] ===== 处理 {len(queries)} 条查询 =====")

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

            # ---- Step 3: 尝试规则快速提取（绕过 VLM 推理） ----
            candidate = self._extract_candidate_from_kg(query, kg_results)

            # ---- Step 4: 提取 KG 文本 ----
            kg_context, num_entities, avg_score = self._extract_kg_text(
                kg_results, query
            )

            # 评估检索质量
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
                "num_results": len(kg_results),
            }

            # ---- Step 5: 分层 Prompt + VLM 生成 ----
            messages = self._build_answer_prompt(
                query, kg_context, query_type_info, retrieval_quality
            )

            raw_answer = self._generate_single_answer(query, image, messages)
            cleaned_answer = self._post_process_answer(raw_answer)

            # ---- Step 6: 置信度评估 ----
            final_answer, was_overridden = self._assess_confidence(
                cleaned_answer, retrieval_quality, query_type_info
            )

            # ---- Step 7: 规则提取答案优先（如果 VLM 没给出好答案） ----
            if candidate and not was_overridden:
                vlm_is_idk = final_answer.strip().lower() == "i don't know"
                vlm_is_junk = len(final_answer) < 3 or final_answer.count(",") > 4
                if vlm_is_idk or vlm_is_junk:
                    final_answer = candidate
                    was_overridden = False  # 规则答案不算覆盖

            responses.append(final_answer)

            if final_answer.strip().lower() == "i don't know":
                idk_count += 1

            # 日志
            turn_time = time.time() - turn_start
            match_icon = "✅" if retrieval_quality["has_good_match"] else "⚠️"
            qtype = query_type_info["query_type"][:4]
            override_tag = " [OVERRIDE→IDK]" if was_overridden else ""
            print(
                f"  [{i+1}/{len(queries)}] {match_icon} "
                f"type={qtype} | KG={len(kg_results)}条/{num_entities}实体 "
                f"| score={best_score:.3f} | "
                f"Q: {query[:45]}... | A: {final_answer[:50]}..."
                f"{override_tag} | ⏱ {turn_time:.1f}s"
            )

        total_time = time.time() - batch_start
        idk_rate = idk_count / len(queries) * 100 if queries else 0
        print(
            f"[Task1 Agent] ===== 完成！总耗时 {total_time:.1f}s, "
            f"平均 {total_time/len(queries):.1f}s/条, "
            f"IDK率: {idk_rate:.1f}% ({idk_count}/{len(queries)}) =====\n"
        )

        return responses
