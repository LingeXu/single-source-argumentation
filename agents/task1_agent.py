"""
Task 1: Single-Source Augmentation Agent (Enhanced)
=====================================================
专为 Mac (Apple Silicon) 设计的 Task 1 Agent。

工作流程（增强版）：
  1. 查询分析：判断问题类型（视觉识别 vs 知识型 vs 推理型）
  2. 图片 → search_pipeline (image search) → 知识图谱结构化数据
  3. 基于相似度分数过滤低质量 KG 结果
  4. 提取实体属性 (description, caption, summary 等)，按分数加权
  5. 构建分层 Prompt：图片 + KG 数据 + 问题 + 查询类型指导
  6. VLM 生成答案
  7. 后处理：置信度判断，不确定时说 "I don't know"

技术栈：
  - 检索：cragmm-search-pipeline (CLIP + BGE)
  - 推理：mlx-vlm + Qwen2-VL-2B-Instruct-4bit (Apple Silicon 原生)
"""

from typing import Dict, List, Any, Tuple, Optional
import re
import time

import numpy as np
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

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

# MLX VLM 模型 (4-bit 量化，适合 18GB 内存)
# 备选: "mlx-community/Llama-3.2-Vision-Instruct-4bit" (需要更多内存)
VLM_MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

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
    # Step 2: 改进的 KG 文本提取
    # ============================================================

    def _extract_kg_text(
        self,
        kg_results: List[Dict],
        query: str = "",
    ) -> Tuple[str, int, float]:
        """
        Step 2: 从 KG 检索结果中提取有用的文本信息

        改进：
        - 基于相似度分数的加权提取
        - 查询感知的字段优先级
        - 更好的去重和截断

        Returns:
            (context_text, num_valid_entities, avg_score)
        """
        if not kg_results:
            return "", 0, 1.0

        all_contexts = []
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

                # 查询感知的字段优先级
                # 如果问题中有特定关键词，调整优先级
                query_lower = query.lower()
                if any(kw in query_lower for kw in ["price", "cost", "how much"]):
                    priority_fields = [
                        "price", "cost", "description", "caption", "summary",
                        "msrp", "list_price", "retail_price"
                    ]
                elif any(kw in query_lower for kw in ["where", "location", "country"]):
                    priority_fields = [
                        "location", "country", "country_of_origin", "assembly_locations",
                        "description", "caption", "summary"
                    ]
                elif any(kw in query_lower for kw in ["when", "year", "date", "release"]):
                    priority_fields = [
                        "production_years", "production_start", "production_start_year",
                        "release_date", "year", "model_year", "description", "caption", "summary"
                    ]
                else:
                    # 默认优先级：包含 KG 中常见的字段名
                    priority_fields = [
                        "description", "caption", "summary",
                        "name", "manufacturer", "brand", "model",
                        "body_style", "body_type", "class", "type",
                        "engine_type", "engine", "transmission", "drive_type",
                        "country_of_origin", "location", "production_years",
                    ]

                entity_texts = []

                # 提取优先字段
                for field in priority_fields:
                    if field in attrs and attrs[field]:
                        text = str(attrs[field]).strip()
                        if text and len(text) > 3:
                            entity_texts.append(text)

                # 提取其他属性
                for key, value in attrs.items():
                    if key not in priority_fields and key not in ("image_url", "url") and value:
                        text = str(value).strip()
                        if text and len(text) > 1:
                            entity_texts.append(f"{key}: {text}")

                if entity_texts:
                    # 合并该实体的所有文本
                    combined = " | ".join(entity_texts)
                    # 如果有实体名，加上前缀
                    if entity_name:
                        combined = f"[{entity_name}] {combined}"
                    all_contexts.append(combined)
                    total_score += 1.0 - score  # 转换为相似度（越高越好）
                    num_valid += 1

        if not all_contexts:
            return "", 0, 1.0

        # 去重（保留首次出现的顺序）
        seen = set()
        unique_contexts = []
        for ctx in all_contexts:
            if ctx not in seen:
                seen.add(ctx)
                unique_contexts.append(ctx)

        # 限制上下文长度 — 小模型需要更简洁的上下文
        max_contexts = 5
        context = "\n".join(
            f"- {ctx}" for ctx in unique_contexts[:max_contexts]
        )

        avg_score = total_score / num_valid if num_valid > 0 else 0.0
        return context, num_valid, avg_score

    # ============================================================
    # Step 3: 分层 Prompt 构建
    # ============================================================

    def _build_answer_prompt(
        self,
        query: str,
        kg_context: str,
        query_type_info: Dict[str, Any],
        retrieval_quality: Dict[str, Any],
    ) -> str:
        """
        Step 3: 构建分层回答 Prompt

        改进：
        - 根据查询类型调整系统指令
        - 根据检索质量调整不确定性表达
        - 更精确的规则约束
        """
        query_type = query_type_info["query_type"]
        has_good_kg = retrieval_quality.get("has_good_match", False)

        # 基础系统指令 — 为 2B 小模型精简
        system_instruction = (
            "You answer questions concisely using image and knowledge graph data. "
            "Rules:\n"
            "1. Answer in 1 short sentence or a single phrase.\n"
            '2. Only use facts from the image or KG data. Never guess.\n'
            '3. If not confident, say EXACTLY: "I don\'t know"\n'
        )

        # 根据查询类型添加简短指导
        type_specific_guidance = {
            "visual": " Look at the image carefully.",
            "knowledge": " Check the KG data for the answer.",
            "reasoning": " Combine visual clues with KG data.",
            "comparison": " Compare based on available data.",
        }

        guidance = type_specific_guidance.get(query_type, "")
        system_instruction += guidance

        # 构建用户消息 — 精简格式
        if kg_context:
            user_message = (
                f"Knowledge Graph data:\n{kg_context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
        else:
            user_message = (
                f"Question: {query}\n"
                f"Answer:"
            )

        # 构建完整 messages 格式
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

            # ---- Step 3: 提取 KG 文本 ----
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

            # ---- Step 4: 分层 Prompt ----
            messages = self._build_answer_prompt(
                query, kg_context, query_type_info, retrieval_quality
            )

            # ---- Step 5: VLM 生成 ----
            raw_answer = self._generate_single_answer(query, image, messages)

            # ---- Step 6: 答案后处理 ----
            cleaned_answer = self._post_process_answer(raw_answer)

            # ---- Step 7: 置信度评估 ----
            final_answer, was_overridden = self._assess_confidence(
                cleaned_answer, retrieval_quality, query_type_info
            )

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
