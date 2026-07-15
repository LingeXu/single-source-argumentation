# ============================================================
# CRAG-MM Agent Configuration
# ============================================================
# 选择你要使用的 Agent，将其取消注释并设为 UserAgent。
# 未使用的 Agent 导入被 try/except 包裹，避免缺失依赖时报错。

# --- Random Agent (无依赖) ---
from task2.random_agent import RandomAgent

# --- Simple RAG Agent (需要 vllm + CUDA GPU) ---
try:
    from task2.rag_agent import SimpleRAGAgent
except ImportError:
    SimpleRAGAgent = None

# --- Llama Vision Model (需要 vllm + CUDA GPU) ---
try:
    from task2.vanilla_llama_vision_agent import LlamaVisionModel
except ImportError:
    LlamaVisionModel = None

# --- MLLM RAG Agent (需要 vllm + CUDA GPU, 比赛获奖方案) ---
try:
    from task2.mllm_rag_agent import MLLMRAGAgent
except ImportError:
    MLLMRAGAgent = None

# --- Task 1 Agent (MLX 版, Apple Silicon / Mac 专用) ---
try:
    from task1.task1_agent import Task1SingleSourceAgent
except ImportError:
    Task1SingleSourceAgent = None

# --- Task 2 Agent (MLX 版, 多源增强, Apple Silicon / Mac 专用) ---
try:
    from task2.task2_agent import Task2MultiSourceAgent
except ImportError:
    Task2MultiSourceAgent = None

# ============================================================
# 🎯 在此选择使用的 Agent
# ============================================================
# UserAgent = RandomAgent
# UserAgent = SimpleRAGAgent
# UserAgent = LlamaVisionModel
# UserAgent = MLLMRAGAgent
# UserAgent = Task1SingleSourceAgent
UserAgent = Task2MultiSourceAgent
