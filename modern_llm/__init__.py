"""
Modern LLM - Sıfırdan Modern Dil Modeli
========================================

Günümüz modern LLM mimarilerinin (GPT-4, Claude, Gemini, LLaMA 3, Qwen 2.5)
temel özelliklerini içeren, sıfırdan yazılmış bir dil modeli implementasyonu.

Mimari Özellikler:
- RMSNorm (Pre-normalization)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU Activation
- Flash Attention desteği
- KV-Cache ile verimli inference
- Chain-of-Thought (CoT) reasoning modülü
- <think>...</think> ile dahili akıl yürütme

Boyutlar:
- Nano  (~30M params)  - Hızlı test
- Small (~125M params) - T4 GPU
- Medium (~350M params) - A100 GPU
- Large (~1.3B params) - H100 GPU

Yazar: Cebrail Bağatar Han
"""

__version__ = "1.0.0"
__author__ = "Cebrail Bağatar Han"

from modern_llm.config import ModelConfig, TrainingConfig, PRESET_CONFIGS
