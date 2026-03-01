"""
Inference Modülü
================

Modern LLM için text üretme ve sohbet arayüzü.

İçerik:
- TextGenerator: Metin üretme (temperature, top-k, top-p, repetition penalty)
- ChatInterface: İnteraktif sohbet (CoT destekli)
"""

from modern_llm.inference.generator import TextGenerator, ChatInterface

__all__ = ["TextGenerator", "ChatInterface"]
