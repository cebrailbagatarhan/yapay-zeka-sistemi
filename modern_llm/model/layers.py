"""
Katman Modülleri
================

Modern LLM'lerin temel yapı taşları:
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU FFN: Swish-Gated Linear Unit Feed-Forward Network
- TransformerBlock: Tek bir decoder katmanı

Bu bileşenler LLaMA, Qwen, Gemma, Mistral gibi modern modellerin
ortak mimarisini oluşturur.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from modern_llm.model.attention import GroupedQueryAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Zhang & Sennrich, "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
    
    LayerNorm'dan farklı olarak:
    - Mean'i çıkarmaz (re-centering yok)
    - Sadece RMS ile bölme yapar (re-scaling)
    - Bias terimi yok
    - Daha hızlı ve hafif
    
    Formula: x_norm = x / sqrt(mean(x^2) + eps) * weight
    
    Tüm modern LLM'ler (LLaMA, Qwen, Gemma, Mistral, Claude, GPT-4)
    Pre-RMSNorm kullanır (attention/FFN öncesi normalization).
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            Normalized tensor [batch, seq_len, hidden_size]
        """
        # Float32'de hesapla (stabilite için)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    Shazeer, "GLU Variants Improve Transformer"
    https://arxiv.org/abs/2002.05202
    
    Standart FFN: 
        FFN(x) = W2 * GELU(W1 * x)
    
    SwiGLU FFN:
        FFN(x) = W_down * (SiLU(W_gate * x) ⊙ W_up * x)
    
    SwiGLU neden kullanılır:
    - Gating mekanizması bilgi akışını kontrol eder
    - SiLU (Swish) smooth bir aktivasyon fonksiyonu
    - Daha iyi gradyan akışı sağlar
    - Tüm modern LLM'lerde standart (LLaMA, Qwen, Gemma, Mistral)
    
    Not: 3 projeksiyon matrisi olduğu için intermediate_size genellikle
    standart FFN'den küçüktür (2/3 oranında). 
    Örnek: hidden=4096, intermediate=11008 (standart olur ~16384)
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Üç projeksiyon matrisi (bias yok - modern standart)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))  # SiLU = x * sigmoid(x)
        up = self.up_proj(x)
        
        # Element-wise çarpım (gating)
        hidden = gate * up
        
        # Down projeksiyon
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Tek bir Transformer Decoder bloğu.
    
    Modern LLM katman yapısı (Pre-Norm):
    
    1. RMSNorm → GQA Attention → Residual Add
    2. RMSNorm → SwiGLU FFN → Residual Add
    
    Pre-Norm vs Post-Norm:
    - Post-Norm (orijinal Transformer): LayerNorm(x + Sublayer(x))
    - Pre-Norm (modern LLM'ler): x + Sublayer(Norm(x))
    
    Pre-Norm avantajları:
    - Daha stabil eğitim (warm-up'a daha az ihtiyaç)
    - Daha iyi gradyan akışı (residual path'de engel yok)
    - Derin modellerde daha iyi performans
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Pre-norm + Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        
        # Pre-norm + FFN
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: KV cache
            use_cache: KV cache kullan
            output_attentions: Attention weights döndür
        
        Returns:
            (hidden_states, attn_weights, past_key_value)
        """
        residual = hidden_states
        
        # === 1. Pre-Norm + Self-Attention ===
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # === 2. Pre-Norm + FFN ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, present_key_value
