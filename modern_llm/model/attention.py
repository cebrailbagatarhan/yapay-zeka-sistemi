"""
Attention Modülü
================

Grouped Query Attention (GQA) + Rotary Position Embeddings (RoPE)
implementasyonu.

Modern LLM'lerdeki (LLaMA 3, Qwen 2.5, Gemma 2, Mistral) attention 
mekanizmalarının sıfırdan yazılmış versiyonu.

Özellikler:
- Rotary Position Embeddings (RoPE) - Mutlak pozisyon bilgisi yerine
  göreceli pozisyon bilgisi kodlar
- Grouped Query Attention (GQA) - KV head sayısını azaltarak bellek
  ve hız optimizasyonu sağlar
- Flash Attention 2 desteği (PyTorch 2.0+ SDPA)
- KV-Cache ile verimli autoregressive inference
- Sliding window attention desteği (opsiyonel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    
    RoPE, pozisyon bilgisini doğrudan attention key ve query vektörlerine
    rotation matrisi olarak uygular. Bu yaklaşım:
    - Göreceli pozisyon bilgisini kodlar
    - Uzun context'lere genelleşebilir
    - Çarpımsal pozisyon kodlaması sağlar
    
    Modern LLM'ler (LLaMA, Qwen, Mistral) yüksek theta (500K-10M) değerleri
    kullanarak uzun context window'ları destekler.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        theta: float = 500000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        
        # Frekans hesaplama: freq_i = 1 / (theta^(2i/d))
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cosine ve sine cache
        self._build_cache(max_position_embeddings, device)
    
    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Cosine ve sine değerlerini önceden hesapla."""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len, dtype=torch.float32, device=device or self.inv_freq.device)
        
        # Scaling uygula
        if self.scaling_type == "linear":
            t = t / self.scaling_factor
        elif self.scaling_type == "dynamic":
            # NTK-aware scaling
            base = self.theta * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) 
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device or self.inv_freq.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # [seq_len, dim] - cos ve sin
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """
        Args:
            x: Input tensor (sadece dtype ve device belirlemek için)
            seq_len: Sequence uzunluğu
        
        Returns:
            cos, sin: [seq_len, dim] boyutunda tensörler
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, x.device)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Tensörün ikinci yarısını negate edip birinci yarıyla yer değiştir."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Query ve key tensörlerine rotary position embedding uygula.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        position_ids: Pozisyon ID'leri (opsiyonel)
    
    Returns:
        Rotated (q, k) tuple
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
    from Multi-Head Checkpoints"
    https://arxiv.org/abs/2305.13245
    
    MHA (Multi-Head Attention) yerine GQA kullanarak:
    - KV head sayısını azaltır (KV cache bellek tasarrufu)
    - Inference hızını artırır
    - Model kalitesinden minimum kayıp
    
    GQA Tipleri:
    - num_kv_heads = num_heads: Standart MHA
    - num_kv_heads = 1: Multi-Query Attention (MQA)
    - 1 < num_kv_heads < num_heads: Grouped Query Attention
    
    Modern LLM'lerde tipik oran: num_heads/num_kv_heads = 4:1 veya 8:1
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout
        self.use_flash_attention = config.use_flash_attention
        self.sliding_window = config.sliding_window
        
        # Q, K, V, O projeksiyon katmanları (bias yok - modern standart)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary Embedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            theta=config.rope_theta,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(self.attention_dropout)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        GQA için KV head'leri tekrarla.
        
        [batch, num_kv_heads, seq_len, head_dim] 
        -> [batch, num_heads, seq_len, head_dim]
        """
        if n_rep == 1:
            return hidden_states
        
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
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
            attention_mask: [batch_size, 1, seq_len, kv_seq_len]
            position_ids: [batch_size, seq_len]
            past_key_value: KV cache tuple
            use_cache: KV cache kullan
            output_attentions: Attention weight döndür
        
        Returns:
            (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.shape
        
        # === Q, K, V Projeksiyonları ===
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape: [batch, seq, heads*dim] -> [batch, heads, seq, dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # === RoPE Uygula ===
        cos, sin = self.rotary_emb(query_states, seq_len=q_len + (past_key_value[0].shape[2] if past_key_value is not None else 0))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # === KV Cache ===
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        new_past_key_value = (key_states, value_states) if use_cache else None
        
        # === GQA: KV head'leri tekrarla ===
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        kv_seq_len = key_states.shape[2]
        
        # === Attention Hesaplama ===
        # Flash Attention (PyTorch 2.0+ SDPA)
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention') and not output_attentions:
            # SDPA causal mask otomatik uygular
            if attention_mask is not None and attention_mask.shape[-1] != kv_seq_len:
                attention_mask = None
            
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=(attention_mask is None and q_len > 1),
            )
            attn_weights = None
        else:
            # Manuel attention hesaplama
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            
            # Sliding window mask
            if self.sliding_window is not None and q_len > 1:
                # Sadece son sliding_window pozisyona attend et
                window_mask = torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=hidden_states.device)
                for i in range(q_len):
                    start = max(0, kv_seq_len - q_len + i - self.sliding_window + 1)
                    end = kv_seq_len - q_len + i + 1
                    window_mask[i, start:end] = False
                window_mask = window_mask.unsqueeze(0).unsqueeze(0) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + window_mask
            
            # Causal mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Softmax (float32'de hesapla, sonra dtype'a dönüştür)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, value_states)
        
        # === Output Projeksiyon ===
        # [batch, heads, seq, dim] -> [batch, seq, heads*dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, new_past_key_value
