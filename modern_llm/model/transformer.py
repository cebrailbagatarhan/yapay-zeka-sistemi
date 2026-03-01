"""
Transformer Model
=================

Modern LLM'nin ana model sınıfı.

Decoder-only Transformer mimarisi:
1. Token Embedding
2. N x TransformerBlock (RMSNorm + GQA + SwiGLU)
3. Final RMSNorm
4. Language Model Head

Tüm modern LLM'ler (GPT-4, Claude, Gemini, LLaMA 3, Qwen 2.5)
bu temel mimariyi kullanır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from modern_llm.config import ModelConfig
from modern_llm.model.layers import RMSNorm, TransformerBlock


@dataclass
class ModelOutput:
    """Model çıktısı."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[List[torch.Tensor]] = None


class ModernLLMModel(nn.Module):
    """
    Modern LLM temel modeli (embedding + transformer blocks + final norm).
    
    Language model head olmadan çıplak transformer.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
    
    def _make_causal_mask(
        self,
        input_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Causal (autoregressive) attention mask oluştur.
        
        Gelecek tokenlara attend edilmesini engelleyen alt-üçgen mask.
        Mask değerleri: 0 (attend) veya -inf (mask)
        """
        bsz, tgt_len = input_shape
        
        # Alt-üçgen mask: [tgt_len, tgt_len]
        mask = torch.full(
            (tgt_len, tgt_len),
            torch.finfo(dtype).min,
            device=device,
        )
        mask_cond = torch.arange(tgt_len, device=device)
        mask.masked_fill_(mask_cond >= (mask_cond + 1).view(-1, 1), 0)
        
        # Past key values için genişlet
        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                mask,
            ], dim=-1)
        
        # [batch, 1, tgt_len, tgt_len + past_len]
        return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1=attend, 0=mask)
            position_ids: [batch_size, seq_len]
            past_key_values: KV cache listesi
            use_cache: KV cache kullan
            output_attentions: Attention weights döndür
            output_hidden_states: Ara hidden states döndür
        
        Returns:
            (last_hidden_state, past_key_values, all_hidden_states, all_attentions)
        """
        batch_size, seq_length = input_ids.shape
        
        # Past key values uzunluğu
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
        
        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Causal Mask
        causal_mask = self._make_causal_mask(
            (batch_size, seq_length),
            hidden_states.dtype,
            hidden_states.device,
            past_key_values_length,
        )
        
        # Padding mask ile birleştir
        if attention_mask is not None:
            # [batch, 1, 1, seq_len] -> expand
            expanded_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(hidden_states.dtype).min
            
            # Padding mask + causal mask
            if expanded_mask.shape[-1] < causal_mask.shape[-1]:
                # Past KV varsa padding mask'i genişlet
                expanded_mask = F.pad(
                    expanded_mask,
                    (causal_mask.shape[-1] - expanded_mask.shape[-1], 0),
                    value=0,
                )
            causal_mask = causal_mask + expanded_mask
        
        # Transformer Blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_past_key_values = () if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights, present_key_value = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,  # past_key_value (checkpointing ile uyumsuz)
                    use_cache,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_weights, present_key_value = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            
            if use_cache:
                next_past_key_values += (present_key_value,)
            
            if output_attentions:
                all_attentions += (attn_weights,)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return (
            hidden_states,
            next_past_key_values if use_cache else None,
            all_hidden_states,
            all_attentions,
        )


class ModernLLMForCausalLM(nn.Module):
    """
    Modern LLM for Causal Language Modeling.
    
    Temel model + Language Model Head.
    Next-token prediction için kullanılır.
    
    Eğitim:
    - Pre-training: Next-token prediction (cross-entropy loss)
    - SFT: Instruction-following eğitimi
    - CoT: Chain-of-thought reasoning eğitimi
    - DPO: Direct Preference Optimization
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Temel model
        self.model = ModernLLMModel(config)
        
        # LM Head
        if config.tie_word_embeddings:
            # Embedding weight'lerini paylaş (bellek tasarrufu)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Weight initialization (GPT-2 tarzı)."""
        std = self.config.initializer_range
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head
    
    def set_output_embeddings(self, value: nn.Linear):
        self.lm_head = value
    
    def enable_gradient_checkpointing(self):
        """Gradient checkpointing'i aktifleştir (bellek tasarrufu)."""
        self.model.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Gradient checkpointing'i deaktifleştir."""
        self.model.gradient_checkpointing = False
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Parametre sayısını döndür."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """
        Forward pass with optional language modeling loss.
        
        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            attention_mask: [batch_size, seq_len] - 1=attend, 0=mask
            position_ids: [batch_size, seq_len]
            past_key_values: KV cache
            labels: [batch_size, seq_len] - Target token IDs for loss
            use_cache: KV cache kullan
            output_attentions: Attention weights döndür
            output_hidden_states: Ara hidden states döndür
        
        Returns:
            ModelOutput with logits, loss, past_key_values, etc.
        """
        # Transformer forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[0]
        
        # LM Head: logits hesapla
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Stabilite için float32
        
        # Loss hesapla (opsiyonel)
        loss = None
        if labels is not None:
            # Shift: logits[:-1] ile labels[1:] karşılaştır
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Padding tokenları ignore et
                reduction='mean',
            )
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            past_key_values=outputs[1],
            hidden_states=outputs[2],
            attentions=outputs[3],
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.LongTensor:
        """
        Autoregressive text generation.
        
        Sampling stratejileri:
        - Temperature: Olasılık dağılımını keskinleştir/yumuşat
        - Top-k: En olası k token arasından seç
        - Top-p (nucleus): Kümülatif olasılığı p'yi geçene kadar token al
        - Repetition penalty: Tekrar eden tokenlara ceza uygula
        
        Args:
            input_ids: [batch_size, seq_len] - Başlangıç token'ları
            max_new_tokens: Üretilecek max yeni token sayısı
            temperature: Sampling temperature (0=greedy, >1=daha rastgele)
            top_p: Nucleus sampling threshold
            top_k: Top-k filtering
            repetition_penalty: Tekrar cezası (>1 tekrarı azaltır)
            do_sample: True=sampling, False=greedy
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            stop_token_ids: Durma token ID listesi
        
        Returns:
            [batch_size, seq_len + new_tokens] - Üretilen token'lar
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if stop_token_ids is None:
            stop_token_ids = [eos_token_id]
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # KV cache
        past_key_values = None
        generated_ids = input_ids.clone()
        
        # Üretilen tüm token ID'lerini takip et (repetition penalty için)
        all_generated = input_ids.clone()
        
        # Tamamlanma durumu (batch'teki her örnek için)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_new_tokens):
            # İlk adımda tüm input, sonraki adımlarda sadece son token
            if past_key_values is None:
                model_input = generated_ids
            else:
                model_input = generated_ids[:, -1:]
            
            # Forward pass
            outputs = self.forward(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            past_key_values = outputs.past_key_values
            
            # Son tokenın logit'leri
            next_token_logits = outputs.logits[:, -1, :]
            
            # === Repetition Penalty ===
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(all_generated[i].tolist()):
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty
            
            if do_sample:
                # === Temperature ===
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # === Top-K Filtering ===
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    min_top_k = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits,
                    )
                
                # === Top-P (Nucleus) Filtering ===
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Kümülatif olasılığı top_p'yi aşan tokenleri mask'le
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')
                
                # Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Tamamlanan sequence'ları padding ile doldur
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)
            
            # Token'ları ekle
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            all_generated = torch.cat([all_generated, next_tokens.unsqueeze(-1)], dim=-1)
            
            # EOS kontrolü
            for stop_id in stop_token_ids:
                finished = finished | (next_tokens == stop_id)
            
            if finished.all():
                break
        
        return generated_ids
    
    def save_pretrained(self, save_path: str):
        """Model ve konfigürasyonu kaydet."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Model weights
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
        
        # Config
        self.config.save(os.path.join(save_path, "config.json"))
        
        # Model info
        info = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "num_parameters": self.num_parameters(trainable_only=False),
            "num_trainable_parameters": self.num_parameters(trainable_only=True),
            "architecture": {
                "type": "decoder-only-transformer",
                "features": [
                    "RMSNorm",
                    "RoPE",
                    "GQA",
                    "SwiGLU",
                    "Flash Attention",
                    "KV-Cache",
                    "CoT",
                ],
            },
        }
        
        import json
        with open(os.path.join(save_path, "model_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model kaydedildi: {save_path}")
        print(f"   Parametreler: {self.num_parameters(False):,}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu") -> 'ModernLLMForCausalLM':
        """Kayıtlı model ve konfigürasyonu yükle."""
        import os
        
        # Config yükle
        config = ModelConfig.load(os.path.join(load_path, "config.json"))
        
        # Model oluştur
        model = cls(config)
        
        # Weights yükle
        state_dict = torch.load(
            os.path.join(load_path, "model.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        
        print(f"✅ Model yüklendi: {load_path}")
        print(f"   Parametreler: {model.num_parameters(False):,}")
        
        return model.to(device)
    
    def __repr__(self) -> str:
        params = self.num_parameters(False)
        if params >= 1e9:
            param_str = f"{params/1e9:.1f}B"
        elif params >= 1e6:
            param_str = f"{params/1e6:.1f}M"
        else:
            param_str = f"{params/1e3:.1f}K"
        
        return (
            f"ModernLLMForCausalLM(\n"
            f"  model_name={self.config.model_name},\n"
            f"  parameters={param_str},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  num_heads={self.config.num_attention_heads},\n"
            f"  num_kv_heads={self.config.num_kv_heads},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  max_seq_len={self.config.max_position_embeddings},\n"
            f"  features=[RoPE, GQA, SwiGLU, RMSNorm, CoT]\n"
            f")"
        )
