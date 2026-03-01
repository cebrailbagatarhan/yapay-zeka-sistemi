"""
Model Konfigürasyonları
=======================

Modern LLM için tüm hyperparameter ve mimari ayarları.
Farklı GPU'lar için optimize edilmiş preset konfigürasyonlar içerir.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import os


@dataclass
class ModelConfig:
    """
    Modern LLM model konfigürasyonu.
    
    Mimari: Transformer Decoder-Only
    Özellikler: RoPE, GQA, SwiGLU, RMSNorm, CoT
    """
    
    # ===== Model Kimliği =====
    model_name: str = "ModernLLM"
    model_version: str = "1.0"
    
    # ===== Temel Mimari =====
    vocab_size: int = 32256  # 32K + özel tokenlar (256 reserved)
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 4  # GQA: 12/4 = 3:1 ratio
    intermediate_size: int = 2048  # SwiGLU FFN boyutu
    max_position_embeddings: int = 2048
    
    # ===== RoPE (Rotary Position Embedding) =====
    rope_theta: float = 500000.0  # Modern modeller yüksek theta kullanır
    rope_scaling_type: Optional[str] = None  # "linear" veya "dynamic"
    rope_scaling_factor: float = 1.0
    
    # ===== Normalization =====
    rms_norm_eps: float = 1e-6
    
    # ===== Activation =====
    hidden_act: str = "silu"  # SwiGLU için SiLU (Swish)
    
    # ===== Dropout =====
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    embedding_dropout: float = 0.0
    
    # ===== Özel Tokenlar =====
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Chat tokenları
    im_start_token_id: int = 4  # <|im_start|>
    im_end_token_id: int = 5    # <|im_end|>
    
    # CoT tokenları
    think_start_token_id: int = 6  # <think>
    think_end_token_id: int = 7    # </think>
    
    # Sistem tokenları
    system_token_id: int = 8
    user_token_id: int = 9
    assistant_token_id: int = 10
    
    # ===== Embedding =====
    tie_word_embeddings: bool = True
    
    # ===== Attention =====
    use_flash_attention: bool = True
    sliding_window: Optional[int] = None  # None = tam attention
    
    # ===== Initialization =====
    initializer_range: float = 0.02
    
    # ===== Cache =====
    use_cache: bool = True
    
    # ===== CoT (Chain-of-Thought) =====
    cot_enabled: bool = True
    max_thinking_tokens: int = 1024  # Düşünme için max token
    
    def __post_init__(self):
        """Konfigürasyon doğrulama."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) num_attention_heads ({self.num_attention_heads}) ile bölünebilir olmalı"
        assert self.num_attention_heads % self.num_kv_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) num_kv_heads ({self.num_kv_heads}) ile bölünebilir olmalı"
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads
    
    @property
    def estimated_params(self) -> int:
        """Tahmini parametre sayısı."""
        # Embedding
        emb = self.vocab_size * self.hidden_size
        
        # Her katman için attention
        q = self.hidden_size * self.hidden_size  # Q projection
        k = self.hidden_size * (self.num_kv_heads * self.head_dim)  # K projection (GQA)
        v = self.hidden_size * (self.num_kv_heads * self.head_dim)  # V projection (GQA)
        o = self.hidden_size * self.hidden_size  # Output projection
        attn = q + k + v + o
        
        # Her katman için SwiGLU FFN
        ffn = 3 * self.hidden_size * self.intermediate_size  # gate + up + down
        
        # Her katman için RMSNorm
        norm = 2 * self.hidden_size
        
        # Toplam per layer
        per_layer = attn + ffn + norm
        
        # Final norm
        final_norm = self.hidden_size
        
        # LM head (tied ise 0)
        lm_head = 0 if self.tie_word_embeddings else self.hidden_size * self.vocab_size
        
        total = emb + (per_layer * self.num_layers) + final_norm + lm_head
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Konfigürasyonu dictionary'ye dönüştür."""
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, path: str):
        """Konfigürasyonu JSON dosyasına kaydet."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        """Dictionary'den konfigürasyon oluştur."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """JSON dosyasından konfigürasyon yükle."""
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class TrainingConfig:
    """
    Eğitim konfigürasyonu.
    """
    
    # ===== Eğitim Temel =====
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    
    # ===== Optimizer =====
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # ===== Learning Rate Schedule =====
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # 0 ise warmup_ratio kullanılır
    min_lr_ratio: float = 0.1  # min_lr = lr * min_lr_ratio
    
    # ===== Precision =====
    mixed_precision: str = "bf16"  # bf16, fp16, fp32
    
    # ===== Logging =====
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    
    # ===== Checkpointing =====
    save_total_limit: int = 3
    output_dir: str = "checkpoints"
    
    # ===== Wandb =====
    use_wandb: bool = False
    wandb_project: str = "modern-llm"
    wandb_run_name: Optional[str] = None
    
    # ===== Data =====
    train_split: float = 0.98
    eval_split: float = 0.02
    shuffle: bool = True
    num_workers: int = 4
    
    # ===== Eğitim Aşamaları =====
    stage: str = "sft"  # pretrain, sft, dpo, cot
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ===== Preset Konfigürasyonlar =====

PRESET_CONFIGS = {
    "nano": ModelConfig(
        model_name="ModernLLM-Nano",
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        num_kv_heads=2,
        intermediate_size=1408,
        max_position_embeddings=2048,
        # ~30M params
    ),
    "small": ModelConfig(
        model_name="ModernLLM-Small",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=4,
        intermediate_size=2048,
        max_position_embeddings=2048,
        # ~100M params
    ),
    "medium": ModelConfig(
        model_name="ModernLLM-Medium",
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        intermediate_size=2816,
        max_position_embeddings=4096,
        # ~350M params
    ),
    "large": ModelConfig(
        model_name="ModernLLM-Large",
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=5504,
        max_position_embeddings=4096,
        # ~1.3B params
    ),
    "xl": ModelConfig(
        model_name="ModernLLM-XL",
        hidden_size=3072,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        max_position_embeddings=8192,
        # ~3B params
    ),
}

# GPU'ya göre önerilen konfigürasyonlar
GPU_PRESETS = {
    "T4": {
        "model": "nano",
        "training": TrainingConfig(
            batch_size=4,
            max_seq_length=512,
            mixed_precision="fp16",
            gradient_accumulation_steps=8,
        ),
    },
    "L4": {
        "model": "small",
        "training": TrainingConfig(
            batch_size=4,
            max_seq_length=1024,
            mixed_precision="bf16",
            gradient_accumulation_steps=4,
        ),
    },
    "A100": {
        "model": "medium",
        "training": TrainingConfig(
            batch_size=8,
            max_seq_length=2048,
            mixed_precision="bf16",
            gradient_accumulation_steps=2,
        ),
    },
    "H100": {
        "model": "large",
        "training": TrainingConfig(
            batch_size=16,
            max_seq_length=4096,
            mixed_precision="bf16",
            gradient_accumulation_steps=1,
        ),
    },
}


def get_config_for_gpu(gpu_name: str) -> tuple:
    """GPU adına göre uygun model ve eğitim konfigürasyonu döndür."""
    gpu_name = gpu_name.upper()
    
    for key in GPU_PRESETS:
        if key in gpu_name:
            preset = GPU_PRESETS[key]
            model_config = PRESET_CONFIGS[preset["model"]]
            train_config = preset["training"]
            return model_config, train_config
    
    # Bilinmeyen GPU - en küçük modeli kullan
    return PRESET_CONFIGS["nano"], TrainingConfig(
        batch_size=2,
        max_seq_length=512,
        mixed_precision="fp16",
    )
