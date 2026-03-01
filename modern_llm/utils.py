"""
Modern LLM - Utility Functions
===============================

Yardımcı fonksiyonlar.
"""

import os
import json
import sys
import time
import math
import random
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn


# ─── GPU / Device ─────────────────────────────────────────────────────────

def get_device(prefer: str = "auto") -> torch.device:
    """
    En iyi device'ı otomatik seç.
    
    Args:
        prefer: "auto", "cuda", "cpu", "mps"
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(prefer)


def get_gpu_info() -> Dict[str, Any]:
    """GPU bilgilerini döndür."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    
    # GPU türünü belirle
    gpu_type = "unknown"
    if "T4" in gpu_name:
        gpu_type = "T4"
    elif "L4" in gpu_name:
        gpu_type = "L4"
    elif "A100" in gpu_name:
        gpu_type = "A100"
    elif "H100" in gpu_name:
        gpu_type = "H100"
    elif "V100" in gpu_name:
        gpu_type = "V100"
    elif "3090" in gpu_name or "4090" in gpu_name:
        gpu_type = "consumer"
    
    return {
        "available": True,
        "name": gpu_name,
        "type": gpu_type,
        "total_memory_gb": round(total_mem, 2),
        "bf16_support": torch.cuda.is_bf16_supported(),
        "flash_attention": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    }


def detect_optimal_config():
    """GPU'ya göre optimal model konfigürasyonu seç."""
    from modern_llm.config import get_config_for_gpu
    
    info = get_gpu_info()
    if not info["available"]:
        return get_config_for_gpu("T4")  # Minimum config
    
    return get_config_for_gpu(info["type"])


# ─── Model Utilities ────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Model parametrelerini say."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Katman bazında
    layer_params = {}
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            layer_params[name] = params
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "layers": layer_params,
    }


def format_params(num: int) -> str:
    """Parametre sayısını okunabilir formata çevir."""
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    return str(num)


def estimate_memory(
    num_params: int,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    batch_size: int = 4,
    seq_len: int = 2048,
) -> Dict[str, float]:
    """
    Model bellek kullanımını tahmin et (GB).
    
    Args:
        num_params: Toplam parametre sayısı
        dtype: "fp32", "fp16", "bf16"
        optimizer: "adamw", "sgd", "adam"
        batch_size: Batch boyutu
        seq_len: Dizi uzunluğu
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]
    
    # Model ağırlıkları
    model_mem = num_params * bytes_per_param / (1024**3)
    
    # Gradient
    grad_mem = num_params * 4 / (1024**3)  # Gradientler fp32
    
    # Optimizer states
    if optimizer == "adamw":
        opt_mem = num_params * 8 / (1024**3)  # 2 state (m, v) * fp32
    elif optimizer == "sgd":
        opt_mem = num_params * 4 / (1024**3)
    else:
        opt_mem = num_params * 8 / (1024**3)
    
    # Activations (kaba tahmin)
    # Activation belleği seq_len ve batch_size'a bağlı
    act_mem = batch_size * seq_len * num_params * 2 / (num_params * 12) / (1024**3)
    act_mem = max(act_mem, 0.5)  # Minimum 0.5 GB
    
    total = model_mem + grad_mem + opt_mem + act_mem
    
    return {
        "model_gb": round(model_mem, 2),
        "gradient_gb": round(grad_mem, 2),
        "optimizer_gb": round(opt_mem, 2),
        "activation_gb": round(act_mem, 2),
        "total_gb": round(total, 2),
    }


# ─── Data Utilities ─────────────────────────────────────────────────────

def set_seed(seed: int):
    """Tüm random seed'leri ayarla (reproducibility)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def load_json(path: str) -> Any:
    """JSON dosyası yükle."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """JSON dosyası kaydet."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ─── Training Utilities ─────────────────────────────────────────────────

def cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))


class TrainingTimer:
    """Eğitim süresini takip eden zamanlayıcı."""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
    
    def step(self):
        """Bir adım zamanını kaydet."""
        self.step_times.append(time.time())
    
    @property
    def elapsed(self) -> float:
        """Geçen süre (saniye)."""
        return time.time() - self.start_time
    
    @property
    def elapsed_str(self) -> str:
        """Geçen süre (okunabilir format)."""
        seconds = self.elapsed
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def eta(self, current_step: int, total_steps: int) -> str:
        """Tahmini kalan süre."""
        if current_step == 0:
            return "?"
        
        avg_time = self.elapsed / current_step
        remaining = avg_time * (total_steps - current_step)
        
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def tokens_per_second(self, total_tokens: int) -> float:
        """Token/saniye hesapla."""
        return total_tokens / max(self.elapsed, 1e-6)


# ─── Text Processing ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Metni temizle."""
    import re
    
    # Çoklu boşlukları tek boşluğa çevir
    text = re.sub(r'\s+', ' ', text)
    
    # Başta ve sondaki boşlukları temizle
    text = text.strip()
    
    return text


def chunk_text(text: str, max_length: int, overlap: int = 0) -> List[str]:
    """
    Metni sabit uzunlukta parçalara böl.
    
    Args:
        text: Bölünecek metin
        max_length: Maksimum parça uzunluğu (karakter)
        overlap: Parçalar arası örtüşme
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


# ─── Printing / Logging ─────────────────────────────────────────────────

def print_model_summary(model: nn.Module, config=None):
    """Model özetini yazdır."""
    params = count_parameters(model)
    
    print("=" * 60)
    print("📊 Model Özeti")
    print("=" * 60)
    
    if config:
        print(f"  Model: {getattr(config, 'model_name', 'Modern LLM')}")
        print(f"  Hidden: {config.hidden_size}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Heads: {config.num_attention_heads}")
        print(f"  KV Heads: {config.num_kv_heads}")
        print(f"  Vocab: {config.vocab_size}")
        print(f"  Max Seq: {config.max_position_embeddings}")
    
    print(f"\n  Toplam Params: {format_params(params['total'])} ({params['total']:,})")
    print(f"  Eğitilebilir: {format_params(params['trainable'])} ({params['trainable']:,})")
    
    # Bellek tahmini
    mem = estimate_memory(params['total'])
    print(f"\n  Tahmini Bellek (BF16 eğitim):")
    print(f"    Model: {mem['model_gb']:.2f} GB")
    print(f"    Toplam: {mem['total_gb']:.2f} GB")
    
    print("=" * 60)


def print_training_progress(
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    timer: Optional[TrainingTimer] = None,
    extra: Optional[Dict] = None,
):
    """Training progress bar."""
    pct = step / max(total_steps, 1) * 100
    bar_len = 30
    filled = int(bar_len * step / max(total_steps, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    
    line = f"\r  [{bar}] {pct:5.1f}% | Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e}"
    
    if timer:
        line += f" | {timer.elapsed_str}"
        eta = timer.eta(step, total_steps)
        line += f" | ETA: {eta}"
    
    if extra:
        for k, v in extra.items():
            if isinstance(v, float):
                line += f" | {k}: {v:.4f}"
            else:
                line += f" | {k}: {v}"
    
    print(line, end="", flush=True)
