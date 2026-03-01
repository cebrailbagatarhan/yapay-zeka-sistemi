#!/usr/bin/env python3
"""
H100 Optimized Training Script
================================

NVIDIA H100 GPU için optimize edilmiş 15 dakikalık eğitim pipeline'ı.
Tüm frontier-quality dataset'leri otomatik yükler.

Kullanım (Colab H100):
    !git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
    %cd yapay-zeka-sistemi
    !pip install torch sentencepiece
    !python train_h100.py

    # Özelleştirme:
    !python train_h100.py --time_budget 900 --model large
    !python train_h100.py --model medium --time_budget 600

Özellikler:
    - torch.compile ile kernel fusion (%15-40 hız artışı)
    - BF16 native precision (H100 Hopper)
    - Zaman bütçeli eğitim (varsayılan 15 dk)
    - Tüm dataset'leri otomatik yükleme
    - Optimize edilmiş DataLoader (persistent workers, prefetch)
    - Azaltılmış checkpoint/log overhead
    - Otomatik GitHub push ve model kaydetme

Yazar: Cebrail Bağatar Han
"""

import os
import sys
import json
import time
import math
import glob
import argparse
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.cuda.amp import autocast, GradScaler

# Modern LLM imports
from modern_llm.config import (
    ModelConfig,
    TrainingConfig,
    PRESET_CONFIGS,
    GPU_PRESETS,
    get_config_for_gpu,
)
from modern_llm.model.transformer import ModernLLMForCausalLM
from modern_llm.tokenizer import ModernTokenizer
from modern_llm.training.dataset import (
    TextDataset,
    ChatDataset,
    CoTDataset,
    load_dataset_from_json,
    create_data_collator,
    convert_to_chat_format,
    convert_cot_format,
)
from modern_llm.inference.generator import TextGenerator, GenerationConfig
from modern_llm.utils import (
    get_device,
    set_seed,
    count_parameters,
    format_params,
    print_model_summary,
)


# ============================================================
# H100 Optimized Constants
# ============================================================

# Zaman bütçesi oranları (toplam bütçenin yüzdesi)
TIME_BUDGET_RATIOS = {
    "tokenizer": 0.05,    # %5  → ~45 saniye
    "pretrain": 0.40,     # %40 → ~6 dakika
    "sft": 0.30,          # %30 → ~4.5 dakika
    "cot": 0.20,          # %20 → ~3 dakika
    "eval_save": 0.05,    # %5  → ~45 saniye
}

# H100 optimized DataLoader settings
H100_DATALOADER_KWARGS = {
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "drop_last": True,
}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          ⚡ H100 Optimized LLM Training Pipeline ⚡             ║
║                                                                  ║
║  15-Minute Budget • torch.compile • BF16 Native                 ║
║  RoPE • GQA • SwiGLU • RMSNorm • CoT Reasoning                 ║
║  Frontier-Quality Datasets • Auto GitHub Push                   ║
║                                                                  ║
║  Yazar: Cebrail Bağatar Han                                     ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ============================================================
# Dataset Loading: Tüm frontier-quality verileri yükle
# ============================================================

def load_all_datasets(data_root: str = ".") -> Dict[str, List]:
    """
    Tüm dataset dosyalarını otomatik yükle.
    
    Returns:
        dict: {"pretrain": [...], "sft": [...], "cot": [...]}
    """
    pretrain_texts = []
    sft_data = []
    cot_data = []
    
    print("\n📂 Dataset'ler yükleniyor...")
    
    # ===== 1. data/datasets/ klasöründen yükle =====
    datasets_dir = os.path.join(data_root, "data", "datasets")
    if os.path.exists(datasets_dir):
        for fname in sorted(os.listdir(datasets_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(datasets_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list) or len(data) == 0:
                    continue
                
                sample = data[0]
                
                # Format tespit et
                if "messages" in sample:
                    # Chat/SFT format
                    sft_data.extend(data)
                    print(f"  ✅ [SFT] {fname}: {len(data)} örnek")
                elif "instruction" in sample and "thinking" in sample:
                    # CoT format
                    cot_data.extend(data)
                    print(f"  ✅ [CoT] {fname}: {len(data)} örnek")
                elif "instruction" in sample and "response" in sample:
                    # Instruction format → CoT olarak kullan
                    cot_data.extend(data)
                    print(f"  ✅ [CoT] {fname}: {len(data)} örnek")
                elif "text" in sample:
                    # Plain text → pretrain
                    pretrain_texts.extend([item["text"] for item in data if item.get("text")])
                    print(f"  ✅ [PRE] {fname}: {len(data)} metin")
                else:
                    print(f"  ⚠️  {fname}: Bilinmeyen format, atlanıyor")
                    
            except Exception as e:
                print(f"  ❌ {fname}: Yükleme hatası - {e}")
    
    # ===== 2. data/cot/ klasöründen yükle =====
    cot_dir = os.path.join(data_root, "data", "cot")
    if os.path.exists(cot_dir):
        for fname in sorted(os.listdir(cot_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(cot_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "instruction" in item and ("thinking" in item or "response" in item):
                            cot_data.append(item)
                    print(f"  ✅ [CoT] cot/{fname}: {len(data)} örnek")
            except Exception as e:
                print(f"  ❌ cot/{fname}: {e}")
    
    # ===== 3. data/training/ klasöründen yükle =====
    training_dir = os.path.join(data_root, "data", "training")
    if os.path.exists(training_dir):
        for fname in sorted(os.listdir(training_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(training_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "messages" in item:
                            sft_data.append(item)
                        elif "instruction" in item:
                            if "thinking" in item:
                                cot_data.append(item)
                            elif "output" in item:
                                converted = convert_to_chat_format([item])
                                sft_data.extend(converted)
                    print(f"  ✅ [MIX] training/{fname}: {len(data)} örnek")
            except Exception as e:
                print(f"  ❌ training/{fname}: {e}")
    
    # ===== 4. data/examples/ klasöründen yükle =====
    examples_dir = os.path.join(data_root, "data", "examples")
    if os.path.exists(examples_dir):
        for fname in sorted(os.listdir(examples_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(examples_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "instruction" in item and "thinking" in item:
                            cot_data.append(item)
                        elif "messages" in item:
                            sft_data.append(item)
                    print(f"  ✅ [MIX] examples/{fname}: {len(data)} örnek")
            except Exception as e:
                pass
    
    # ===== 5. Yerleşik pretrain verileri (her zaman ekle) =====
    builtin_pretrain = _get_builtin_pretrain_texts()
    pretrain_texts.extend(builtin_pretrain)
    
    # ===== 6. Yerleşik SFT verileri =====
    builtin_sft = _get_builtin_sft_data()
    sft_data.extend(builtin_sft)
    
    # Dedup (basit)
    cot_data = _dedup_by_instruction(cot_data)
    
    print(f"\n📊 Toplam Veri:")
    print(f"  📚 Pre-training metinleri: {len(pretrain_texts)}")
    print(f"  💬 SFT (Chat) örnekleri: {len(sft_data)}")
    print(f"  🧠 CoT örnekleri: {len(cot_data)}")
    
    # CoT verilerinden pretrain için ek metin çıkar (bonus)
    for item in cot_data:
        combined = f"{item.get('instruction', '')} {item.get('thinking', '')} {item.get('response', '')}"
        if len(combined) > 50:
            pretrain_texts.append(combined)
    
    # SFT verilerinden pretrain için ek metin çıkar (bonus)
    for item in sft_data:
        if "messages" in item:
            combined = " ".join(msg.get("content", "") for msg in item["messages"])
            if len(combined) > 50:
                pretrain_texts.append(combined)
    
    print(f"  📚 Pre-training (augmented): {len(pretrain_texts)}")
    
    return {
        "pretrain": pretrain_texts,
        "sft": sft_data,
        "cot": cot_data,
    }


def _dedup_by_instruction(data: List[Dict]) -> List[Dict]:
    """Tekrar eden instruction'ları temizle."""
    seen = set()
    unique = []
    for item in data:
        key = item.get("instruction", "")[:100]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _get_builtin_pretrain_texts() -> List[str]:
    """Yerleşik pretrain metinleri."""
    return [
        "Yapay zeka, makinelerin insan benzeri zekâ sergilemesini sağlayan bilgisayar bilimi dalıdır. "
        "Makine öğrenmesi, derin öğrenme ve doğal dil işleme gibi alt dalları vardır.",
        
        "Türkiye, Avrupa ve Asya kıtalarında toprakları bulunan bir ülkedir. "
        "Başkenti Ankara, en büyük şehri İstanbul'dur. Resmi dili Türkçedir.",
        
        "Python programlama dili, 1991 yılında Guido van Rossum tarafından geliştirilmiştir. "
        "Okunabilir sözdizimi ve geniş kütüphane desteği ile popüler bir dildir.",
        
        "Transformer mimarisi, 2017 yılında Google tarafından 'Attention Is All You Need' makalesi ile tanıtıldı. "
        "Self-attention mekanizması sayesinde paralel hesaplama yapabilir ve uzun menzilli bağımlılıkları öğrenebilir.",
        
        "Derin öğrenme, yapay sinir ağlarının çok katmanlı versiyonlarını kullanan bir makine öğrenmesi yöntemidir. "
        "Görüntü tanıma, doğal dil işleme ve konuşma tanıma gibi alanlarda kullanılır.",
        
        "GPU'lar, paralel hesaplama yetenekleri sayesinde derin öğrenme eğitiminde kritik rol oynar. "
        "NVIDIA'nın CUDA platformu, GPU üzerinde genel amaçlı hesaplama yapılmasını sağlar.",
        
        "Büyük dil modelleri, milyarlarca parametre içeren ve büyük metin külliyatları üzerinde eğitilen modellerdir. "
        "GPT, LLaMA, Mistral ve Claude gibi modeller bu kategoriye girer.",
        
        "Tokenization, metni daha küçük parçalara ayırma işlemidir. "
        "BPE, WordPiece ve SentencePiece yaygın kullanılan tokenization yöntemleridir.",
        
        "Attention mekanizması, modelin girdi sekansındaki farklı pozisyonlara farklı ağırlıklar vermesini sağlar. "
        "Multi-head attention, bu işlemi birden fazla paralel attention başlıkları ile gerçekleştirir.",
        
        "Transfer öğrenme, bir görev üzerinde eğitilmiş modelin bilgisini başka bir göreve aktarma tekniğidir. "
        "Önceden eğitilmiş dil modelleri, fine-tuning ile özel görevlere uyarlanabilir.",
        
        "Kuantum bilgisayarlar, klasik bilgisayarlardan farklı olarak kübit kullanır. "
        "Süperpozisyon ve dolanıklık sayesinde belirli problemleri çok daha hızlı çözebilir.",
        
        "Siber güvenlik, bilgisayar sistemlerini yetkisiz erişimden koruma uygulamalarını kapsar. "
        "Şifreleme, güvenlik duvarları ve saldırı tespit sistemleri temel araçlardır.",
        
        "Bulut bilişim, bilgi işlem kaynaklarının internet üzerinden sunulmasıdır. "
        "Amazon AWS, Google Cloud ve Microsoft Azure en büyük sağlayıcılardır.",
        
        "Makine öğrenmesi algoritmaları denetimli, denetimsiz ve pekiştirmeli öğrenme olarak üç ana kategoriye ayrılır. "
        "Denetimli öğrenmede model, etiketli veriler üzerinde eğitilir.",
        
        "İstanbul, Türkiye'nin en kalabalık şehri ve ekonomik merkezidir. "
        "Boğaziçi Köprüsü ile Avrupa ve Asya kıtalarını birbirine bağlar.",
        
        "Doğal dil işleme, bilgisayarların insan dilini anlamasını ve üretmesini sağlayan yapay zeka dalıdır. "
        "Metin sınıflandırma, duygu analizi, makine çevirisi ve soru cevaplama görevlerini kapsar.",
        
        "5G teknolojisi, beşinci nesil mobil iletişim standardıdır. "
        "Düşük gecikme süresi ve yüksek bant genişliği ile nesnelerin interneti uygulamalarını destekler.",
        
        "Blokzincir teknolojisi, dağıtık ve değiştirilemez bir kayıt defteri sunar. "
        "Bitcoin ve Ethereum gibi kripto paralar blokzincir üzerine inşa edilmiştir.",
        
        "Robotik, robotların tasarımı, üretimi ve kullanımı ile ilgilenen mühendislik dalıdır. "
        "Endüstriyel robotlar, otonom araçlar ve insansı robotlar uygulama örnekleridir.",
        
        "Veri bilimi, veriden anlamlı bilgi çıkarmak için istatistik, matematik ve bilgisayar bilimini birleştirir. "
        "Pandas, NumPy ve Scikit-learn Python'daki en popüler veri bilimi kütüphaneleridir.",
    ]


def _get_builtin_sft_data() -> List[Dict]:
    """Yerleşik SFT verileri."""
    return [
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Python'da liste nasıl oluşturulur?"},
                {"role": "assistant", "content": "Python'da liste oluşturmak için köşeli parantez kullanılır:\n\n```python\n# Boş liste\nmy_list = []\n\n# Elemanlarla liste\nnumbers = [1, 2, 3, 4, 5]\nfruits = ['elma', 'armut', 'muz']\n```\n\nAyrıca `list()` fonksiyonu ile de oluşturabilirsiniz."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Yapay zeka nedir?"},
                {"role": "assistant", "content": "Yapay zeka (AI), makinelerin insan benzeri zekâ sergilemesini sağlayan bilgisayar bilimi dalıdır. Temel bileşenleri:\n\n1. **Makine Öğrenmesi**: Veriden öğrenme\n2. **Derin Öğrenme**: Çok katmanlı sinir ağları\n3. **Doğal Dil İşleme**: İnsan dilini anlama\n4. **Bilgisayarlı Görü**: Görüntü analizi"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Transformer mimarisi nasıl çalışır?"},
                {"role": "assistant", "content": "Transformer mimarisi şu bileşenlerden oluşur:\n\n1. **Self-Attention**: Her token, diğer tüm tokenlarla ilişkisini hesaplar\n   - Attention = softmax(QK^T / √d_k) × V\n\n2. **Multi-Head Attention**: Birden fazla attention başlığı paralel çalışır\n\n3. **Feed-Forward Network**: Her pozisyona ayrı uygulanan tam bağlantılı katman\n\n4. **Positional Encoding**: Token pozisyon bilgisi (modern modellerde RoPE)\n\n5. **Layer Normalization**: Eğitim stabilitesi (RMSNorm tercih edilir)"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Fibonacci dizisini Python'da nasıl yazarım?"},
                {"role": "assistant", "content": "```python\n# Yöntem 1: Döngü\ndef fibonacci(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\n# Yöntem 2: Generator\ndef fib_generator():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nprint(fibonacci(10))\n# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n```"},
            ]
        },
    ]


# ============================================================
# H100 Optimized Trainer
# ============================================================

class H100OptimizedTrainer:
    """
    H100 için optimize edilmiş trainer.
    
    Optimizasyonlar:
    - torch.compile (Triton kernel fusion)
    - BF16 native (Hopper architecture)
    - Zaman bütçeli eğitim
    - Optimize DataLoader
    - Minimal checkpoint overhead
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        time_budget_seconds: float = 900,
        output_dir: str = "output_h100",
        use_compile: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config
        self.time_budget = time_budget_seconds
        self.output_dir = output_dir
        self.use_compile = use_compile
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # BF16 setup (H100 native)
        self.use_amp = True
        self.amp_dtype = torch.bfloat16
        
        # torch.compile for H100 (Triton kernel fusion)
        if self.use_compile and self.device.type == "cuda":
            try:
                print("  ⚡ torch.compile uygulanıyor (bu birkaç dakika sürebilir)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  ✅ torch.compile başarılı!")
            except Exception as e:
                print(f"  ⚠️ torch.compile başarısız (normal devam): {e}")
        
        # Collate function
        self.collate_fn = create_data_collator(tokenizer, train_config.max_seq_length)
        
        # Global metrics
        self.all_metrics = {}
        self.pipeline_start_time = None
    
    def _create_dataloader(self, dataset, shuffle=True) -> DataLoader:
        """H100 optimize DataLoader."""
        kwargs = H100_DATALOADER_KWARGS.copy()
        
        # Küçük dataset'lerde persistent_workers kapatılabilir
        if len(dataset) < 10:
            kwargs["persistent_workers"] = False
            kwargs["num_workers"] = 0
            kwargs.pop("prefetch_factor", None)
        
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
    
    def _create_optimizer(self) -> torch.optim.AdamW:
        """Fused AdamW optimizer."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ['norm', 'bias', 'embed']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Fused AdamW for CUDA (faster)
        use_fused = self.device.type == "cuda"
        
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.train_config.learning_rate,
                betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
                eps=self.train_config.adam_epsilon,
                fused=use_fused,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.train_config.learning_rate,
                betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
                eps=self.train_config.adam_epsilon,
            )
        
        return optimizer
    
    def _get_cosine_lr(self, step: int, total_steps: int, warmup_steps: int, lr: float) -> float:
        """Cosine learning rate with warmup."""
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
    
    def _train_stage(
        self,
        stage_name: str,
        dataset,
        learning_rate: float,
        max_epochs: int,
        time_limit_seconds: float,
        gradient_accumulation: int = 1,
    ) -> Dict[str, Any]:
        """
        Tek bir eğitim aşamasını zamaan bütçesi ile çalıştır.
        
        Returns:
            Eğitim metrikleri
        """
        print(f"\n{'='*60}")
        print(f"  ⚡ {stage_name} (time limit: {time_limit_seconds:.0f}s)")
        print(f"{'='*60}")
        
        if dataset is None or len(dataset) < 2:
            print(f"  ⚠️ {stage_name}: Yeterli veri yok, atlanıyor.")
            return {"skipped": True}
        
        # Train/eval split
        eval_size = max(1, int(len(dataset) * 0.05))
        train_size = len(dataset) - eval_size
        train_ds, eval_ds = random_split(dataset, [train_size, eval_size])
        
        print(f"  Dataset: {len(dataset)} total ({train_size} train, {eval_size} eval)")
        
        # DataLoader
        train_loader = self._create_dataloader(train_ds, shuffle=True)
        
        if len(train_loader) == 0:
            print(f"  ⚠️ Batch boyutu veri sayısından büyük, batch_size azaltılıyor...")
            self.train_config.batch_size = max(1, len(train_ds))
            train_loader = self._create_dataloader(train_ds, shuffle=True)
        
        # Optimizer
        optimizer = self._create_optimizer()
        
        # Total steps estimation
        steps_per_epoch = len(train_loader) // gradient_accumulation
        total_steps = steps_per_epoch * max_epochs
        warmup_steps = max(1, int(total_steps * 0.05))
        
        print(f"  Steps/epoch: {steps_per_epoch}, Max epochs: {max_epochs}")
        print(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {self.train_config.batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation}")
        print(f"  Effective batch size: {self.train_config.batch_size * gradient_accumulation}")
        
        # Training loop
        self.model.train()
        stage_start = time.time()
        global_step = 0
        total_loss = 0.0
        log_loss = 0.0
        log_steps = 0
        best_loss = float('inf')
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # ⏱️ Zaman kontrolü
                elapsed = time.time() - stage_start
                if elapsed >= time_limit_seconds:
                    print(f"\n  ⏱️ Zaman bütçesi doldu ({elapsed:.0f}s >= {time_limit_seconds:.0f}s)")
                    break
                
                # Move to device
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                # Forward pass (BF16)
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    loss = outputs.loss / gradient_accumulation
                
                # Backward pass (BF16 doesn't need scaler!)
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation
                epoch_steps += 1
                log_loss += loss.item() * gradient_accumulation
                log_steps += 1
                
                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.max_grad_norm,
                    )
                    
                    # LR update
                    current_lr = self._get_cosine_lr(global_step, total_steps, warmup_steps, learning_rate)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    
                    # Log (her 20 adımda bir — overhead azaltmak için)
                    if global_step % 20 == 0:
                        avg_loss = log_loss / max(log_steps, 1)
                        ppl = math.exp(min(avg_loss, 20))
                        elapsed = time.time() - stage_start
                        remaining = time_limit_seconds - elapsed
                        speed = global_step / elapsed if elapsed > 0 else 0
                        
                        print(
                            f"  [{stage_name}] Step {global_step:>5d} | "
                            f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Speed: {speed:.1f} step/s | "
                            f"⏱️ {remaining:.0f}s kaldı"
                        )
                        log_loss = 0.0
                        log_steps = 0
            
            # Zaman kontrolü (epoch sonrası)
            elapsed = time.time() - stage_start
            if elapsed >= time_limit_seconds:
                break
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            ppl = math.exp(min(avg_epoch_loss, 20))
            print(f"  📊 Epoch {epoch+1}/{max_epochs}: Loss={avg_epoch_loss:.4f}, PPL={ppl:.1f}")
        
        # Stage metrics
        stage_time = time.time() - stage_start
        final_loss = epoch_loss / max(epoch_steps, 1) if epoch_steps > 0 else float('inf')
        
        metrics = {
            "stage": stage_name,
            "steps": global_step,
            "epochs_completed": epoch + 1,
            "final_loss": final_loss,
            "final_ppl": math.exp(min(final_loss, 20)),
            "time_seconds": stage_time,
            "samples_processed": global_step * self.train_config.batch_size * gradient_accumulation,
        }
        
        print(f"\n  ✅ {stage_name} tamamlandı!")
        print(f"     Steps: {global_step} | Loss: {final_loss:.4f} | Süre: {stage_time:.0f}s")
        
        return metrics
    
    def run_pipeline(self, data: Dict[str, List]) -> Dict[str, Any]:
        """
        Tam eğitim pipeline'ını çalıştır.
        
        3 aşama: Pretrain → SFT → CoT
        Zaman bütçesi otomatik bölünür.
        """
        self.pipeline_start_time = time.time()
        
        print(f"\n⏱️ Toplam zaman bütçesi: {self.time_budget:.0f}s ({self.time_budget/60:.1f} dk)")
        
        pretrain_texts = data.get("pretrain", [])
        sft_data = data.get("sft", [])
        cot_data = data.get("cot", [])
        
        # Zaman bütçeleri
        pretrain_time = self.time_budget * TIME_BUDGET_RATIOS["pretrain"]
        sft_time = self.time_budget * TIME_BUDGET_RATIOS["sft"]
        cot_time = self.time_budget * TIME_BUDGET_RATIOS["cot"]
        
        print(f"  Pre-train: {pretrain_time:.0f}s | SFT: {sft_time:.0f}s | CoT: {cot_time:.0f}s")
        
        all_metrics = {}
        
        # ===== Tokenizer Eğitimi =====
        tok_start = time.time()
        if pretrain_texts and len(pretrain_texts) > 5:
            print("\n🔤 Tokenizer eğitiliyor...")
            try:
                self.tokenizer.train(
                    texts=pretrain_texts[:500],  # İlk 500 metin yeterli
                    vocab_size=self.model_config.vocab_size,
                    output_path=os.path.join(self.output_dir, "tokenizer"),
                )
                print("  ✅ Tokenizer eğitimi tamamlandı")
            except Exception as e:
                print(f"  ⚠️ Tokenizer eğitimi başarısız (byte-fallback devam): {e}")
        tok_time = time.time() - tok_start
        
        # Kalan zamanı yeniden hesapla
        remaining = self.time_budget - tok_time
        pretrain_time = remaining * TIME_BUDGET_RATIOS["pretrain"] / (1 - TIME_BUDGET_RATIOS["tokenizer"])
        sft_time = remaining * TIME_BUDGET_RATIOS["sft"] / (1 - TIME_BUDGET_RATIOS["tokenizer"])
        cot_time = remaining * TIME_BUDGET_RATIOS["cot"] / (1 - TIME_BUDGET_RATIOS["tokenizer"])
        
        # ===== AŞAMA 1: Pre-training =====
        if pretrain_texts and len(pretrain_texts) >= 5:
            pretrain_dataset = TextDataset(
                texts=pretrain_texts,
                tokenizer=self.tokenizer,
                max_length=self.train_config.max_seq_length,
            )
            
            metrics = self._train_stage(
                stage_name="Pre-training",
                dataset=pretrain_dataset,
                learning_rate=3e-4,
                max_epochs=5,
                time_limit_seconds=pretrain_time,
                gradient_accumulation=self.train_config.gradient_accumulation_steps,
            )
            all_metrics["pretrain"] = metrics
        
        # ===== AŞAMA 2: SFT =====
        if sft_data and len(sft_data) >= 2:
            sft_dataset = ChatDataset(
                data=sft_data,
                tokenizer=self.tokenizer,
                max_length=self.train_config.max_seq_length,
                mask_user_tokens=True,
            )
            
            # Kalan zamanı kontrol et
            elapsed_total = time.time() - self.pipeline_start_time
            sft_time = min(sft_time, self.time_budget - elapsed_total - cot_time - 30)
            sft_time = max(sft_time, 60)  # Minimum 60 saniye
            
            metrics = self._train_stage(
                stage_name="SFT",
                dataset=sft_dataset,
                learning_rate=2e-5,
                max_epochs=3,
                time_limit_seconds=sft_time,
                gradient_accumulation=self.train_config.gradient_accumulation_steps,
            )
            all_metrics["sft"] = metrics
        
        # ===== AŞAMA 3: CoT =====
        if cot_data and len(cot_data) >= 2:
            cot_dataset = CoTDataset(
                data=cot_data,
                tokenizer=self.tokenizer,
                max_length=self.train_config.max_seq_length,
            )
            
            # Kalan zamanı kontrol et
            elapsed_total = time.time() - self.pipeline_start_time
            cot_time = min(cot_time, self.time_budget - elapsed_total - 30)
            cot_time = max(cot_time, 60)  # Minimum 60 saniye
            
            metrics = self._train_stage(
                stage_name="CoT",
                dataset=cot_dataset,
                learning_rate=1e-5,
                max_epochs=5,
                time_limit_seconds=cot_time,
                gradient_accumulation=max(1, self.train_config.gradient_accumulation_steps * 2),
            )
            all_metrics["cot"] = metrics
        
        # Pipeline toplam süre
        total_time = time.time() - self.pipeline_start_time
        all_metrics["total_time_seconds"] = total_time
        all_metrics["total_time_minutes"] = total_time / 60
        
        self.all_metrics = all_metrics
        return all_metrics
    
    def evaluate_model(self):
        """Hızlı model değerlendirme."""
        print("\n🔍 Model Değerlendirme...")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        try:
            generator = TextGenerator(self.model, self.tokenizer, device=device)
        except Exception as e:
            print(f"  ⚠️ Generator oluşturulamadı: {e}")
            return
        
        test_prompts = [
            "Yapay zeka nedir?",
            "Python'da for döngüsü nasıl kullanılır?",
            "Merhaba, nasılsın?",
            "Transformer mimarisi nasıl çalışır?",
        ]
        
        print("\n📝 Örnek çıktılar:\n")
        for prompt in test_prompts:
            print(f"  Q: {prompt}")
            try:
                response = generator.generate(
                    prompt=prompt,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.9,
                )
                display = response[:200].replace('\n', ' ')
                print(f"  A: {display}\n")
            except Exception as e:
                print(f"  ⚠️ Hata: {e}\n")
    
    def save_model(self):
        """Model ve tüm dosyaları kaydet."""
        save_path = os.path.join(self.output_dir, "final_model")
        print(f"\n💾 Model kaydediliyor: {save_path}")
        
        # unwrap compiled model if needed
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod
        
        # Model kaydet
        model_to_save.save_pretrained(save_path)
        print("  ✅ Model ağırlıkları kaydedildi")
        
        # Tokenizer kaydet
        tokenizer_path = os.path.join(save_path, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        self.tokenizer.save(tokenizer_path)
        print("  ✅ Tokenizer kaydedildi")
        
        # Eğitim bilgileri
        info = {
            "model_name": self.model_config.model_name,
            "model_version": self.model_config.model_version,
            "hidden_size": self.model_config.hidden_size,
            "num_layers": self.model_config.num_layers,
            "num_attention_heads": self.model_config.num_attention_heads,
            "num_kv_heads": self.model_config.num_kv_heads,
            "vocab_size": self.model_config.vocab_size,
            "parameters": count_parameters(model_to_save),
            "cot_enabled": self.model_config.cot_enabled,
            "gpu": "H100",
            "optimization": "torch.compile + BF16 + time-budgeted",
            "training_metrics": self.all_metrics,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        }
        
        info_path = os.path.join(save_path, "training_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)
        print("  ✅ Eğitim bilgileri kaydedildi")
        
        print(f"\n🎉 Model başarıyla kaydedildi: {save_path}")
        return save_path


# ============================================================
# GitHub Push
# ============================================================

def push_to_github(output_dir: str):
    """Değişiklikleri GitHub'a push et."""
    print("\n📤 GitHub'a push ediliyor...")
    
    try:
        import subprocess
        
        # Git add
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        print("  ✅ git add tamamlandı")
        
        # Git commit
        commit_msg = "Add frontier-quality datasets + H100 optimized training"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  ✅ git commit: {commit_msg}")
        else:
            print(f"  ℹ️ Commit: {result.stdout.strip() or result.stderr.strip()}")
        
        # Git push
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  ✅ git push başarılı!")
        else:
            print(f"  ⚠️ Push hatası: {result.stderr.strip()}")
            # Force push dene
            result = subprocess.run(
                ["git", "push", "-f", "origin", "main"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  ✅ Force push başarılı!")
            else:
                print(f"  ❌ Push başarısız: {result.stderr.strip()}")
                
    except Exception as e:
        print(f"  ❌ GitHub push hatası: {e}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="H100 Optimized LLM Training")
    
    parser.add_argument("--model", type=str, default="large",
                        choices=["nano", "small", "medium", "large", "xl"],
                        help="Model büyüklüğü (default: large for H100)")
    parser.add_argument("--time_budget", type=int, default=900,
                        help="Toplam eğitim süresi saniye (default: 900 = 15dk)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch boyutu (default: H100 preset)")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Max sequence uzunluğu (default: H100 preset)")
    parser.add_argument("--output_dir", type=str, default="output_h100",
                        help="Çıktı dizini")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_compile", action="store_true",
                        help="torch.compile kullanma")
    parser.add_argument("--no_push", action="store_true",
                        help="GitHub'a push etme")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Değerlendirmeyi atla")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print_banner()
    
    # Seed
    set_seed(args.seed)
    
    # ===== GPU Info =====
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"🖥️  GPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_mem:.1f} GB")
        print(f"🔧 CUDA: {torch.version.cuda}")
        print(f"🔥 PyTorch: {torch.__version__}")
    else:
        print("⚠️ CUDA bulunamadı! CPU mode (çok yavaş)")
    
    # ===== Config =====
    model_config = PRESET_CONFIGS[args.model]
    
    # H100 preset training config
    train_config = TrainingConfig(
        batch_size=args.batch_size or 16,
        max_seq_length=args.max_seq_length or 2048,  # 2048 for faster training, 4096 for quality
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        num_workers=4,
    )
    
    print(f"\n⚙️ Model: {model_config.model_name}")
    print(f"  Hidden: {model_config.hidden_size} | Layers: {model_config.num_layers}")
    print(f"  Heads: {model_config.num_attention_heads} | KV Heads: {model_config.num_kv_heads}")
    print(f"  Batch: {train_config.batch_size} | Seq: {train_config.max_seq_length}")
    print(f"  Precision: {train_config.mixed_precision}")
    print(f"  torch.compile: {not args.no_compile}")
    
    # ===== Load Datasets =====
    data = load_all_datasets(".")
    
    # ===== Create Model =====
    print("\n🏗️  Model oluşturuluyor...")
    model = ModernLLMForCausalLM(model_config)
    params = count_parameters(model)
    print(f"  Parametreler: {format_params(params['total'])}")
    print(f"  Eğitilebilir: {format_params(params['trainable'])}")
    
    # ===== Create Tokenizer =====
    tokenizer = ModernTokenizer(vocab_size=model_config.vocab_size)
    
    # ===== Train =====
    trainer = H100OptimizedTrainer(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config,
        time_budget_seconds=args.time_budget,
        output_dir=args.output_dir,
        use_compile=not args.no_compile,
    )
    
    metrics = trainer.run_pipeline(data)
    
    # ===== Evaluate =====
    if not args.skip_eval:
        trainer.evaluate_model()
    
    # ===== Save =====
    save_path = trainer.save_model()
    
    # ===== GitHub Push =====
    if not args.no_push:
        push_to_github(args.output_dir)
    
    # ===== Summary =====
    total_time = metrics.get("total_time_seconds", 0)
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    
    print(f"""
{'='*60}
🎉 EĞİTİM TAMAMLANDI!
{'='*60}

  Model: {model_config.model_name}
  Parametreler: {format_params(params['total'])}
  Toplam süre: {mins}dk {secs}sn
  Kayıt: {save_path}
  
  📊 Aşama Metrikleri:
""")
    
    for stage in ["pretrain", "sft", "cot"]:
        if stage in metrics and not metrics[stage].get("skipped"):
            m = metrics[stage]
            print(f"    {stage}: Loss={m['final_loss']:.4f}, Steps={m['steps']}, Time={m['time_seconds']:.0f}s")
    
    print(f"""
  📥 Modeli indirmek için:
    # Colab'dan indirme:
    from google.colab import files
    !zip -r model.zip {save_path}
    files.download('model.zip')
    
    # Veya GitHub'dan:
    git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
    
  🔮 Modeli kullanmak için:
    from modern_llm.model.transformer import ModernLLMForCausalLM
    from modern_llm.inference.generator import TextGenerator
    
    model = ModernLLMForCausalLM.from_pretrained("{save_path}")
    generator = TextGenerator(model, tokenizer)
    response = generator.generate("Merhaba!", max_new_tokens=100)
""")


if __name__ == "__main__":
    main()
