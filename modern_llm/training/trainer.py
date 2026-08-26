"""
Trainer Modülü
==============

Sıfırdan yazılmış modern LLM eğitim döngüsü.

Özellikler:
- Mixed precision eğitim (BF16/FP16)
- Gradient accumulation
- Gradient clipping
- Cosine learning rate schedule with warmup
- Checkpoint kaydetme ve yükleme
- Wandb logging
- Gradient checkpointing (bellek optimizasyonu)
- Eval loop ile validation
- Training metrics takibi
"""

import os
import json
import time
import math
import inspect
from contextlib import nullcontext
from typing import Optional, Dict, Any, List
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler

from modern_llm.config import ModelConfig, TrainingConfig


class Trainer:
    """
    Modern LLM Trainer.
    
    Sıfırdan yazılmış eğitim döngüsü.
    HuggingFace Trainer kullanmadan, tam kontrol ile eğitim.
    
    Kullanım:
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            training_config=train_config,
            tokenizer=tokenizer,
        )
        trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset=None,
        training_config: Optional[TrainingConfig] = None,
        tokenizer=None,
        collate_fn=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config or TrainingConfig()
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Mixed precision. CPU FP16 autocast desteklenmediği için FP32'ye düşer.
        requested_amp = self.config.mixed_precision in ("bf16", "fp16")
        cpu_fp16 = self.device.type == "cpu" and self.config.mixed_precision == "fp16"
        self.use_amp = requested_amp and not cpu_fp16
        self.amp_dtype = (
            torch.bfloat16
            if self.config.mixed_precision == "bf16"
            else torch.float16
        )
        self.use_grad_scaler = (
            self.device.type == "cuda"
            and self.config.mixed_precision == "fp16"
        )
        self.scaler = GradScaler(enabled=self.use_grad_scaler)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # DataLoaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_loader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None
        
        # Learning rate scheduler. Son kısmi accumulation grubu da bir step'tir.
        updates_per_epoch = math.ceil(
            len(self.train_loader) / self.config.gradient_accumulation_steps
        )
        self.total_steps = max(1, updates_per_epoch * self.config.num_epochs)
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.training_logs = []
        
        # Wandb
        self.wandb_run = None
        if self.config.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        AdamW optimizer oluştur.
        
        Weight decay:
        - Embedding ve norm katmanlarına uygulanmaz
        - Linear katmanlara uygulanır
        """
        # Weight decay grupları
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Norm ve embedding parametrelerine weight decay uygulanmaz
            if any(nd in name for nd in ['norm', 'bias', 'embed']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_epsilon,
        }
        
        # fused yalnızca destekleyen PyTorch sürümlerinde ve CUDA'da gönderilir.
        if (
            self.device.type == "cuda"
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        ):
            optimizer_kwargs["fused"] = True
        
        return torch.optim.AdamW(param_groups, **optimizer_kwargs)
    
    def _autocast_context(self):
        """Cihaz ve precision için geçerli autocast context'i döndür."""
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )
    
    def _create_scheduler(self):
        """Cosine learning rate scheduler with warmup."""
        # Warmup adımları
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        
        def lr_lambda(current_step):
            # Warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(
                max(1, self.total_steps - warmup_steps)
            )
            
            if self.config.lr_scheduler_type == "cosine":
                return max(
                    self.config.min_lr_ratio,
                    0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            elif self.config.lr_scheduler_type == "linear":
                return max(
                    self.config.min_lr_ratio,
                    1.0 - progress,
                )
            else:  # constant
                return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_dataloader(self, dataset, shuffle: bool = True) -> Optional[DataLoader]:
        """DataLoader oluştur."""
        if dataset is None:
            return None
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=min(self.config.num_workers, os.cpu_count() or 1),
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
    
    def _init_wandb(self):
        """Wandb'ı başlat."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "model": self.model.config.to_dict() if hasattr(self.model, 'config') else {},
                    "training": self.config.to_dict(),
                },
            )
        except ImportError:
            print("⚠️ wandb yüklü değil: pip install wandb")
            self.config.use_wandb = False
    
    def train(self):
        """
        Ana eğitim döngüsü.
        
        Epoch -> Batch -> Forward -> Loss -> Backward -> Step
        """
        print("=" * 60)
        print("🚀 Eğitim Başlıyor")
        print("=" * 60)
        print(f"  Model: {self.model.config.model_name if hasattr(self.model, 'config') else 'Unknown'}")
        print(f"  Parametreler: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Eğitilebilir: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Device: {self.device}")
        print(f"  Precision: {self.config.mixed_precision}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Grad accum: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Train samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"  Eval samples: {len(self.eval_dataset)}")
        print("=" * 60)
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            num_batches = len(self.train_loader)
            remainder = num_batches % self.config.gradient_accumulation_steps
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Batch'i device'a taşı
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                in_partial_group = (
                    remainder > 0 and batch_idx >= num_batches - remainder
                )
                accumulation_divisor = (
                    remainder
                    if in_partial_group
                    else self.config.gradient_accumulation_steps
                )
                
                # Forward pass (mixed precision)
                with self._autocast_context():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    raw_loss = outputs.loss
                    loss = raw_loss / accumulation_divisor
                
                # Backward pass
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += raw_loss.item()
                epoch_steps += 1
                
                should_step = (
                    (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_idx + 1 == num_batches
                )
                
                # Gradient accumulation
                if should_step:
                    # Gradient clipping
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    
                    # Optimizer step
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        log_entry = {
                            "step": self.global_step,
                            "epoch": epoch + (batch_idx + 1) / len(self.train_loader),
                            "loss": avg_loss,
                            "lr": lr,
                            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "elapsed_seconds": elapsed,
                        }
                        self.training_logs.append(log_entry)
                        
                        # Perplexity
                        ppl = math.exp(min(avg_loss, 20))  # Overflow guard
                        
                        print(
                            f"  Step {self.global_step:>6d}/{self.total_steps} | "
                            f"Epoch {epoch+1} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"PPL: {ppl:.2f} | "
                            f"LR: {lr:.2e} | "
                            f"GradNorm: {grad_norm:.2f}"
                        )
                        
                        if self.wandb_run:
                            import wandb
                            wandb.log(log_entry)
                    
                    # Evaluation
                    if self.eval_loader and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self.model.train()
                        
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best")
                    
                    # Checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # Epoch sonu
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            ppl = math.exp(min(avg_epoch_loss, 20))
            print(f"\n📊 Epoch {epoch + 1}/{self.config.num_epochs} tamamlandı")
            print(f"   Ortalama Loss: {avg_epoch_loss:.4f} | PPL: {ppl:.2f}")
            
            # Epoch sonu evaluation
            if self.eval_loader:
                eval_loss = self.evaluate()
                self.model.train()
            
            # Epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        # Eğitim tamamlandı
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ Eğitim Tamamlandı!")
        print(f"   Süre: {total_time / 3600:.2f} saat")
        print(f"   Final Loss: {avg_epoch_loss:.4f}")
        print(f"   Best Eval Loss: {self.best_eval_loss:.4f}")
        print("=" * 60)
        
        # Final model kaydet
        self.save_checkpoint("final")
        
        # Training log kaydet
        self._save_training_log()
        
        if self.wandb_run:
            import wandb
            wandb.finish()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluation loop.
        
        Returns:
            Ortalama eval loss
        """
        if self.eval_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with self._autocast_context():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        
        print(f"\n📋 Eval | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
        
        if self.wandb_run:
            import wandb
            wandb.log({
                "eval_loss": avg_loss,
                "eval_ppl": ppl,
                "step": self.global_step,
            })
        
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Checkpoint kaydet."""
        save_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # Optimizer & Scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(save_dir, "trainer_state.pt"))
        
        # Training config
        with open(os.path.join(save_dir, "training_config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"   💾 Checkpoint kaydedildi: {save_dir}")
        
        # Eski checkpoint'ları temizle
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, path: str):
        """Checkpoint'tan yükle."""
        # Model
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        
        # Trainer state
        state_path = os.path.join(path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_eval_loss = state["best_eval_loss"]
        
        print(f"✅ Checkpoint yüklendi: {path}")
    
    def _cleanup_checkpoints(self):
        """Eski checkpoint'ları temizle."""
        if self.config.save_total_limit <= 0:
            return
        
        import glob
        import shutil
        
        checkpoints = sorted(
            glob.glob(os.path.join(self.config.output_dir, "step_*")),
            key=os.path.getmtime,
        )
        
        # "best", "final", "epoch_*" korunur
        while len(checkpoints) > self.config.save_total_limit:
            old = checkpoints.pop(0)
            shutil.rmtree(old, ignore_errors=True)
    
    def _save_training_log(self):
        """Training logunu kaydet."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        log_path = os.path.join(self.config.output_dir, "training_log.json")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_logs, f, indent=2)
        
        print(f"   📝 Training log kaydedildi: {log_path}")
