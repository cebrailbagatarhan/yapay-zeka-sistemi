"""
MATH-openai-split Dataset ile Model Eğitimi
12,500 matematik problemi ile GPT-2 fine-tuning
"""

import os
import time
import json
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathDataset(Dataset):
    """Matematik problemleri için özel dataset"""
    
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Problem ve çözümü birleştir
        text = f"Problem: {item['problem']}\n\nSolution: {item['solution']}\n\nAnswer: {item['answer']}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

def check_gpu():
    """GPU kontrolü"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU bulundu: {gpu_name}")
        logger.info(f"💾 GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        logger.warning("⚠️ GPU bulunamadı! CPU ile eğitim yapılacak (çok yavaş olabilir)")
        return False

def download_and_prepare_dataset():
    """Dataset'i indir ve hazırla"""
    logger.info("📥 MATH-openai-split dataset indiriliyor...")
    
    try:
        # Dataset'i indir
        dataset = load_dataset("MathMindsAGI/MATH-openai-split")
        
        logger.info(f"✅ Dataset indirildi!")
        logger.info(f"📊 Train örnekleri: {len(dataset['train'])}")
        logger.info(f"📊 Test örnekleri: {len(dataset['test']) if 'test' in dataset else 'Yok'}")
        
        # İlk örneği göster
        logger.info("\n📝 Örnek problem:")
        sample = dataset['train'][0]
        logger.info(f"Problem: {sample['problem'][:100]}...")
        logger.info(f"Subject: {sample['subject']}")
        logger.info(f"Level: {sample['level']}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Dataset indirilemedi: {e}")
        return None

def setup_model_and_tokenizer():
    """Model ve tokenizer'ı hazırla"""
    logger.info("🤖 Model hazırlanıyor...")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # GPU'ya taşı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"✅ Model hazır: {model.num_parameters():,} parametreler")
    logger.info(f"📍 Device: {device}")
    
    return model, tokenizer, device

def train_model(model, tokenizer, dataset, output_dir="./math-gpt2-model"):
    """Modeli eğit"""
    logger.info("\n🚀 Eğitim başlıyor...")
    
    # Dataset'i hazırla
    train_dataset = MathDataset(dataset['train'], tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 3 epoch (25-30 dakika)
        per_device_train_batch_size=8,  # Batch size
        gradient_accumulation_steps=2,  # Effective batch: 16
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),  # Mixed precision (sadece GPU'da)
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        report_to="none",  # TensorBoard kapalı
        disable_tqdm=False,
        load_best_model_at_end=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 için CLM (Causal Language Modeling)
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Eğitimi başlat
    logger.info(f"⏱️ Tahmini süre: 25-40 dakika (3 epoch)")
    logger.info(f"📊 Toplam örnek: {len(train_dataset)}")
    logger.info(f"📦 Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"🔄 Epoch: {training_args.num_train_epochs}")
    
    start_time = time.time()
    
    try:
        trainer.train()
        
        # Süreyi hesapla
        elapsed_time = time.time() - start_time
        minutes = elapsed_time / 60
        
        logger.info(f"\n✅ Eğitim tamamlandı!")
        logger.info(f"⏱️ Toplam süre: {minutes:.2f} dakika ({elapsed_time/3600:.2f} saat)")
        
        # Modeli kaydet
        logger.info(f"💾 Model kaydediliyor: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # İstatistikleri kaydet
        stats = {
            "training_time_minutes": minutes,
            "training_time_hours": elapsed_time / 3600,
            "num_examples": len(train_dataset),
            "num_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "model_name": "gpt2",
            "dataset": "MathMindsAGI/MATH-openai-split",
            "gpu_used": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        
        with open(f"{output_dir}/training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\n📊 Eğitim İstatistikleri:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return trainer, stats
        
    except Exception as e:
        logger.error(f"❌ Eğitim hatası: {e}")
        raise

def test_model(model, tokenizer):
    """Modeli test et"""
    logger.info("\n🧪 Model testi yapılıyor...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Test sorusu
    test_problem = "What is the sum of 2 + 2?"
    prompt = f"Problem: {test_problem}\n\nSolution:"
    
    logger.info(f"\n📝 Test Sorusu: {test_problem}")
    logger.info("🤔 Model cevabı üretiyor...")
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"\n💡 Model Cevabı:\n{response}\n")
    
    return response

def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("🎓 MATH-openai-split Dataset ile Model Eğitimi")
    print("="*70 + "\n")
    
    # GPU kontrolü
    has_gpu = check_gpu()
    if not has_gpu:
        response = input("\n⚠️ GPU bulunamadı! CPU ile devam etmek çok yavaş olabilir. Devam? (e/h): ")
        if response.lower() != 'e':
            logger.info("❌ Eğitim iptal edildi.")
            return
    
    # Dataset indir
    dataset = download_and_prepare_dataset()
    if dataset is None:
        logger.error("❌ Dataset indirilemedi, çıkılıyor.")
        return
    
    # Model hazırla
    model, tokenizer, device = setup_model_and_tokenizer()
    
    # Kullanıcıya sor
    print("\n" + "-"*70)
    print("🎯 Eğitim Ayarları:")
    print(f"  • Model: GPT-2 Small (124M parametreler)")
    print(f"  • Dataset: 12,500 matematik problemi")
    print(f"  • Epoch: 3")
    print(f"  • Tahmini Süre: 25-40 dakika")
    print(f"  • GPU: {'✅ ' + torch.cuda.get_device_name(0) if has_gpu else '❌ CPU (yavaş!)'}")
    print("-"*70)
    
    response = input("\n🚀 Eğitimi başlatmak istiyor musunuz? (e/h): ")
    if response.lower() != 'e':
        logger.info("❌ Eğitim iptal edildi.")
        return
    
    # Eğitimi başlat
    try:
        trainer, stats = train_model(model, tokenizer, dataset)
        
        # Test yap
        test_model(model, tokenizer)
        
        print("\n" + "="*70)
        print("✅ TÜM İŞLEMLER TAMAMLANDI!")
        print("="*70)
        print(f"\n📁 Model kaydedildi: ./math-gpt2-model/")
        print(f"⏱️ Toplam süre: {stats['training_time_minutes']:.2f} dakika")
        print(f"\n💡 Kullanım:")
        print(f"   from transformers import GPT2LMHeadModel, GPT2Tokenizer")
        print(f"   model = GPT2LMHeadModel.from_pretrained('./math-gpt2-model')")
        print(f"   tokenizer = GPT2Tokenizer.from_pretrained('./math-gpt2-model')")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Eğitim kullanıcı tarafından durduruldu!")
    except Exception as e:
        logger.error(f"\n❌ Hata oluştu: {e}")
        raise

if __name__ == "__main__":
    main()
