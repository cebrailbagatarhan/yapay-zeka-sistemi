"""
Çoklu Dataset Eğitim Sistemi
Birden fazla kaliteli dataset ile sıralı eğitim
"""

import os
import time
import json
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

print("=" * 80)
print("🎯 ÇOKLU DATASET EĞİTİM SİSTEMİ")
print("=" * 80)

# GPU kontrolü
has_gpu = torch.cuda.is_available()
device = "cuda" if has_gpu else "cpu"
print(f"\n{'✅' if has_gpu else '⚠️'} GPU: {torch.cuda.get_device_name(0) if has_gpu else 'CPU'}")

# Dataset listesi (CPU için optimize edilmiş)
DATASETS = [
    {
        "name": "fka/awesome-chatgpt-prompts",
        "size": 203,
        "time_cpu": "5-10 dk",
        "time_gpu": "2-3 dk",
        "desc": "ChatGPT prompts - Rol oynama",
        "quality": "⭐⭐⭐⭐"
    },
    {
        "name": "timdettmers/openassistant-guanaco",
        "size": 10000,
        "time_cpu": "1-2 saat",
        "time_gpu": "10-15 dk",
        "desc": "Sohbet konuşmaları",
        "quality": "⭐⭐⭐⭐"
    },
    {
        "name": "databricks/databricks-dolly-15k",
        "size": 15000,
        "time_cpu": "2-3 saat",
        "time_gpu": "15-20 dk",
        "desc": "Instruction dataset",
        "quality": "⭐⭐⭐⭐⭐"
    },
]

print("\n" + "=" * 80)
print("📊 EĞİTİM PLANI")
print("=" * 80)

total_time_cpu = 0
total_time_gpu = 0

for i, ds in enumerate(DATASETS, 1):
    print(f"\n{i}. {ds['name']}")
    print(f"   📊 Boyut: {ds['size']:,} örnek")
    print(f"   ⭐ Kalite: {ds['quality']}")
    print(f"   📝 Açıklama: {ds['desc']}")
    print(f"   ⏱️ Süre: {ds['time_cpu']} (CPU) / {ds['time_gpu']} (GPU)")

if not has_gpu:
    print("\n" + "=" * 80)
    print("⚠️ CPU İLE EĞİTİM UYARISI")
    print("=" * 80)
    print("\nToplam tahmini süre:")
    print("  Dataset 1: ~10 dakika")
    print("  Dataset 2: ~2 saat")
    print("  Dataset 3: ~3 saat")
    print("  ─────────────────────")
    print("  TOPLAM: ~5-6 saat")
    
    print("\n💡 ÖNERİLER:")
    print("  1. Sadece Dataset 1 (hızlı test)")
    print("  2. Dataset 1 + 2 (iyi denge)")
    print("  3. Hepsini eğit (en iyi sonuç, uzun süre)")
    
    choice = input("\n🚀 Hangi seçeneği istersiniz? (1/2/3): ").strip()
    
    if choice == "1":
        DATASETS = [DATASETS[0]]
        print("\n✅ Sadece ChatGPT prompts ile eğitilecek (~10 dakika)")
    elif choice == "2":
        DATASETS = DATASETS[:2]
        print("\n✅ ChatGPT + Guanaco ile eğitilecek (~2 saat)")
    else:
        print("\n✅ Tüm dataset'lerle eğitilecek (~5-6 saat)")
else:
    print("\n✅ GPU var! Tüm dataset'ler hızlıca eğitilecek (~30-40 dakika)")

# Modeli başlat
print("\n" + "=" * 80)
print("🤖 MODEL HAZIRLANIYOR")
print("=" * 80)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

print(f"✅ Model hazır: {model.num_parameters():,} parametreler")
print(f"📍 Device: {device}")

# Dataset'leri sırayla eğit
all_stats = []

for idx, ds_info in enumerate(DATASETS, 1):
    print("\n" + "=" * 80)
    print(f"📦 DATASET {idx}/{len(DATASETS)}: {ds_info['name']}")
    print("=" * 80)
    
    try:
        # Dataset indir
        print(f"\n📥 Dataset indiriliyor...")
        dataset = load_dataset(ds_info['name'])
        
        # Train split'i al
        if 'train' in dataset:
            train_data = dataset['train']
        else:
            train_data = dataset[list(dataset.keys())[0]]
        
        print(f"✅ {len(train_data):,} örnek yüklendi")
        
        # İlk örneği göster
        print(f"\n📝 Örnek:")
        print(str(train_data[0])[:200] + "...")
        
        # Tokenize fonksiyonu
        def tokenize_function(examples):
            # Dataset yapısına göre text oluştur
            texts = []
            
            if 'text' in examples:
                texts = examples['text']
            elif 'instruction' in examples and 'response' in examples:
                texts = [f"Question: {q}\nAnswer: {a}" 
                        for q, a in zip(examples['instruction'], examples['response'])]
            elif 'prompt' in examples and 'response' in examples:
                texts = [f"{p}\n{r}" 
                        for p, r in zip(examples['prompt'], examples['response'])]
            elif 'act' in examples and 'prompt' in examples:
                texts = [f"{act}: {prompt}" 
                        for act, prompt in zip(examples['act'], examples['prompt'])]
            else:
                # Fallback: en uzun string alanı kullan
                text_keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
                if text_keys:
                    longest_key = max(text_keys, key=lambda k: len(str(examples[k][0])))
                    texts = examples[longest_key]
                else:
                    texts = [str(ex) for ex in examples]
            
            result = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
            return result
        
        print(f"\n📝 Tokenize ediliyor...")
        tokenized = train_data.map(
            tokenize_function,
            batched=True,
            remove_columns=train_data.column_names,
            desc="Tokenizing"
        )
        
        # Labels ekle
        tokenized = tokenized.add_column("labels", tokenized["input_ids"])
        
        # Training arguments with CHECKPOINTING
        output_dir = f"./model-step{idx}"
        batch_size = 4 if not has_gpu else 8
        epochs = 3
        
        # Checkpoint her 50 adımda bir kaydet (veya dataset küçükse daha sık)
        save_steps_count = min(50, max(10, len(tokenized) // (batch_size * 4)))
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            fp16=has_gpu,
            
            # 🔥 CHECKPOINTING AYARLARI
            save_strategy="steps",           # Her X adımda kaydet
            save_steps=save_steps_count,     # Kaç adımda bir
            save_total_limit=3,              # Son 3 checkpoint'i sakla
            load_best_model_at_end=False,    # Her zaman en son modeli kullan
            
            # Progress tracking
            logging_steps=max(5, save_steps_count // 2),
            logging_dir=f'{output_dir}/logs',
            
            # Diğer ayarlar
            warmup_steps=min(100, len(tokenized) // (batch_size * 4)),
            weight_decay=0.01,
            report_to="none",
            
            # ⚡ Otomatik checkpoint yükleme
            resume_from_checkpoint=True,     # Eğer checkpoint varsa devam et
        )
        
        print(f"\n🔄 CHECKPOINTING AKTİF:")
        print(f"   💾 Checkpoint sıklığı: Her {save_steps_count} adımda bir")
        print(f"   📂 Checkpoint konumu: {output_dir}/checkpoint-XXX/")
        print(f"   ⚡ Otomatik devam etme: Aktif")
        print(f"   💡 Eğitim kesilirse kaldığı yerden devam eder!")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        
        # 🔍 Checkpoint kontrolü
        checkpoint_dir = None
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # En son checkpoint'i bul
                checkpoint_nums = [int(cp.split("-")[1]) for cp in checkpoints]
                latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
                checkpoint_dir = os.path.join(output_dir, latest_checkpoint)
                
                print(f"\n🔄 CHECKPOINT BULUNDU!")
                print(f"   📂 Konum: {checkpoint_dir}")
                print(f"   ⚡ Eğitim kaldığı yerden devam edecek...")
        
        # Eğitim
        print(f"\n🚀 Eğitim başlıyor...")
        print(f"   📊 Örnekler: {len(train_data):,}")
        print(f"   🔄 Epoch: {epochs}")
        print(f"   📦 Batch: {batch_size}")
        print(f"   ⏱️ Tahmini: {ds_info['time_cpu' if not has_gpu else 'time_gpu']}")
        
        start_time = time.time()
        
        # 🚀 Checkpoint'ten devam et veya yeni başla
        if checkpoint_dir:
            print(f"\n⚡ Checkpoint'ten devam ediliyor: {checkpoint_dir}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            print(f"\n🚀 Yeni eğitim başlıyor (checkpoint yok)...")
            trainer.train()
        
        elapsed = time.time() - start_time
        minutes = elapsed / 60
        
        print(f"\n✅ Dataset {idx} tamamlandı!")
        print(f"⏱️ Süre: {minutes:.1f} dakika ({elapsed/3600:.2f} saat)")
        
        # İstatistikleri kaydet
        stats = {
            "step": idx,
            "dataset": ds_info['name'],
            "dataset_size": len(train_data),
            "training_time_minutes": minutes,
            "epochs": epochs,
            "batch_size": batch_size,
            "total_steps": trainer.state.global_step,
            "checkpoints_saved": len([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]) if os.path.exists(output_dir) else 0,
            "resumed_from_checkpoint": checkpoint_dir is not None,
        }
        all_stats.append(stats)
        
        # Modeli kaydet
        print(f"💾 Model kaydediliyor: {output_dir}/")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        with open(f"{output_dir}/training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Kayıt tamamlandı!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Dataset {idx} eğitimi kullanıcı tarafından durduruldu!")
        break
    except Exception as e:
        print(f"\n❌ Dataset {idx} hatası: {e}")
        import traceback
        traceback.print_exc()
        continue

# Final model
if all_stats:
    print("\n" + "=" * 80)
    print("🎉 TÜM EĞİTİMLER TAMAMLANDI!")
    print("=" * 80)
    
    # Son modeli kaydet
    final_dir = "./final-trained-model"
    print(f"\n💾 Final model kaydediliyor: {final_dir}/")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Toplam istatistikler
    total_examples = sum(s['dataset_size'] for s in all_stats)
    total_minutes = sum(s['training_time_minutes'] for s in all_stats)
    total_steps = sum(s['total_steps'] for s in all_stats)
    
    summary = {
        "total_datasets": len(all_stats),
        "total_examples": total_examples,
        "total_training_time_minutes": total_minutes,
        "total_training_time_hours": total_minutes / 60,
        "total_steps": total_steps,
        "gpu_used": has_gpu,
        "gpu_name": torch.cuda.get_device_name(0) if has_gpu else "CPU",
        "datasets": all_stats
    }
    
    with open(f"{final_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Final model kaydedildi!")
    
    # Test
    print("\n" + "=" * 80)
    print("🧪 MODEL TESTİ")
    print("=" * 80)
    
    model.eval()
    
    test_prompts = [
        "How can I learn Python programming?",
        "What is artificial intelligence?",
        "Tell me about machine learning",
        "Act as a Linux terminal",
        "Explain quantum computing",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n💬 Prompt: {prompt}")
        print(f"🤖 Cevap: {response[:200]}...")
    
    # Özet rapor
    print("\n" + "=" * 80)
    print("📊 EĞİTİM ÖZETİ")
    print("=" * 80)
    
    print(f"\n✅ Tamamlanan dataset'ler: {len(all_stats)}")
    print(f"📊 Toplam örnek: {total_examples:,}")
    print(f"⏱️ Toplam süre: {total_minutes:.1f} dakika ({total_minutes/60:.2f} saat)")
    print(f"🔄 Toplam adım: {total_steps:,}")
    print(f"💾 Final model: {final_dir}/")
    
    print("\nDataset bazında:")
    for stat in all_stats:
        print(f"  {stat['step']}. {stat['dataset']}")
        print(f"     Örnekler: {stat['dataset_size']:,}")
        print(f"     Süre: {stat['training_time_minutes']:.1f} dk")
        print(f"     Adımlar: {stat['total_steps']:,}")
    
    print("\n" + "=" * 80)
    print("✅ BAŞARILI! Model kullanıma hazır.")
    print("=" * 80)
    print(f"\n📁 Model konumu: {final_dir}/")
    print(f"📊 İstatistikler: {final_dir}/training_summary.json")
    print(f"\n💡 Kullanım:")
    print(f"   from transformers import GPT2LMHeadModel, GPT2Tokenizer")
    print(f"   model = GPT2LMHeadModel.from_pretrained('{final_dir}')")
    print(f"   tokenizer = GPT2Tokenizer.from_pretrained('{final_dir}')")

else:
    print("\n❌ Hiçbir eğitim tamamlanamadı.")
