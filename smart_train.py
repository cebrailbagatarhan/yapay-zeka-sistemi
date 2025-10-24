"""
Akıllı Dataset Seçimi ve Hızlı Eğitim
Küçük ama kaliteli dataset'lerle hızlı sonuç
"""

import os
import time
import json
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

print("=" * 70)
print("🎯 AKILLI DATASET SEÇİMİ VE HIZLI EĞİTİM")
print("=" * 70)

# GPU kontrolü
has_gpu = torch.cuda.is_available()
device = "cuda" if has_gpu else "cpu"
print(f"\n{'✅' if has_gpu else '⚠️'} GPU: {torch.cuda.get_device_name(0) if has_gpu else 'CPU (yavaş!)'}")

# En iyi dataset'ler (küçük ama kaliteli)
DATASETS = {
    "1": {
        "name": "databricks/databricks-dolly-15k",
        "size": "15K",
        "desc": "Yüksek kaliteli instruction dataset (İngilizce)",
        "time_cpu": "2-3 saat",
        "time_gpu": "15-20 dakika",
        "quality": "⭐⭐⭐⭐⭐"
    },
    "2": {
        "name": "Open-Orca/SlimOrca",
        "size": "518K",
        "desc": "GPT-4 kalitesinde sohbet dataset'i",
        "time_cpu": "20-30 saat",
        "time_gpu": "2-3 saat",
        "quality": "⭐⭐⭐⭐⭐"
    },
    "3": {
        "name": "timdettmers/openassistant-guanaco",
        "size": "10K",
        "desc": "Çok iyi sohbet örnekleri",
        "time_cpu": "1-2 saat",
        "time_gpu": "10-15 dakika",
        "quality": "⭐⭐⭐⭐"
    },
    "4": {
        "name": "fka/awesome-chatgpt-prompts",
        "size": "203",
        "desc": "ChatGPT prompt'ları (ÇOK HIZLI!)",
        "time_cpu": "5-10 dakika",
        "time_gpu": "2-3 dakika",
        "quality": "⭐⭐⭐⭐"
    },
    "5": {
        "name": "philschmid/dolly-15k-oai-style",
        "size": "15K",
        "desc": "Dolly dataset OpenAI formatında",
        "time_cpu": "2-3 saat",
        "time_gpu": "15-20 dakika",
        "quality": "⭐⭐⭐⭐⭐"
    }
}

print("\n" + "=" * 70)
print("📊 MÜKEMMELİYET PLANLI DATASET'LER")
print("=" * 70)

for key, ds in DATASETS.items():
    print(f"\n{key}. {ds['name']}")
    print(f"   Boyut: {ds['size']} örnek")
    print(f"   Kalite: {ds['quality']}")
    print(f"   Açıklama: {ds['desc']}")
    print(f"   Süre (CPU): {ds['time_cpu']}")
    print(f"   Süre (GPU): {ds['time_gpu']}")

# Otomatik seçim: CPU için en hızlısı
if not has_gpu:
    print("\n" + "=" * 70)
    print("⚠️ GPU BULUNAMADI - OTOMATIK ÖNERİ")
    print("=" * 70)
    print("\n💡 CPU için en hızlı ve kaliteli seçenek:")
    print("   → Dataset #4: awesome-chatgpt-prompts (203 örnek)")
    print("   → Süre: Sadece 5-10 dakika!")
    print("   → Kalite: ⭐⭐⭐⭐ (Çok iyi prompt'lar)")
    
    choice = input("\n🚀 Bu dataset ile devam edilsin mi? (e/h) veya başka numara girin: ").strip().lower()
    
    if choice == 'h':
        print("❌ Eğitim iptal edildi.")
        exit()
    elif choice == 'e':
        selected = "4"
    else:
        selected = choice if choice in DATASETS else "4"
else:
    print("\n" + "=" * 70)
    print("✅ GPU VAR - ÖNERİLEN SEÇENEKLER")
    print("=" * 70)
    print("\n💡 GPU ile önerilen:")
    print("   → Dataset #1 veya #5: Dolly (15K) - 15-20 dakika")
    print("   → Dataset #3: Guanaco (10K) - 10-15 dakika")
    print("   → Dataset #4: ChatGPT Prompts (203) - 2-3 dakika (test için)")
    
    choice = input("\n🚀 Hangi dataset'i seçmek istersiniz? (1-5): ").strip()
    selected = choice if choice in DATASETS else "1"

# Dataset bilgisi
selected_ds = DATASETS[selected]
print(f"\n✅ Seçilen: {selected_ds['name']}")
print(f"📊 Boyut: {selected_ds['size']}")
print(f"⏱️ Tahmini süre: {selected_ds['time_cpu' if not has_gpu else 'time_gpu']}")

# Dataset'i indir
print(f"\n📥 Dataset indiriliyor: {selected_ds['name']}...")
try:
    dataset = load_dataset(selected_ds['name'])
    print(f"✅ Dataset yüklendi!")
    
    # Dataset yapısını kontrol et
    if 'train' in dataset:
        train_data = dataset['train']
    else:
        # İlk split'i al
        train_data = dataset[list(dataset.keys())[0]]
    
    print(f"📊 Toplam örnek: {len(train_data)}")
    print(f"\n📝 İlk örnek:")
    print(train_data[0])
    
except Exception as e:
    print(f"❌ Hata: {e}")
    print("\n💡 Alternatif: Kendi dataset'imizi kullanacağız...")
    
    # Fallback: Kendi hazırladığımız dataset
    print("\n📦 Yerel dataset yükleniyor...")
    with open("data/training/conversational_dataset.json", 'r', encoding='utf-8') as f:
        local_data = json.load(f)
    
    train_data = Dataset.from_dict({
        "text": [f"Question: {item['question']}\nAnswer: {item['response']}" 
                 for item in local_data]
    })
    
    print(f"✅ Yerel dataset yüklendi: {len(train_data)} örnek")

# Model hazırla
print(f"\n🤖 Model hazırlanıyor (GPT-2)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

print(f"✅ Model hazır: {model.num_parameters():,} parametreler")
print(f"📍 Device: {device}")

# Tokenize fonksiyonu
def tokenize_function(examples):
    # Dataset yapısına göre text alanını bul
    if 'text' in examples:
        texts = examples['text']
    elif 'instruction' in examples and 'response' in examples:
        texts = [f"Question: {q}\nAnswer: {a}" 
                for q, a in zip(examples['instruction'], examples['response'])]
    elif 'prompt' in examples:
        texts = examples['prompt']
    elif 'act' in examples and 'prompt' in examples:
        texts = [f"{act}: {prompt}" 
                for act, prompt in zip(examples['act'], examples['prompt'])]
    else:
        # En fazla bilgi içeren alanı kullan
        text_keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
        if text_keys:
            texts = examples[text_keys[0]]
        else:
            texts = [str(examples)]
    
    return tokenizer(texts, truncation=True, max_length=256, padding="max_length")

print(f"\n📝 Dataset tokenize ediliyor...")
tokenized_dataset = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=train_data.column_names
)

# Labels ekle (input_ids'i kopyala)
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

# Training arguments
batch_size = 4 if not has_gpu else 8
epochs = 3

training_args = TrainingArguments(
    output_dir="./trained-model",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    fp16=has_gpu,  # Mixed precision sadece GPU'da
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Eğitimi başlat
print("\n" + "=" * 70)
print("🚀 EĞİTİM BAŞLIYOR")
print("=" * 70)
print(f"📊 Örnekler: {len(train_data)}")
print(f"🔄 Epoch: {epochs}")
print(f"📦 Batch size: {batch_size}")
print(f"⏱️ Tahmini süre: {selected_ds['time_cpu' if not has_gpu else 'time_gpu']}")
print("=" * 70)

start_time = time.time()

try:
    trainer.train()
    
    # Süreyi hesapla
    elapsed = time.time() - start_time
    minutes = elapsed / 60
    
    print("\n" + "=" * 70)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"⏱️ Süre: {minutes:.1f} dakika ({elapsed/3600:.2f} saat)")
    print(f"📁 Model kaydediliyor...")
    
    # Modeli kaydet
    trainer.save_model("./trained-model")
    tokenizer.save_pretrained("./trained-model")
    
    # İstatistikleri kaydet
    stats = {
        "dataset": selected_ds['name'],
        "dataset_size": len(train_data),
        "training_time_minutes": minutes,
        "training_time_hours": elapsed / 3600,
        "epochs": epochs,
        "batch_size": batch_size,
        "gpu_used": has_gpu,
        "gpu_name": torch.cuda.get_device_name(0) if has_gpu else "CPU",
        "model": "gpt2",
        "total_steps": trainer.state.global_step,
    }
    
    with open("./trained-model/training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Model kaydedildi: ./trained-model/")
    
    # Test
    print("\n" + "=" * 70)
    print("🧪 MODEL TESTİ")
    print("=" * 70)
    
    model.eval()
    
    test_prompts = [
        "How can I learn Python?",
        "What is AI?",
        "Tell me a joke",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n💬 {prompt}")
        print(f"🤖 {response[:200]}...")
    
    print("\n" + "=" * 70)
    print("✅ TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 70)
    print(f"\n📊 Eğitim Özeti:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Eğitim kullanıcı tarafından durduruldu!")
except Exception as e:
    print(f"\n\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
