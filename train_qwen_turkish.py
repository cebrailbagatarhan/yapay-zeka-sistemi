"""
Qwen2.5-1.5B Türkçe Fine-Tuning
Daha iyi Türkçe performans için eğitim
"""

print("🇹🇷 QWEN TÜRKÇE FINE-TUNING")
print("="*60)

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
import torch
import os

# 1. MODEL YÜKLE
print("\n1️⃣ Qwen modeli yükleniyor...")
model_path = "./qwen-model"

if not os.path.exists(model_path):
    print("⚠️ Model bulunamadı, Hugging Face'den indirilecek...")
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

# Pad token ayarla
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

print("✅ Model hazır!")

# 2. TÜRKÇE DATASET YÜKLE
print("\n2️⃣ Türkçe dataset'ler yükleniyor...")

datasets_to_load = []

try:
    # ChatGPT prompts (Türkçe çevirili)
    print("   📥 ChatGPT prompts...")
    ds1 = load_dataset("fka/awesome-chatgpt-prompts", split="train[:500]")
    datasets_to_load.append(ds1)
    print(f"   ✅ {len(ds1)} örnek")
except Exception as e:
    print(f"   ⚠️ ChatGPT prompts yüklenemedi: {e}")

try:
    # OpenAssistant Guanaco (çok dilli)
    print("   📥 Guanaco conversations...")
    ds2 = load_dataset("timdettmers/openassistant-guanaco", split="train[:5000]")
    datasets_to_load.append(ds2)
    print(f"   ✅ {len(ds2)} örnek")
except Exception as e:
    print(f"   ⚠️ Guanaco yüklenemedi: {e}")

try:
    # Alpaca instruction dataset
    print("   📥 Alpaca instructions...")
    ds3 = load_dataset("yahma/alpaca-cleaned", split="train[:3000]")
    datasets_to_load.append(ds3)
    print(f"   ✅ {len(ds3)} örnek")
except Exception as e:
    print(f"   ⚠️ Alpaca yüklenemedi: {e}")

if not datasets_to_load:
    print("❌ Hiç dataset yüklenemedi!")
    exit(1)

print(f"\n✅ Toplam {len(datasets_to_load)} dataset yüklendi")

# 3. VERİYİ HAZIRLAMA
print("\n3️⃣ Veriler hazırlanıyor...")

def format_prompt(example):
    """Alpaca formatında prompt oluştur"""
    if "act" in example:  # ChatGPT prompts
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Act as {example['act']}. {example['prompt']}

### Response:
"""
    elif "text" in example:  # Guanaco
        text = example["text"]
    elif "instruction" in example:  # Alpaca
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            text = f"""Below is an instruction that describes a task, paired with an input. Write a response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            text = f"""Below is an instruction that describes a task. Write a response.

### Instruction:
{instruction}

### Response:
{output}"""
    else:
        text = str(example)
    
    return {"text": text}

# Her dataset'i formatla
formatted_datasets = []
for ds in datasets_to_load:
    try:
        formatted = ds.map(format_prompt, remove_columns=ds.column_names)
        formatted_datasets.append(formatted)
    except Exception as e:
        print(f"⚠️ Dataset formatlama hatası: {e}")

if not formatted_datasets:
    print("❌ Hiç dataset formatlanamadı!")
    exit(1)

# Dataset'leri birleştir
train_dataset = concatenate_datasets(formatted_datasets)
print(f"✅ {len(train_dataset)} örnek hazır!")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("\n4️⃣ Tokenization yapılıyor...")
tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
print(f"✅ {len(tokenized_dataset)} örnek tokenize edildi")

# 4. EĞİTİM AYARLARI
print("\n5️⃣ Eğitim ayarları yapılıyor...")

output_dir = "./qwen-turkish-model"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,  # 5 epoch (orta seviye)
    per_device_train_batch_size=2,  # Küçük batch (CPU için)
    gradient_accumulation_steps=4,  # Etkili batch_size=8
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    fp16=False,  # CPU için False
    report_to="none",
)

print(f"📊 Eğitim Parametreleri:")
print(f"   • Epoch: {training_args.num_train_epochs}")
print(f"   • Batch size: {training_args.per_device_train_batch_size}")
print(f"   • Learning rate: {training_args.learning_rate}")
print(f"   • Örnek sayısı: {len(tokenized_dataset)}")

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

# 5. EĞİTİM BAŞLAT
print("\n" + "🚀"*30)
print("EĞİTİM BAŞLIYOR!")
print("🚀"*30)
print("\n⏱️ Tahmini Süre:")
print(f"   • {len(tokenized_dataset)} örnek")
print(f"   • {training_args.num_train_epochs} epoch")
print(f"   • Yaklaşık: 2-4 saat (CPU)")
print("\n💡 İpuçları:")
print("   • Checkpoint'ler her 200 adımda kaydedilecek")
print("   • Ctrl+C ile duraklatabilirsin")
print("   • Tekrar çalıştırınca devam eder")
print("\n" + "="*60)

confirm = input("\n✅ Eğitime başlamak istediğinize emin misiniz? (e/h): ").strip().lower()

if confirm != 'e':
    print("❌ İptal edildi.")
    exit(0)

print("\n🔥 EĞİTİM BAŞLIYOR...")
print("="*60)

try:
    # Eğitim başlat
    trainer.train()
    
    print("\n" + "🎉"*30)
    print("✅ EĞİTİM TAMAMLANDI!")
    print("🎉"*30)
    
    # Modeli kaydet
    print("\n💾 Model kaydediliyor...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model kaydedildi: {output_dir}")
    
    # Test
    print("\n🧪 HIZLI TEST:")
    print("="*60)
    
    test_prompts = [
        "Python'da döngü nasıl yazılır?",
        "Türkiye'nin başkenti neresidir?",
        "Bana bir şiir yaz",
    ]
    
    model.eval()
    for prompt in test_prompts:
        formatted = f"""Below is an instruction. Write a response.

### Instruction:
{prompt}

### Response:
"""
        inputs = tokenizer(formatted, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()[:200]
        
        print(f"\n👤 Sen: {prompt}")
        print(f"🤖 Qwen-TR: {response}")
        print("-"*60)
    
    print("\n✅ Test tamamlandı!")
    print(f"📂 Eğitilmiş model: {output_dir}")
    print("\n💡 Şimdi bu modeli main.py'de kullanabilirsin!")

except KeyboardInterrupt:
    print("\n\n⚠️ Eğitim durduruldu!")
    print(f"💾 Son checkpoint: {output_dir}")
    print("💡 Tekrar çalıştırırsan devam eder!")

except Exception as e:
    print(f"\n❌ Eğitim hatası: {e}")
    import traceback
    traceback.print_exc()
