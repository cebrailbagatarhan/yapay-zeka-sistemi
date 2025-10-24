"""
Model İyileştirme Scriptleri
Modeli daha iyi hale getirmek için kullan
"""

import os

# 1. DAHA FAZLA TÜRKÇE VERİ EKLE
def add_turkish_datasets():
    """Türkçe dataset'leri ekle"""
    print("🇹🇷 Türkçe Dataset'ler Ekleniyor...")
    
    turkish_datasets = [
        "ytu-ce-cosmos/turkish-instructions",
        "turkish-nlp/turkish-qa-dataset",
        # Daha fazla eklenebilir
    ]
    
    print(f"✅ {len(turkish_datasets)} Türkçe dataset hazır!")
    return turkish_datasets


# 2. DAHA UZUN EĞİTİM YAP
def train_longer(model_name, num_epochs=10):
    """Daha uzun eğitim - 10 epoch"""
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir="./improved-model",
        num_train_epochs=num_epochs,  # 10 epoch
        per_device_train_batch_size=4,
        learning_rate=2e-5,  # Daha düşük
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
    )
    
    print(f"🔄 {num_epochs} epoch eğitim başlıyor...")
    return training_args


# 3. PROMPT FORMATI İYİLEŞTİR
def format_prompt_alpaca(instruction, input_text=""):
    """Alpaca formatında prompt oluştur (en iyi format)"""
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    return prompt


# 4. DATA AUGMENTATION
def augment_data(text):
    """Veriyi çoğalt - paraphrase yap"""
    variations = []
    
    # Basit varyasyonlar
    if "nasıl" in text.lower():
        variations.append(text.replace("nasıl", "ne şekilde"))
        variations.append(text.replace("nasıl", "hangi yöntemle"))
    
    if "nedir" in text.lower():
        variations.append(text.replace("nedir", "ne demek"))
        variations.append(text.replace("nedir", "tanımı nedir"))
    
    return [text] + variations


# 5. ÇOK AŞAMALI EĞİTİM
def multi_stage_training():
    """Aşamalı eğitim planı"""
    stages = {
        "stage1_general": {
            "description": "Genel dil öğrenme",
            "datasets": ["wikipedia", "books"],
            "epochs": 3
        },
        "stage2_instruction": {
            "description": "Instruction-following",
            "datasets": ["alpaca", "dolly"],
            "epochs": 5
        },
        "stage3_conversation": {
            "description": "Konuşma yetenekleri",
            "datasets": ["guanaco", "oasst"],
            "epochs": 5
        },
        "stage4_domain": {
            "description": "Özel alan uzmanlaşması",
            "datasets": ["code", "math", "turkish"],
            "epochs": 10
        }
    }
    
    print("🎯 ÇOK AŞAMALI EĞİTİM PLANI:")
    for stage, config in stages.items():
        print(f"\n{stage}:")
        print(f"  • {config['description']}")
        print(f"  • Datasets: {', '.join(config['datasets'])}")
        print(f"  • Epochs: {config['epochs']}")
    
    return stages


# 6. QWEN MODELİNİ KULLAN
def use_qwen_model():
    """Qwen modelini yükle ve kullan"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("🚀 Qwen2.5-1.5B Model Yükleniyor...")
    
    model_path = "./qwen-model"
    
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        print("⚠️ Yerel model bulunamadı, Hugging Face'den indirilecek...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print("✅ Qwen model hazır!")
    return model, tokenizer


# 7. KALİTE ARTIRMA ÖNERİLERİ
def show_improvement_suggestions():
    """Model iyileştirme önerileri göster"""
    print("\n" + "="*60)
    print("🎯 MODEL İYİLEŞTİRME ÖNERİLERİ")
    print("="*60)
    
    suggestions = [
        {
            "title": "1. Daha Fazla Veri",
            "impact": "⭐⭐⭐⭐⭐",
            "effort": "Orta",
            "details": "10K → 100K örnek = 5x daha iyi"
        },
        {
            "title": "2. Daha Büyük Model",
            "impact": "⭐⭐⭐⭐⭐",
            "effort": "Kolay",
            "details": "GPT-2 → Qwen = 10x daha iyi"
        },
        {
            "title": "3. Daha Uzun Eğitim",
            "impact": "⭐⭐⭐⭐",
            "effort": "Kolay",
            "details": "3 → 10 epoch = 2x daha iyi"
        },
        {
            "title": "4. Türkçe Veri",
            "impact": "⭐⭐⭐⭐",
            "effort": "Orta",
            "details": "50K Türkçe örnek ekle"
        },
        {
            "title": "5. Data Augmentation",
            "impact": "⭐⭐⭐",
            "effort": "Orta",
            "details": "Her örneği 3-5 varyasyonla çoğalt"
        }
    ]
    
    for i, sug in enumerate(suggestions, 1):
        print(f"\n{sug['title']}")
        print(f"  Etki: {sug['impact']}")
        print(f"  Çaba: {sug['effort']}")
        print(f"  💡 {sug['details']}")
    
    print("\n" + "="*60)
    print("🏆 EN ÖNEMLİ 3: Büyük Model + Fazla Veri + Uzun Eğitim")
    print("="*60)


# MAIN
if __name__ == "__main__":
    print("\n🤖 MODEL İYİLEŞTİRME ARACI")
    print("="*60)
    
    print("\n📋 MENÜ:")
    print("1. İyileştirme önerilerini göster")
    print("2. Türkçe dataset'leri listele")
    print("3. Çok aşamalı eğitim planı")
    print("4. Qwen modelini yükle")
    print("5. Prompt formatı örnekleri")
    
    choice = input("\nSeçim (1-5): ").strip()
    
    if choice == "1":
        show_improvement_suggestions()
    
    elif choice == "2":
        datasets = add_turkish_datasets()
        print(f"\n📚 {len(datasets)} Türkçe dataset:")
        for ds in datasets:
            print(f"  • {ds}")
    
    elif choice == "3":
        stages = multi_stage_training()
        print(f"\n✅ {len(stages)} aşamalı eğitim planı hazır!")
    
    elif choice == "4":
        try:
            model, tokenizer = use_qwen_model()
            print("\n✅ Qwen model kullanıma hazır!")
        except Exception as e:
            print(f"\n❌ Hata: {e}")
    
    elif choice == "5":
        print("\n📝 PROMPT FORMAT ÖRNEKLERİ:")
        print("\n1. Alpaca Format (Önerilen):")
        print(format_prompt_alpaca("Python'da liste nasıl oluşturulur?"))
        
        print("\n2. Veri Augmentation:")
        original = "Yapay zeka nedir?"
        variations = augment_data(original)
        print(f"Orijinal: {original}")
        print(f"Varyasyonlar: {variations}")
    
    else:
        print("❌ Geçersiz seçim!")
    
    print("\n" + "="*60)
    print("✅ İşlem tamamlandı!")
