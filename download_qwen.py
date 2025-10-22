"""
Qwen2.5-1.5B-Instruct Model İndirici ve Test
En iyi CPU + 16GB RAM modeli!
"""

print("🚀 QWEN2.5-1.5B-INSTRUCT MODEL İNDİRİCİ")
print("="*60)
print("📦 Model: Qwen/Qwen2.5-1.5B-Instruct")
print("💾 Boyut: ~3GB")
print("🎯 Özellikler: Türkçe + İngilizce, Mükemmel instruction-following")
print("="*60)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("\n📥 Model indiriliyor... (İlk seferde 3GB indirilecek)")
    print("⏳ Lütfen bekleyin, bu 2-5 dakika sürebilir...")
    
    # Tokenizer indir
    print("\n1️⃣ Tokenizer indiriliyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("   ✅ Tokenizer hazır!")
    
    # Model indir
    print("\n2️⃣ Model indiriliyor (3GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU için float32
        device_map="cpu",
        trust_remote_code=True
    )
    print("   ✅ Model hazır!")
    
    # Model kaydet (yerel kullanım için)
    save_path = "./qwen-model"
    print(f"\n3️⃣ Model kaydediliyor: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("   ✅ Model kaydedildi!")
    
    print("\n" + "🎉"*30)
    print("✅ QWEN MODEL BAŞARIYLA İNDİRİLDİ!")
    print("🎉"*30)
    
    # Hızlı test
    print("\n🧪 HIZLI TEST BAŞLIYOR...")
    print("="*60)
    
    test_prompts = [
        "Merhaba! Nasılsın?",
        "Python'da liste nasıl oluşturulur?",
        "5 + 3 kaç eder?",
        "Yapay zeka nedir?",
        "Bana bir şaka anlat."
    ]
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Test:")
        print(f"👤 Sen: {prompt}")
        
        # Qwen formatı
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer([text], return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        # Clean
        response = response[:200].strip()
        
        print(f"🤖 Qwen: {response}")
        print("-"*60)
    
    print("\n✅ TEST TAMAMLANDI!")
    print("\n💡 Şimdi main.py'ye entegre edebiliriz!")
    print(f"📂 Model konumu: {save_path}")
    
except ImportError as e:
    print(f"\n❌ Kütüphane hatası: {e}")
    print("\n💡 Şu komutu çalıştır:")
    print("   pip install transformers torch accelerate")
except Exception as e:
    print(f"\n❌ Hata: {e}")
    import traceback
    traceback.print_exc()
