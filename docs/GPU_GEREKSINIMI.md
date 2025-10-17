# ⚠️ GPU Gereksinimi - Eğitim Durumu

## 📊 Durum Raporu

### ✅ Yapılan İşlemler
1. ✅ **Dataset indirildi:** MATH-openai-split (12,500 örnek)
2. ✅ **Model hazırlandı:** GPT-2 Small (124M parametreler)
3. ✅ **Eğitim script'i oluşturuldu:** `train_math_model.py`
4. ✅ **GitHub'a yüklendi:** Tüm dosyalar commit edildi

### ⚠️ Önemli Not: GPU Gereksinimi

Eğitim başlatıldı ama **GPU bulunamadı**. CPU ile eğitim çok yavaş olduğu için durduruldu.

```
⚠️ GPU bulunamadı! CPU ile eğitim yapılacak (çok yavaş olabilir)
⏱️ CPU ile tahmini süre: 6-12 SAAT (çok yavaş!)
⏱️ GPU ile tahmini süre: 25-40 dakika (önerilen)
```

---

## 🎯 Eğitimi Tamamlamak İçin Seçenekler

### Seçenek 1: GPU'lu Bilgisayarda Çalıştırma (ÖNERİLEN)
```bash
# GPU'lu bir bilgisayarda:
git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
cd yapay-zeka-sistemi
pip install transformers datasets torch accelerate
python train_math_model.py

# GPU kontrolü:
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**Gereksinimler:**
- NVIDIA GPU (RTX 2060 veya üzeri)
- 6-8 GB VRAM
- CUDA toolkit

---

### Seçenek 2: Google Colab (ÜCRETSİZ GPU!)

**Adımlar:**
1. [Google Colab](https://colab.research.google.com/)'a git
2. Yeni notebook oluştur
3. **Runtime → Change runtime type → GPU** seç
4. Bu kodu çalıştır:

```python
# Colab'da GPU ile eğitim
!git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
%cd yapay-zeka-sistemi

!pip install transformers datasets torch accelerate -q

# GPU kontrolü
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"GPU Adı: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yok'}")

# Eğitimi başlat
!python train_math_model.py
```

**Avantajlar:**
- ✅ Ücretsiz GPU (Tesla T4)
- ✅ Kurulum gerekmez
- ✅ 25-40 dakikada tamamlanır
- ✅ Model indirilebilir

---

### Seçenek 3: Kaggle (ÜCRETSİZ GPU!)

**Adımlar:**
1. [Kaggle](https://www.kaggle.com/) hesabı aç
2. **Code → New Notebook** oluştur
3. **Settings → Accelerator → GPU** seç
4. Bu kodu çalıştır:

```python
# Kaggle'da GPU ile eğitim
!git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
%cd yapay-zeka-sistemi

!pip install transformers datasets accelerate -q

# Eğitimi başlat
!python train_math_model.py
```

**Avantajlar:**
- ✅ Ücretsiz GPU (Tesla P100)
- ✅ 30 saat/hafta GPU kullanımı
- ✅ Dataset storage
- ✅ Public/Private notebook

---

### Seçenek 4: Hugging Face Spaces (Cloud)

```python
# Hugging Face'de deploy et
# Spaces'de GPU kiralayarak eğit
# https://huggingface.co/spaces
```

---

### Seçenek 5: CPU ile Eğitim (Sabırlı olanlar için)

Eğer GPU yoksa ve sabırlıysanız, CPU ile de çalıştırabilirsiniz:

```bash
# CPU ile eğitim (6-12 saat sürer!)
python train_math_model.py
# "e" basarak devam edin
```

**Not:** 
- ⏱️ 6-12 saat sürecek
- 🔥 Laptop ısınabilir
- 🔋 Elektriğe takılı bırakın
- ⚠️ Bilgisayarı kapatmayın

---

## 📊 Performans Karşılaştırması

| Platform | GPU | VRAM | Ücretsiz | Süre | Önerilen |
|----------|-----|------|---------|------|----------|
| **Google Colab** | Tesla T4 | 15 GB | ✅ Evet | 25-40 dk | ⭐⭐⭐⭐⭐ |
| **Kaggle** | Tesla P100 | 16 GB | ✅ Evet | 30-45 dk | ⭐⭐⭐⭐⭐ |
| **Kendi GPU** | RTX 3060+ | 8+ GB | ❌ Hayır | 25-40 dk | ⭐⭐⭐⭐ |
| **AWS/GCP** | Various | 8+ GB | ❌ Hayır | 20-35 dk | ⭐⭐⭐ |
| **CPU** | - | RAM | ✅ Evet | 6-12 saat | ⭐ |

---

## 🚀 ÖNERİLEN: Google Colab ile Eğitim

En kolay ve ücretsiz yöntem Google Colab:

### Hızlı Başlangıç (5 dakika)

1. **Colab'ı aç:** https://colab.research.google.com/
2. **Yeni notebook oluştur**
3. **GPU'yu aktifleştir:** Runtime → Change runtime type → GPU → Save
4. **Bu kodu yapıştır ve çalıştır:**

```python
# ============================================
# MATH-openai-split Dataset ile Model Eğitimi
# ============================================

# 1. Projeyi klonla
!git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
%cd yapay-zeka-sistemi

# 2. Paketleri yükle
!pip install transformers datasets torch accelerate -q

# 3. GPU kontrolü
import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Yok"
print(f"✅ GPU: {gpu_name}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 4. Eğitimi otomatik başlat (kullanıcı onayı olmadan)
import subprocess
import sys

# Script'i otomatik onay ile çalıştır
proc = subprocess.Popen(
    [sys.executable, "train_math_model.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Otomatik olarak "e" (evet) yanıtı ver
proc.stdin.write("e\n")
proc.stdin.write("e\n")
proc.stdin.flush()

# Çıktıları göster
for line in proc.stdout:
    print(line, end='')

proc.wait()
print(f"\n✅ Eğitim tamamlandı! Exit code: {proc.returncode}")

# 5. Model dosyalarını listele
!ls -lh math-gpt2-model/

# 6. Eğitim istatistiklerini göster
import json
with open("math-gpt2-model/training_stats.json") as f:
    stats = json.load(f)
    print("\n📊 Eğitim İstatistikleri:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

# 7. Modeli test et
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./math-gpt2-model")
tokenizer = GPT2Tokenizer.from_pretrained("./math-gpt2-model")

test_problem = "What is 15 + 27?"
prompt = f"Problem: {test_problem}\n\nSolution:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs["input_ids"],
    max_length=200,
    temperature=0.7,
    top_p=0.9
)

print(f"\n🧪 Test:")
print(f"Soru: {test_problem}")
print(f"Cevap: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# 8. Modeli indir (optional)
from google.colab import files
!zip -r math-gpt2-model.zip math-gpt2-model/
# files.download("math-gpt2-model.zip")  # İndirmek için yorum satırını kaldır
```

---

## 📦 Eğitilmiş Modeli İndirme

Eğitim tamamlandıktan sonra:

### Google Colab'dan
```python
from google.colab import files
!zip -r math-gpt2-model.zip math-gpt2-model/
files.download("math-gpt2-model.zip")
```

### Kaggle'dan
```python
# Output klasörüne kaydedilir, otomatik indirilir
!cp -r math-gpt2-model /kaggle/working/
```

---

## 🎯 Sonuç

### Yapılması Gerekenler:
1. ✅ **Colab/Kaggle'da GPU ile eğit** (25-40 dakika)
2. ✅ **Modeli indir**
3. ✅ **Kendi projende kullan**

### Alternatif:
- ⏳ CPU ile eğit (6-12 saat, sabır gerekir)

---

## 💡 İpuçları

### GPU Kontrolü
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### Colab GPU Limiti
- Free: 12-15 saat/gün
- Colab Pro: 24 saat/gün, daha hızlı GPU

### Kaggle GPU Limiti
- Free: 30 saat/hafta
- 9 saat maksimum session

---

## 📚 Kaynaklar

- **Google Colab:** https://colab.research.google.com/
- **Kaggle:** https://www.kaggle.com/
- **Hugging Face:** https://huggingface.co/
- **GitHub Repo:** https://github.com/cebrailbagatarhan/yapay-zeka-sistemi

---

**Tavsiye:** Google Colab kullanın, ücretsiz ve kolay! 🚀

**Son Güncelleme:** 17 Ekim 2025
