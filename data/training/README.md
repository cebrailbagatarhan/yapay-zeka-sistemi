# 📚 Eğitim Veri Setleri (Training Datasets)

AI modeli eğitimi ve chat/reasoning modları için hazırlanmış kapsamlı veri setleri koleksiyonu.

## 📁 Dataset'ler

### 1. **reasoning_chat_dataset.json** (10 örnekler)
Teknik açıklamalar ve derin reasoning örnekleri

**İçerik:**
- ✅ Machine Learning vs Deep Learning
- ✅ REST API prensipleri
- ✅ Git merge vs rebase
- ✅ Docker vs Virtual Machine
- ✅ SQL Injection güvenliği
- ✅ Asenkron programlama
- ✅ JWT authentication
- ✅ Microservices mimarisi
- ✅ E-ticaret için veritabanı seçimi
- ✅ Sistem tasarımı kararları

**Format:**
```json
{
  "question": "Soru metni",
  "reasoning": "Adım adım düşünme süreci",
  "answer": "Kısa özet cevap",
  "type": "technical_explanation",
  "category": "AI/ML",
  "difficulty": "intermediate"
}
```

**Kullanım:** Derin reasoning ve teknik açıklamalar için model eğitimi

---

### 2. **conversational_dataset.json** (10 konuşma)
Doğal sohbet ve chat bot eğitimi

**İçerik:**
- 👋 Selamlama ve bilgilendirme
- 📚 Öğrenme tavsiyeleri (Python, ML)
- 🐛 Debug ve hata çözme
- 💡 Motivasyon ve cesaret
- 🔍 Kavram açıklamaları (API, Git/GitHub)
- 🎨 Proje önerileri (Portfolio)
- 📖 Basitleştirme (Regex)
- 💪 Kariyer tavsiyesi (Mülakat hazırlığı)

**Format:**
```json
{
  "question": "Kullanıcı sorusu",
  "response": "AI cevabı (emoji ve formatlamayla)",
  "type": "greeting_info",
  "category": "General",
  "tone": "friendly"
}
```

**Özellikler:**
- Emoji kullanımı 🎉
- Markdown formatlama
- Farklı tonlar (friendly, encouraging, professional)
- Adım adım açıklamalar
- Pratik örnekler

**Kullanım:** Chat bot personality ve doğal konuşma eğitimi

---

### 3. **code_examples_dataset.json** (6 kod örneği)
Kod örnekleri ve implementasyonlar

**İçerik:**
- ✅ Basit fonksiyonlar (add_numbers)
- ✅ List filtreleme (çift sayılar)
- ✅ Fibonacci (3 farklı yöntem)
- ✅ Binary search (iterative + recursive)
- ✅ Palindrome kontrolü
- ✅ REST API (Flask + JWT)

**Format:**
```json
{
  "problem": "Problem tanımı",
  "solution": "Kod çözümü (markdown code block)",
  "explanation": "Açıklama ve karmaşıklık analizi",
  "difficulty": "beginner|intermediate|advanced",
  "category": "Algorithms",
  "tags": ["recursion", "dynamic-programming"]
}
```

**Özellikler:**
- Çoklu çözüm yöntemleri
- Time/Space complexity analizi
- Docstring ve comment'ler
- Test case'ler
- Best practices

**Kullanım:** Kod üretimi ve problem çözme eğitimi

---

## 📊 İstatistikler

| Dataset | Örnekler | Kategoriler | Zorluk Seviyeleri |
|---------|----------|-------------|-------------------|
| Reasoning & Chat | 10 | 6 (AI/ML, Security, DevOps, vb.) | Beginner → Advanced |
| Conversational | 10 | 5 (General, Programming, Career, vb.) | N/A |
| Code Examples | 6 | 4 (Functions, Algorithms, Strings, Web) | Beginner → Advanced |
| **TOPLAM** | **26** | **15+** | **3 seviye** |

---

## 🎯 Kullanım Örnekleri

### Python'da Dataset Yükleme

```python
import json

# Reasoning dataset
with open('data/training/reasoning_chat_dataset.json', 'r', encoding='utf-8') as f:
    reasoning_data = json.load(f)

# Her bir örneği işle
for example in reasoning_data:
    question = example['question']
    reasoning = example['reasoning']
    answer = example['answer']
    
    print(f"Q: {question}")
    print(f"R: {reasoning}")
    print(f"A: {answer}\n")
```

### Model Eğitimi (Basit Örnek)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Dataset hazırla
def prepare_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Her örneği "Question: ... Answer: ..." formatına çevir
    texts = []
    for example in data:
        if 'question' in example and 'answer' in example:
            text = f"Question: {example['question']}\nAnswer: {example['answer']}"
            texts.append(text)
    
    return texts

# Eğitim
texts = prepare_dataset('data/training/reasoning_chat_dataset.json')
# ... (tokenization ve training kodu)
```

### Chat Bot Kullanımı

```python
import json
import random

class SimpleBot:
    def __init__(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def respond(self, user_input):
        # Basit keyword matching
        user_lower = user_input.lower()
        
        for example in self.data:
            if any(keyword in user_lower for keyword in example['question'].lower().split()):
                return example['response']
        
        return "Üzgünüm, bu konuda bilgim yok. Başka bir soru sorabilir misiniz?"

# Kullanım
bot = SimpleBot('data/training/conversational_dataset.json')
print(bot.respond("Python öğrenmek istiyorum"))
```

---

## 🔧 Dataset Genişletme

### Yeni Örnek Ekleme

```json
{
  "question": "Yeni sorunuz",
  "reasoning": "Adım adım düşünme",
  "answer": "Kısa cevap",
  "type": "technical_explanation",
  "category": "YourCategory",
  "difficulty": "intermediate"
}
```

### Best Practices

1. **Tutarlı Format:** Her dataset'te aynı field'ları kullanın
2. **Çeşitlilik:** Farklı zorluk seviyeleri ve kategoriler
3. **Kalite > Miktar:** Az ama kaliteli örnek
4. **Gerçek Senaryolar:** Gerçek kullanım case'leri
5. **Açıklamalar:** Her örneğe detaylı açıklama

---

## 📈 Gelecek Planlar

- [ ] Dataset boyutunu 100+ örneğe çıkar
- [ ] Türkçe örnekler ekle
- [ ] Görüntü ve kod dataset'leri
- [ ] Benchmark sonuçları
- [ ] Data augmentation
- [ ] Validation set ayırma
- [ ] Difficulty distribution balance

---

## 🤝 Katkıda Bulunma

Yeni örnekler eklemek için:

1. İlgili JSON dosyasını düzenleyin
2. Format'a uyun
3. Pull request açın

**Örnek Formatlar:**
- Reasoning: `question` + `reasoning` + `answer`
- Conversational: `question` + `response`
- Code: `problem` + `solution` + `explanation`

---

## 📝 Lisans

MIT License - Eğitim amaçlı kullanım için serbesttir.

---

## 🌟 Kullanım Alanları

### 1. Model Fine-tuning
```python
# GPT, BERT, T5 gibi modelleri bu dataset ile fine-tune edin
```

### 2. RAG (Retrieval Augmented Generation)
```python
# Vector database'e ekleyip retrieval yapın
```

### 3. Prompt Engineering
```python
# Few-shot learning için example'lar olarak kullanın
```

### 4. Evaluation
```python
# Model performansını test edin
```

---

**Son Güncelleme:** 17 Ekim 2025  
**Dataset Versiyonu:** 1.0.0  
**Toplam Token Sayısı:** ~50K+ (tahmini)

---

🚀 **Bu dataset'ler sürekli güncellenecek ve genişletilecektir!**
