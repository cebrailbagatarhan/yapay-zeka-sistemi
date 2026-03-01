# � Modern LLM - Sıfırdan Yapay Zeka Sistemi

[![GitHub](https://img.shields.io/badge/GitHub-cebrailbagatarhan-blue?style=flat&logo=github)](https://github.com/cebrailbagatarhan/yapay-zeka-sistemi)
[![Python](https://img.shields.io/badge/Python-3.11+-green?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebrailbagatarhan/yapay-zeka-sistemi/blob/main/modern_llm_training.ipynb)

**Sıfırdan yazılmış, günümüz modern LLM mimarilerini (GPT-4, Claude, Gemini, LLaMA 3) temel alan bir dil modeli.**
Chain-of-Thought (CoT) reasoning, Türkçe dil desteği ve Google Colab üzerinde eğitim imkanı sunar.

## ✨ Temel Özellikler

### 🏗️ Modern LLM Mimarisi (Sıfırdan)
- **RMSNorm** — Pre-normalization (LayerNorm yerine)
- **Rotary Position Embeddings (RoPE)** — θ=500,000, linear/dynamic/NTK scaling
- **Grouped Query Attention (GQA)** — 4:1 ratio ile verimli attention
- **SwiGLU Activation** — Gate + Up + Down projections (GELU yerine)
- **Flash Attention** — PyTorch 2.0+ SDPA, manual fallback
- **KV-Cache** — Verimli autoregressive inference
- **Weight Tying** — Embedding ve LM Head ağırlık paylaşımı

### 🧠 Chain-of-Thought (CoT) Reasoning
- **`<think>...</think>`** blokları ile dahili akıl yürütme
- Otomatik görev tipi tespiti (math, code, logic, analysis)
- Güven skoru hesaplama
- 35+ CoT eğitim örneği (Türkçe + İngilizce)
- Claude / DeepSeek-R1 tarzı reasoning

### 🔤 Custom Tokenizer
- **SentencePiece BPE** tokenizer (sıfırdan eğitim)
- **Byte-level fallback** (SentencePiece olmadan da çalışır)
- 14 özel token: `<pad>`, `<bos>`, `<eos>`, `<think>`, `</think>`, chat tokenları
- Türkçe karakter desteği (çÇğĞıİöÖşŞüÜ)
- Chat template sistemi

### 📊 Model Boyutları

| Model | Params | GPU | Hidden | Layers | Heads | KV Heads |
|-------|--------|-----|--------|--------|-------|----------|
| **Nano** | ~30M | T4 (16GB) | 512 | 8 | 8 | 2 |
| **Small** | ~100M | L4 (24GB) | 768 | 12 | 12 | 4 |
| **Medium** | ~350M | A100 (40GB) | 1024 | 24 | 16 | 4 |
| **Large** | ~1.3B | H100 (80GB) | 2048 | 24 | 32 | 8 |
| **XL** | ~3B | H100 (80GB) | 3072 | 32 | 32 | 8 |

### 🎯 3 Aşamalı Eğitim Pipeline
1. **Pre-training** — Causal LM (next-token prediction)
2. **SFT** — Supervised Fine-Tuning (instruction-following)
3. **CoT Fine-tuning** — Chain-of-Thought reasoning eğitimi

### 🇹🇷 Türkçe Dil Desteği
- Türkçe CoT eğitim verileri (Atatürk, coğrafya, dilbilgisi, tarih vb.)
- HuggingFace Türkçe dataset entegrasyonu
- Türkçe karakter setine özel tokenizer desteği

### 🔧 Ek Özellikler
- Qwen 2.5 fine-tuning (mevcut, ayrı notebook)
- AI destekli web araştırma
- Akıllı kod üretimi
- İnteraktif sohbet arayüzü

## 📁 Proje Yapısı

```
yapay-zeka-sistemi/
├── modern_llm/                      # 🧠 Sıfırdan Modern LLM
│   ├── __init__.py                  # Paket tanımları
│   ├── config.py                    # Model + Training konfigürasyonları
│   ├── tokenizer.py                 # SentencePiece BPE tokenizer
│   ├── utils.py                     # GPU, bellek, seed, timer utilities
│   ├── model/                       # Transformer mimarisi
│   │   ├── attention.py             # RoPE + GQA + Flash Attention
│   │   ├── layers.py                # RMSNorm + SwiGLU + TransformerBlock
│   │   └── transformer.py           # ModernLLMForCausalLM (tam model)
│   ├── training/                    # Eğitim pipeline
│   │   ├── dataset.py               # TextDataset, ChatDataset, CoTDataset
│   │   └── trainer.py               # Sıfırdan Trainer (mixed precision, grad accum)
│   ├── inference/                   # Inference pipeline
│   │   └── generator.py             # TextGenerator + ChatInterface
│   └── cot/                         # Chain-of-Thought modülü
│       └── engine.py                # CoT Engine (task detection, confidence)
├── modern_llm_training.ipynb        # 🚀 Colab Eğitim Notebook'u
├── colab_runner.ipynb               # Qwen fine-tuning notebook'u
├── turkish_h100_training.ipynb      # Türkçe H100 eğitim notebook'u
├── data/
│   ├── cot/                         # CoT eğitim verileri
│   │   ├── cot_training_data.json   # 25 CoT örneği (EN+TR)
│   │   └── turkish_cot_data.json    # 10 Türkçe CoT örneği
│   ├── training/                    # SFT eğitim verileri
│   ├── examples/                    # Ek dataset örnekleri
│   └── knowledge_base.json
├── src/                             # Orijinal AI sistemi
├── qwen-model/                      # Qwen 2.5 model dosyaları
├── main.py                          # Ana menü
└── tests/                           # Test dosyaları
```

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.11+
- PyTorch 2.0+
- GPU (önerilen): T4/L4/A100/H100

### Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
cd yapay-zeka-sistemi

# Sanal ortam oluşturun
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Bağımlılıkları yükleyin
pip install torch sentencepiece datasets wandb
pip install -r requirements.txt
```

### Google Colab'da Eğitim (Önerilen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebrailbagatarhan/yapay-zeka-sistemi/blob/main/modern_llm_training.ipynb)

1. Yukarıdaki butona tıklayın
2. **Runtime > Change runtime type > GPU** seçin (H100/A100 önerilir)
3. Hücreleri sırasıyla çalıştırın

Notebook otomatik olarak:
- GPU tipini algılar ve uygun model boyutunu seçer
- Eğitim verisini hazırlar (Türkçe + İngilizce)
- Tokenizer'ı eğitir
- 3 aşamalı eğitimi çalıştırır (Pre-train → SFT → CoT)
- Modeli kaydeder ve test eder

## 📖 Kullanım

### Python'da Kullanım

```python
from modern_llm.config import PRESET_CONFIGS
from modern_llm.model.transformer import ModernLLMForCausalLM
from modern_llm.tokenizer import ModernTokenizer
from modern_llm.inference.generator import TextGenerator

# Model oluştur (veya eğitilmiş modeli yükle)
config = PRESET_CONFIGS["nano"]
model = ModernLLMForCausalLM(config)

# Tokenizer
tokenizer = ModernTokenizer(vocab_size=config.vocab_size)

# Metin üret
generator = TextGenerator(model, tokenizer)
output = generator.generate("Yapay zeka nedir?", max_new_tokens=200)
print(output)
```

### CoT (Düşünme) Modu

```python
# Chain-of-Thought ile akıl yürütme
result = generator.generate_with_thinking(
    prompt="15 * 23 kaçtır? Adım adım hesapla.",
    max_new_tokens=300,
    temperature=0.3,
)
print(f"Düşünme: {result['thinking']}")
print(f"Cevap: {result['response']}")
```

### İnteraktif Sohbet

```python
from modern_llm.inference.generator import ChatInterface

chat = ChatInterface(model, tokenizer)
chat.start()  # Terminal'de interaktif sohbet başlatır

# Komutlar: /think (CoT aç/kapa), /reset, /temp 0.5, /quit
```

### Eğitilmiş Modeli Yükleme

```python
# Kayıtlı modeli yükle
model = ModernLLMForCausalLM.from_pretrained("trained_model/")
tokenizer = ModernTokenizer(model_path="trained_model/tokenizer")
generator = TextGenerator(model, tokenizer)

output = generator.generate("Merhaba!")
```

## 🏗️ Mimari Detayları

### Transformer Blok Akışı

```
Input → Embed → [TransformerBlock × N] → RMSNorm → LM Head → Logits

TransformerBlock:
  x → RMSNorm → GQA(RoPE) → + residual
    → RMSNorm → SwiGLU FFN → + residual
```

### Grouped Query Attention (GQA)

```
Query Heads:  [H1] [H2] [H3] [H4] [H5] [H6] [H7] [H8] [H9] [H10] [H11] [H12]
               ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓     ↓     ↓
KV Heads:     [KV1      ][KV2      ][KV3      ][KV4      ]  (4:1 ratio)
```

### SwiGLU FFN

```
SwiGLU(x) = SiLU(W_gate · x) ⊙ (W_up · x)
Output    = W_down · SwiGLU(x)
```

### CoT Format

```
<|im_start|><|assistant|>
<think>
Adım 1: Problemi anlıyorum...
Adım 2: Çözüm yolunu belirliyorum...
Adım 3: Hesaplama yapıyorum...
</think>
Nihai cevabım şudur: ...
<|im_end|>
```

## 📊 GPU Performans Tablosu

### Modern LLM (Sıfırdan)

| GPU | Model | Params | Batch | Seq Len | Precision | Grad Accum |
|-----|-------|--------|-------|---------|-----------|------------|
| **T4 16GB** | Nano | ~30M | 4 | 512 | FP16 | 8 |
| **L4 24GB** | Small | ~100M | 4 | 1024 | BF16 | 4 |
| **A100 40GB** | Medium | ~350M | 8 | 2048 | BF16 | 2 |
| **H100 80GB** | Large | ~1.3B | 16 | 4096 | BF16 | 1 |

### Qwen Fine-tuning (Mevcut)

| GPU | Model | Batch | LoRA Rank | Precision |
|-----|-------|-------|-----------|-----------|
| **T4** | Qwen 2.5-1.5B | 2 | 16 | FP16 4-bit |
| **A100** | Qwen 2.5-7B | 4 | 32 | BF16 |
| **H100** | Qwen 2.5-7B | 8 | 64 | BF16 |

## 🛠️ Teknolojiler

- **AI/ML**: PyTorch 2.0+, SentencePiece, Wandb
- **Mimari**: RMSNorm, RoPE, GQA, SwiGLU, Flash Attention, KV-Cache
- **Eğitim**: Mixed Precision (BF16/FP16), Gradient Accumulation, Cosine LR
- **NLP**: Hugging Face Datasets, Transformers, PEFT, TRL
- **Web**: BeautifulSoup4, Requests, Wikipedia API
- **Platform**: Google Colab, CUDA 12+

## 📝 Geliştirme Yol Haritası

- [x] Sıfırdan modern LLM mimarisi (RoPE, GQA, SwiGLU, RMSNorm)
- [x] Chain-of-Thought (CoT) reasoning modülü
- [x] Custom BPE tokenizer (Türkçe destekli)
- [x] 3 aşamalı eğitim pipeline (Pretrain → SFT → CoT)
- [x] Google Colab eğitim notebook'u
- [x] CoT eğitim verileri (35+ örnek, TR+EN)
- [x] Qwen fine-tuning (ayrı notebook)
- [x] Türkçe H100 eğitim notebook'u
- [ ] DPO (Direct Preference Optimization) desteği
- [ ] RLHF pipeline
- [ ] Daha büyük CoT dataset (1000+ örnek)
- [ ] Daha fazla HuggingFace dataset entegrasyonu
- [ ] Web UI (Gradio/Streamlit)
- [ ] Model Hub'a yükleme
- [ ] Docker containerization
- [ ] Multi-GPU eğitim (FSDP/DeepSpeed)

## 📄 Lisans

MIT Lisansı — Detaylar için `LICENSE` dosyasına bakın.

## 👤 Geliştirici

**Cebrail Bağatar Han**
- GitHub: [@cebrailbagatarhan](https://github.com/cebrailbagatarhan)

## 🙏 Teşekkürler

- [PyTorch](https://pytorch.org/) — Temel deep learning framework
- [Hugging Face](https://huggingface.co/) — Dataset ve model ekosistemi
- [SentencePiece](https://github.com/google/sentencepiece) — Tokenizer
- [LLaMA](https://ai.meta.com/llama/) / [Mistral](https://mistral.ai/) — Mimari ilham
- Tüm açık kaynak topluluğuna

---

⭐ **Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

🚀 **Happy Coding!**
