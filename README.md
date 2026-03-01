# 🤖 Gelişmiş Yapay Zeka Sistemi

[![GitHub](https://img.shields.io/badge/GitHub-cebrailbagatarhan-blue?style=flat&logo=github)](https://github.com/cebrailbagatarhan/yapay-zeka-sistemi)
[![Python](https://img.shields.io/badge/Python-3.11+-green?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Türkçe destekli, çok yetenekli yapay zeka asistanı ve model eğitim platformu. GPT-4 tarzı derin akıl yürütme (reasoning), AI destekli web araştırması, akıllı kod üretimi ve çoklu dataset ile model eğitimi özellikleri sunar.

## ✨ Özellikler

### 🧠 AI Yetenekleri
- **Gelişmiş Reasoning**: Derin analiz, çoklu perspektif, sentez ve context memory
- **Akıllı Kod Üretimi**: İsteğe göre kod yazma, iyileştirme önerileri, dosyaya kaydetme
- **Derin Web Araştırma**: AI destekli web analizi, otomatik link gezinme ve akıllı özet oluşturma
- **Chatbot**: Matematik, Python, ML ve genel konularda açıklamalı yanıtlar
- **Demo Modülleri**: LoRA, CoT dataset, RL tutor, Knowledge Distillation

### 📚 Model Eğitimi
- **Çoklu Dataset Desteği**: 
  - 🧮 CAMEL AI Math (5K matematik problemi)
  - 📝 Alpaca Cleaned (5K instruction-following)
  - 🎯 Tatsu-Lab Alpaca (3K genel görev)
  - 💻 Open Platypus (2K kod + STEM)
- **Otomatik Dataset İndirme**: Hugging Face entegrasyonu
- **Esnek Eğitim Modları**: Test, küçük, orta, tam paket seçenekleri

### 🇹🇷 Türkçe Model Eğitimi (YENİ)
- **H100/A100/T4 GPU Desteği**: Otomatik GPU algılama ve optimizasyon
- **300K+ Türkçe Dataset**: HuggingFace'den otomatik indirme
  - 📖 alibayram/turkish_instructions_150k (150K talimat)
  - 📝 malhajar/alpaca-turkish (52K Alpaca)
  - 🎯 merve/turkish_instructions (53K talimat)
  - 🌍 MBZUAI/Bactrian-X - Türkçe (67K çokdilli)
  - 📚 wikimedia/wikipedia - Türkçe (100K makale)
  - 📰 uonlp/CulturaX - Türkçe (50K streaming)
- **İki Aşamalı Eğitim**: Continued Pre-training + SFT (Supervised Fine-Tuning)
- **Qwen 2.5 Tabanlı**: 7B (H100/A100), 3B (L4), 1.5B (T4) otomatik seçim
- **Google Colab Notebook**: `turkish_h100_training.ipynb`

### 🔍 Derin Web Araştırma
- Wikipedia + DuckDuckGo otomatik tarama
- 20+ güvenilir kaynak desteği
- İlgili linklere otomatik gezinme (3 seviye derinlik)
- AI destekli akıllı özet oluşturma
- Anahtar kelime analizi ve kaynak güvenilirlik kontrolü
- JSON + TXT formatında sonuç kaydetme

## 📁 Proje Yapısı

```
yapay-zeka-sistemi/
├── main.py                         # Ana menü (21 özellik)
├── turkish_h100_training.ipynb     # 🇹🇷 Türkçe H100 eğitim notebook'u
├── colab_runner.ipynb              # Google Colab eğitim notebook'u
├── README.md                       # Proje dokümantasyonu
├── requirements.txt                # Python bağımlılıkları
├── requirements_web.txt            # Web araştırma bağımlılıkları
├── src/
│   ├── advanced_reasoning.py       # Reasoning, Web ve Kod üretim motorları
│   ├── web_research.py             # Web araştırma, bilgi tabanı
│   ├── deep_web_researcher.py      # AI destekli derin web araştırma
│   ├── model_trainer.py            # Çoklu dataset model eğitimi
│   ├── advanced_ai.py              # İleri seviye AI demoları
│   └── __init__.py                 # Model ve yardımcı sınıflar
├── notebooks/                      # Jupyter notebook'lar
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_advanced_ai_techniques.ipynb
├── data/
│   ├── knowledge_base.json         # Bilgi tabanı
│   ├── turkish_datasets/           # 🇹🇷 Türkçe dataset kaynakları
│   │   └── dataset_sources.json
│   ├── training/                   # Eğitim verileri
│   │   ├── turkish_general_dataset.json
│   │   ├── conversational_dataset.json
│   │   ├── reasoning_chat_dataset.json
│   │   └── code_examples_dataset.json
│   └── examples/
├── qwen-model/                     # Qwen 2.5 model dosyaları
├── generated/                      # Kod üretici çıktıları
└── tests/                          # Test dosyaları
```

## 🚀 Kurulum

### Gereksinimler
- Python 3.11+
- pip veya conda
- Git

### Kurulum Adımları

```bash
# Repository'yi klonlayın
git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
cd yapay-zeka-sistemi

# Sanal ortam oluşturun (önerilen)
python -m venv .conda
.conda\Scripts\activate  # Windows
# source .conda/bin/activate  # Linux/Mac

# Bağımlılıkları yükleyin
pip install -r requirements.txt
pip install -r requirements_web.txt
```

## 📖 Kullanım

### Hızlı Başlangıç

```bash
python main.py
```

Ana menüden istediğiniz özelliği seçin (21 seçenek).

### 🇹🇷 Türkçe Model Eğitimi (Google Colab)

H100/A100/T4 GPU üzerinde Türkçe LLM eğitimi için:

1. `turkish_h100_training.ipynb` dosyasını Google Colab'da açın
2. **Runtime > Change runtime type > GPU** (H100 önerilir)
3. Tüm hücreleri sırasıyla çalıştırın

#### GPU Performans Tablosu

| GPU | Model | Batch Size | Sequence | LoRA Rank | Precision | Tahmini Süre |
|-----|-------|-----------|----------|-----------|-----------|-------------|
| **H100 80GB** | Qwen 2.5-7B | 8 | 2048 | 64 | BF16 | ~2 saat |
| **A100 40GB** | Qwen 2.5-7B | 4 | 2048 | 32 | BF16 | ~4 saat |
| **L4 24GB** | Qwen 2.5-3B | 4 | 1024 | 32 | BF16 | ~6 saat |
| **T4 16GB** | Qwen 2.5-1.5B | 2 | 512 | 16 | FP16 4-bit | ~8 saat |

#### Eğitim Pipeline

```
1. HuggingFace'den 300K+ Türkçe dataset indirme
2. Veri birleştirme ve chat formatına dönüştürme
3. Continued Pre-training (Wikipedia + Web corpus)
4. SFT (Supervised Fine-Tuning) Türkçe talimatlarla
5. Otomatik Türkçe kalite değerlendirmesi
6. Model kaydetme (lokal + Google Drive + HuggingFace Hub)
```

#### H100 Optimizasyonları
- Flash Attention 2 (otomatik)
- BF16 precision (quantization yok)
- AdamW Fused optimizer
- LoRA r=64, alpha=128
- Gradient accumulation steps: 2
- Max sequence length: 2048

### 🎯 Diğer Ana Özellikler

#### 🤖 Gelişmiş Sohbet
- Doğal dil işleme ve bağlam anlama
- Matematik ve programlama desteği
- Context memory ile akıllı yanıtlar

#### 🔍 Derin Web Araştırma (Seçenek 20)
- Wikipedia + DuckDuckGo otomatik tarama
- İlgili linklere otomatik gezinme (3 seviye)
- AI destekli akıllı özet oluşturma
- JSON + TXT formatında kaydetme

#### 📚 Çoklu Dataset Model Eğitimi (Seçenek 21)
- Otomatik dataset indirme (HuggingFace)
- Çoklu dataset birleştirme
- Hata toleranslı yükleme
- İlerleme takibi

#### 💻 Akıllı Kod Üretici
- Otomatik kod yazma
- Kod açıklama ve iyileştirme
- `generated/` klasörüne kaydetme

## 🛠️ Teknolojiler

- **AI/ML**: TensorFlow, PyTorch, scikit-learn
- **NLP**: Hugging Face Transformers, Datasets, PEFT, TRL
- **Web Scraping**: BeautifulSoup4, Requests, Wikipedia API
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn
- **Training**: Flash Attention 2, bitsandbytes, LoRA/QLoRA
- **Development**: Python 3.11+

## 📊 Dataset Bilgileri

### İngilizce Datasetler

| Dataset | Boyut | Ağırlık | Amaç |
|---------|-------|---------|------|
| CAMEL AI Math | 5,000 | 30% | Matematik problemleri |
| Alpaca Cleaned | 5,000 | 30% | Instruction-following |
| Tatsu-Lab Alpaca | 3,000 | 20% | Genel görevler |
| Open Platypus | 2,000 | 20% | Kod + STEM |

### 🇹🇷 Türkçe Datasetler

| Dataset | Boyut | Kaynak | Amaç |
|---------|-------|--------|------|
| turkish_instructions_150k | 150,000 | alibayram | Türkçe talimatlar |
| alpaca-turkish | 52,000 | malhajar | Türkçe Alpaca |
| turkish_instructions | 53,000 | merve | Türkçe talimatlar |
| Bactrian-X (tr) | 67,000 | MBZUAI | Çokdilli talimatlar |
| Wikipedia (tr) | 100,000 | wikimedia | Ön-eğitim corpus |
| CulturaX (tr) | 50,000 | uonlp | Web corpus |

**Toplam Türkçe: ~472,000 eğitim örneği**

## 💡 Kullanım Örnekleri

### Chatbot Örnekleri
```python
# Matematik
"sin(pi/2) hesapla"
"√16 + 2^3"

# Python
"Python'da for döngüsü nasıl kullanılır?"
"Liste dilimleme nedir?"
```

### Kod Üretici Örnekleri
```python
"Flask ile REST API yaz"
"Binary search algoritması"
"Pandas ile CSV dosyası analiz et"
```

### Web Araştırma Örnekleri
```python
"LLM evaluation yöntemleri"
"RAG mimarisi nedir?"
"Transformer architecture"
```

## 🎯 Özellik Detayları

### Derin Web Araştırma Akışı
1. **Kaynak Tarama**: Wikipedia + DuckDuckGo + 20 güvenilir site
2. **Link Toplama**: Her kaynaktan ilgili linkler
3. **Derin Gezinme**: Linklere girerek içerik çıkarma (3 seviye)
4. **AI Analizi**: Anahtar kelime + kaynak analizi
5. **Özet Oluşturma**: AI destekli akıllı özet
6. **Kaydetme**: JSON (ham veri) + TXT (özet)

### Model Eğitimi Akışı
1. **Dataset Seçimi**: 4 İngilizce + 6 Türkçe dataset
2. **Otomatik İndirme**: Hugging Face'den
3. **Birleştirme**: Ağırlıklı örnekleme
4. **Eğitim**: GPU'ya göre optimize edilmiş parametreler
5. **Kaydetme**: Checkpoint + metrikler

## 🔧 Sorun Giderme

### Paket Hataları
```bash
# Web araştırma hatası
pip install -r requirements_web.txt

# Dataset hatası
pip install datasets transformers

# AI demo hatası
pip install gymnasium
```

### Karakter Kodlama Sorunları
- Windows Terminal kullanın (UTF-8 desteği için)
- PowerShell'de: `chcp 65001`

### Dataset İndirme Hataları
- İnternet bağlantınızı kontrol edin
- Hugging Face hesabınızla giriş yapın (bazı datasetler için)
- VPN kullanıyorsanız kapatın

### Colab / GPU Hataları
- H100 kullanıyorsanız: Flash Attention 2 otomatik yüklenir
- T4 kullanıyorsanız: 4-bit quantization otomatik aktif olur
- CUDA out of memory: Batch size veya sequence length azaltın

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit yapın (`git commit -m 'Yeni özellik: XYZ'`)
4. Push yapın (`git push origin feature/YeniOzellik`)
5. Pull Request açın

## 📝 Geliştirme Yol Haritası

- [x] Türkçe model eğitim notebook'u (H100 optimizeli)
- [x] 300K+ Türkçe dataset entegrasyonu
- [x] Çoklu GPU desteği (H100/A100/L4/T4)
- [ ] GPT-4 API entegrasyonu
- [ ] Vektör veritabanı desteği (Pinecone, Weaviate)
- [ ] RAG (Retrieval Augmented Generation)
- [ ] Fine-tuning pipeline iyileştirmeleri
- [ ] Web UI (Gradio/Streamlit)
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👤 Geliştirici

**Cebrail Bağatar Han**
- GitHub: [@cebrailbagatarhan](https://github.com/cebrailbagatarhan)

## 🙏 Teşekkürler

- [Hugging Face](https://huggingface.co/) - Dataset ve model desteği
- [OpenAI](https://openai.com/) - İlham kaynağı
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page) - Web araştırma
- [Qwen Team](https://huggingface.co/Qwen) - Temel model
- Tüm açık kaynak topluluğuna

## 📞 İletişim

Sorularınız ve önerileriniz için:
- Issue açın: [GitHub Issues](https://github.com/cebrailbagatarhan/yapay-zeka-sistemi/issues)
- Pull Request gönderin

---

⭐ **Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

🚀 **Happy Coding!**
