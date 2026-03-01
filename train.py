#!/usr/bin/env python3
"""
Modern LLM Eğitim Script'i
===========================

Sıfırdan modern bir LLM eğitmek için tek dosya script.
Google Colab veya herhangi bir GPU ortamında çalıştırılabilir.

Kullanım:
    # Colab'da:
    !git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
    %cd yapay-zeka-sistemi
    !pip install torch sentencepiece
    !python train.py

    # Argümanlarla:
    !python train.py --model nano --epochs 3 --batch_size 4
    !python train.py --model small --gpu auto --stage all
    !python train.py --model medium --data_source huggingface --hf_dataset "alibayram/turkish_instructions_150k"

Aşamalar:
    1. Pre-training (causal LM): Ham metin üzerinde dil modeli eğitimi
    2. SFT (Supervised Fine-Tuning): Chat formatında instruction eğitimi
    3. CoT (Chain-of-Thought): Düşünme yeteneği eğitimi

Yazar: Cebrail Bağatar Han
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

# Modern LLM modüllerini import et
from modern_llm.config import (
    ModelConfig,
    TrainingConfig,
    PRESET_CONFIGS,
    GPU_PRESETS,
    get_config_for_gpu,
)
from modern_llm.model.transformer import ModernLLMForCausalLM
from modern_llm.tokenizer import ModernTokenizer
from modern_llm.training.trainer import Trainer
from modern_llm.training.dataset import (
    TextDataset,
    ChatDataset,
    CoTDataset,
    load_dataset_from_json,
    load_huggingface_dataset,
    convert_to_chat_format,
    convert_cot_format,
    create_data_collator,
)
from modern_llm.inference.generator import TextGenerator, GenerationConfig
from modern_llm.cot.engine import ChainOfThoughtEngine
from modern_llm.utils import (
    get_device,
    get_gpu_info,
    set_seed,
    count_parameters,
    format_params,
    print_model_summary,
)


# ============================================================
# Yardımcı Fonksiyonlar
# ============================================================

def print_banner():
    """Başlangıç banner'ını yazdır."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              🧠 Modern LLM Training Pipeline 🧠             ║
║                                                              ║
║  Sıfırdan modern bir dil modeli eğitimi                     ║
║  RoPE • GQA • SwiGLU • RMSNorm • CoT Reasoning             ║
║                                                              ║
║  Yazar: Cebrail Bağatar Han                                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def detect_gpu():
    """GPU bilgisini tespit et ve yazdır."""
    if not torch.cuda.is_available():
        print("⚠️  CUDA bulunamadı! CPU ile devam ediliyor (çok yavaş olacak).")
        return "CPU"
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_mem:.1f} GB")
    print(f"🔧 CUDA: {torch.version.cuda}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # GPU tipini belirle
    gpu_upper = gpu_name.upper()
    if "H100" in gpu_upper:
        return "H100"
    elif "A100" in gpu_upper:
        return "A100"
    elif "L4" in gpu_upper:
        return "L4"
    elif "T4" in gpu_upper:
        return "T4"
    elif "V100" in gpu_upper:
        return "T4"  # V100 da T4 preset'ini kullansın
    else:
        print(f"ℹ️  Bilinmeyen GPU ({gpu_name}), bellek bazlı seçim yapılıyor...")
        if gpu_mem >= 70:
            return "H100"
        elif gpu_mem >= 35:
            return "A100"
        elif gpu_mem >= 20:
            return "L4"
        else:
            return "T4"


def prepare_sample_data():
    """Eğitim verisi hazırla (yerleşik örnekler + varsa dosyalar)."""
    
    # ===== Pre-training verileri =====
    pretrain_texts = [
        # Türkçe genel bilgi
        "Yapay zeka, makinelerin insan benzeri zekâ sergilemesini sağlayan bilgisayar bilimi dalıdır. "
        "Makine öğrenmesi, derin öğrenme ve doğal dil işleme gibi alt dalları vardır.",
        
        "Türkiye, Avrupa ve Asya kıtalarında toprakları bulunan bir ülkedir. "
        "Başkenti Ankara, en büyük şehri İstanbul'dur. Resmi dili Türkçedir.",
        
        "Python programlama dili, 1991 yılında Guido van Rossum tarafından geliştirilmiştir. "
        "Okunabilir sözdizimi ve geniş kütüphane desteği ile popüler bir dildir.",
        
        "Derin öğrenme, yapay sinir ağlarının çok katmanlı versiyonlarını kullanan bir makine öğrenmesi yöntemidir. "
        "Görüntü tanıma, doğal dil işleme ve konuşma tanıma gibi alanlarda kullanılır.",
        
        "Transformer mimarisi, 2017 yılında Google tarafından 'Attention Is All You Need' makalesi ile tanıtıldı. "
        "Self-attention mekanizması sayesinde paralel hesaplama yapabilir ve uzun menzilli bağımlılıkları öğrenebilir.",
        
        "İstanbul, Türkiye'nin en kalabalık şehri ve ekonomik merkezidir. "
        "Boğaziçi Köprüsü ile Avrupa ve Asya kıtalarını birbirine bağlar.",
        
        "Makine öğrenmesi algoritmaları denetimli, denetimsiz ve pekiştirmeli öğrenme olarak üç ana kategoriye ayrılır. "
        "Denetimli öğrenmede model, etiketli veriler üzerinde eğitilir.",
        
        "Doğal dil işleme, bilgisayarların insan dilini anlamasını ve üretmesini sağlayan yapay zeka dalıdır. "
        "Metin sınıflandırma, duygu analizi, makine çevirisi ve soru cevaplama gibi görevleri kapsar.",
        
        "GPU'lar, paralel hesaplama yetenekleri sayesinde derin öğrenme eğitiminde kritik rol oynar. "
        "NVIDIA'nın CUDA platformu, GPU üzerinde genel amaçlı hesaplama yapılmasını sağlar.",
        
        "Büyük dil modelleri, milyarlarca parametre içeren ve büyük metin külliyatları üzerinde eğitilen modellerdir. "
        "GPT, LLaMA, Mistral ve Claude gibi modeller bu kategoriye girer.",
        
        "Tokenization, metni daha küçük parçalara (token) ayırma işlemidir. "
        "BPE (Byte Pair Encoding), WordPiece ve SentencePiece yaygın kullanılan tokenization yöntemleridir.",
        
        "Attention mekanizması, modelin girdi sekansındaki farklı pozisyonlara farklı ağırlıklar vermesini sağlar. "
        "Multi-head attention, bu işlemi birden fazla paralel attention başlıkları ile gerçekleştirir.",
        
        "Transfer öğrenme, bir görev üzerinde eğitilmiş modelin bilgisini başka bir göreve aktarma tekniğidir. "
        "Önceden eğitilmiş dil modelleri, fine-tuning ile özel görevlere uyarlanabilir.",
        
        "Veri bilimi, veriden anlamlı bilgi çıkarmak için istatistik, matematik ve bilgisayar bilimini birleştiren disiplindir. "
        "Pandas, NumPy ve Scikit-learn Python'daki en popüler veri bilimi kütüphaneleridir.",
        
        "Siber güvenlik, bilgisayar sistemlerini, ağları ve verileri yetkisiz erişimden koruma uygulamalarını kapsar. "
        "Şifreleme, güvenlik duvarları ve saldırı tespit sistemleri temel siber güvenlik araçlarıdır.",
        
        "Kuantum bilgisayarlar, klasik bilgisayarlardan farklı olarak kübit (qubit) kullanır. "
        "Süperpozisyon ve dolanıklık gibi kuantum mekanik özellikleri sayesinde belirli problemleri çok daha hızlı çözebilir.",
        
        "Bulut bilişim, bilgi işlem kaynaklarının internet üzerinden sunulmasıdır. "
        "Amazon AWS, Google Cloud ve Microsoft Azure en büyük bulut bilişim sağlayıcılarıdır.",
        
        "Robotik, robotların tasarımı, üretimi ve kullanımı ile ilgilenen mühendislik dalıdır. "
        "Endüstriyel robotlar, otonom araçlar ve insansı robotlar bu alanın uygulama örnekleridir.",
        
        "Blokzincir teknolojisi, dağıtık ve değiştirilemez bir kayıt defteri sunar. "
        "Bitcoin ve Ethereum gibi kripto paralar blokzincir teknolojisi üzerine inşa edilmiştir.",
        
        "5G teknolojisi, beşinci nesil mobil iletişim standardıdır. "
        "Düşük gecikme süresi ve yüksek bant genişliği ile nesnelerin interneti uygulamalarını destekler.",
    ]
    
    # Yerel dosyalardan veri yükleme
    local_data_dirs = ["data/raw", "data/training", "data/processed"]
    for data_dir in local_data_dirs:
        if os.path.exists(data_dir):
            for fname in os.listdir(data_dir):
                fpath = os.path.join(data_dir, fname)
                if fname.endswith(".txt"):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text:
                                pretrain_texts.append(text)
                    except Exception:
                        pass
                elif fname.endswith(".json"):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, str):
                                        pretrain_texts.append(item)
                                    elif isinstance(item, dict) and "text" in item:
                                        pretrain_texts.append(item["text"])
                    except Exception:
                        pass
    
    # ===== SFT (Chat) verileri =====
    sft_data = [
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Python'da liste nasıl oluşturulur?"},
                {"role": "assistant", "content": "Python'da liste oluşturmak için köşeli parantez kullanılır:\n\n```python\n# Boş liste\nmy_list = []\n\n# Elemanlarla liste\nnumbers = [1, 2, 3, 4, 5]\nfruits = ['elma', 'armut', 'muz']\n\n# Karışık tipler\nmixed = [1, 'hello', 3.14, True]\n```\n\nAyrıca `list()` fonksiyonu ile de oluşturabilirsiniz."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Yapay zeka nedir?"},
                {"role": "assistant", "content": "Yapay zeka (AI), makinelerin insan benzeri zekâ sergilemesini sağlayan bilgisayar bilimi dalıdır. Temel bileşenleri şunlardır:\n\n1. **Makine Öğrenmesi**: Veriden öğrenme\n2. **Derin Öğrenme**: Çok katmanlı sinir ağları\n3. **Doğal Dil İşleme**: İnsan dilini anlama\n4. **Bilgisayarlı Görü**: Görüntü analizi\n\nYapay zeka, sağlıktan finansa, eğitimden ulaşıma kadar birçok alanda kullanılmaktadır."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Fibonacci dizisini Python'da nasıl yazarım?"},
                {"role": "assistant", "content": "Fibonacci dizisi için birkaç yöntem:\n\n```python\n# Yöntem 1: Döngü ile\ndef fibonacci(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\n# Yöntem 2: Recursive\ndef fib_recursive(n):\n    if n <= 1:\n        return n\n    return fib_recursive(n-1) + fib_recursive(n-2)\n\n# Yöntem 3: Generator\ndef fib_generator():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nprint(fibonacci(10))\n# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n```"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Transformer mimarisi nasıl çalışır?"},
                {"role": "assistant", "content": "Transformer mimarisi şu temel bileşenlerden oluşur:\n\n1. **Self-Attention**: Her token, diğer tüm tokenlarla ilişkisini hesaplar\n   - Query (Q), Key (K), Value (V) matrisleri\n   - Attention = softmax(QK^T / √d_k) × V\n\n2. **Multi-Head Attention**: Birden fazla attention başlığı paralel çalışır\n\n3. **Feed-Forward Network**: Her pozisyona ayrı ayrı uygulanan tam bağlantılı katman\n\n4. **Positional Encoding**: Token pozisyon bilgisi (modern modellerde RoPE kullanılır)\n\n5. **Layer Normalization**: Eğitim stabilitesi için normalizasyon (RMSNorm tercih edilir)\n\nModern LLM'lerde decoder-only mimari kullanılır ve causal masking ile her token sadece önceki tokenlara bakar."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Docker nedir ve neden kullanılır?"},
                {"role": "assistant", "content": "Docker, uygulamaları konteyner adı verilen izole edilmiş ortamlarda çalıştırmaya yarayan bir platformdur.\n\n**Neden kullanılır:**\n- Ortam bağımsızlığı: 'Benim bilgisayarımda çalışıyor' sorunu çözülür\n- Hızlı deployment: Konteynerler saniyeler içinde başlar\n- Kaynak verimliliği: VM'lerden çok daha hafiftir\n- Ölçeklenebilirlik: Kubernetes ile kolayca ölçeklenir\n\n**Temel komutlar:**\n```bash\n# Image oluştur\ndocker build -t myapp .\n\n# Konteyner çalıştır\ndocker run -d -p 8080:80 myapp\n\n# Çalışan konteynerleri listele\ndocker ps\n```"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "İstanbul'un tarihi hakkında bilgi ver."},
                {"role": "assistant", "content": "İstanbul, 3000 yılı aşkın tarihiyle dünyanın en eski ve önemli şehirlerinden biridir.\n\n**Tarihsel Dönemler:**\n- **Byzantion** (MÖ 660): Megara'lı kolonistler tarafından kuruldu\n- **Konstantinopolis** (MS 330): Roma İmparatoru Konstantin tarafından başkent ilan edildi\n- **Osmanlı Dönemi** (1453): Fatih Sultan Mehmet'in fethi ile Osmanlı başkenti oldu\n- **Cumhuriyet** (1923): Başkent Ankara'ya taşındı, İstanbul ticaret ve kültür merkezi olarak kaldı\n\n**Önemli Yapılar:** Ayasofya, Topkapı Sarayı, Sultanahmet Camii, Galata Kulesi, Kapalıçarşı\n\nBugün 16 milyonu aşkın nüfusuyla Türkiye'nin en büyük şehridir."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Machine learning ile deep learning arasındaki fark nedir?"},
                {"role": "assistant", "content": "Machine Learning ve Deep Learning arasındaki temel farklar:\n\n| Özellik | Machine Learning | Deep Learning |\n|---------|-----------------|---------------|\n| **Model** | Lineer regresyon, SVM, Karar ağaçları | Sinir ağları (CNN, RNN, Transformer) |\n| **Veri** | Daha az veri yeterli | Büyük veri gerektirir |\n| **Özellik** | Manuel özellik mühendisliği | Otomatik özellik çıkarımı |\n| **Donanım** | CPU yeterli | GPU/TPU gerekli |\n| **Yorumlama** | Daha yorumlanabilir | Kara kutu |\n\n**Kısaca:** Deep learning, machine learning'in bir alt dalıdır. Derin sinir ağları kullanarak veriden otomatik özellik çıkarır ve karmaşık örüntüleri öğrenir."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir yapay zeka asistanısın."},
                {"role": "user", "content": "Git versiyon kontrol sistemi nasıl kullanılır?"},
                {"role": "assistant", "content": "Git, en yaygın kullanılan dağıtık versiyon kontrol sistemidir.\n\n**Temel Komutlar:**\n```bash\n# Repo oluştur\ngit init\n\n# Değişiklikleri takip et\ngit add .                    # Tüm dosyaları ekle\ngit commit -m \"mesaj\"        # Commit oluştur\n\n# Branch işlemleri\ngit branch feature-x         # Yeni branch\ngit checkout feature-x       # Branch'e geç\ngit merge feature-x          # Birleştir\n\n# Uzak repo\ngit remote add origin URL    # Uzak repo ekle\ngit push origin main         # Push\ngit pull origin main         # Pull\n```\n\n**İyi Uygulamalar:**\n- Küçük, anlamlı commit'ler yapın\n- Açıklayıcı commit mesajları yazın\n- Feature branch'leri kullanın\n- Pull request ile code review yapın"},
            ]
        },
    ]
    
    # Yerel SFT verilerini yükle
    sft_files = [
        "data/training/conversational_dataset.json",
        "data/training/code_examples_dataset.json",
        "data/training/reasoning_chat_dataset.json",
    ]
    for sft_file in sft_files:
        if os.path.exists(sft_file):
            try:
                raw = load_dataset_from_json(sft_file)
                for item in raw:
                    if "messages" in item:
                        sft_data.append(item)
                    elif "instruction" in item and "output" in item:
                        converted = convert_to_chat_format([item])
                        sft_data.extend(converted)
                print(f"  ✅ {sft_file}: {len(raw)} örnek yüklendi")
            except Exception as e:
                print(f"  ⚠️ {sft_file} yüklenemedi: {e}")
    
    # ===== CoT verileri =====
    cot_data = []
    cot_files = [
        "data/cot/cot_training_data.json",
        "data/cot/turkish_cot_data.json",
        "data/examples/cot_dataset.json",
    ]
    for cot_file in cot_files:
        if os.path.exists(cot_file):
            try:
                raw = load_dataset_from_json(cot_file)
                for item in raw:
                    # Formatı kontrol et ve dönüştür
                    if "instruction" in item and "thinking" in item:
                        cot_data.append(item)
                    elif "question" in item and "reasoning" in item:
                        converted = convert_cot_format([item])
                        cot_data.extend(converted)
                    elif "instruction" in item and "response" in item:
                        cot_data.append(item)
                print(f"  ✅ {cot_file}: {len(raw)} CoT örneği yüklendi")
            except Exception as e:
                print(f"  ⚠️ {cot_file} yüklenemedi: {e}")
    
    # ===== data/datasets/ klasöründen otomatik yükle =====
    datasets_dir = "data/datasets"
    if os.path.exists(datasets_dir):
        for fname in sorted(os.listdir(datasets_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(datasets_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if not isinstance(raw, list) or len(raw) == 0:
                    continue
                sample = raw[0]
                count_loaded = 0
                if "messages" in sample:
                    sft_data.extend(raw)
                    count_loaded = len(raw)
                    print(f"  ✅ [SFT] datasets/{fname}: {count_loaded} örnek")
                elif "instruction" in sample and ("thinking" in sample or "response" in sample):
                    cot_data.extend(raw)
                    count_loaded = len(raw)
                    print(f"  ✅ [CoT] datasets/{fname}: {count_loaded} örnek")
                elif "text" in sample:
                    for item in raw:
                        if item.get("text"):
                            pretrain_texts.append(item["text"])
                    count_loaded = len(raw)
                    print(f"  ✅ [PRE] datasets/{fname}: {count_loaded} metin")
            except Exception as e:
                print(f"  ⚠️ datasets/{fname} yüklenemedi: {e}")
    
    # Varsayılan CoT örnekleri
    if not cot_data:
        cot_data = [
            {
                "instruction": "15 * 23 kaçtır?",
                "thinking": "Bu bir çarpma işlemi. Basamak değerlerine ayıralım:\n15 * 23 = 15 * (20 + 3) = (15 * 20) + (15 * 3) = 300 + 45 = 345",
                "response": "15 × 23 = 345"
            },
            {
                "instruction": "Python'da bir listenin elemanlarını tersine çevir.",
                "thinking": "Python'da liste tersine çevirmek için birkaç yöntem var:\n1. reversed() fonksiyonu\n2. Slice notation [::-1]\n3. .reverse() metodu (in-place)\nEn Pythonic yol slice notation kullanmaktır.",
                "response": "```python\nmy_list = [1, 2, 3, 4, 5]\nreversed_list = my_list[::-1]\nprint(reversed_list)  # [5, 4, 3, 2, 1]\n```"
            },
            {
                "instruction": "Yapay sinir ağlarında vanishing gradient problemi nedir?",
                "thinking": "Vanishing gradient problemi:\n1. Derin ağlarda geri yayılım sırasında gradyanlar katman katman çarpılır\n2. Sigmoid/tanh aktivasyon fonksiyonlarının türevleri 0-1 arasındadır\n3. Çok sayıda katmanda bu çarpımlar gradyanı sıfıra yaklaştırır\n4. Sonuç: İlk katmanlar öğrenemez\nÇözümler: ReLU aktivasyon, ResNet skip connections, LayerNorm, LSTM/GRU",
                "response": "Vanishing gradient, derin sinir ağlarında geri yayılım sırasında gradyanların katmanlar boyunca küçülerek sıfıra yaklaşması problemidir. Bu durum, ağın ilk katmanlarının etkili bir şekilde öğrenmesini engeller. Modern çözümler arasında ReLU aktivasyon fonksiyonu, residual connections (skip connections), layer normalization ve LSTM/GRU gibi kapılı yapılar bulunur."
            },
            {
                "instruction": "Bir sayının asal olup olmadığını kontrol eden Python fonksiyonu yaz.",
                "thinking": "Asal sayı kontrolü:\n1. 2'den küçük sayılar asal değildir\n2. 2 tek çift asal sayıdır\n3. Çift sayılar (2 hariç) asal değildir\n4. √n'ye kadar bölenlerini kontrol etmek yeterlidir\n5. Sadece tek sayıları kontrol etmek yeterli (2 ayrı kontrol edildi)",
                "response": "```python\nimport math\n\ndef is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(math.sqrt(n)) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\n# Test\nfor num in range(20):\n    if is_prime(num):\n        print(num, end=' ')  # 2 3 5 7 11 13 17 19\n```"
            },
            {
                "instruction": "HTTP ve HTTPS arasındaki fark nedir?",
                "thinking": "HTTP vs HTTPS:\n1. HTTP = HyperText Transfer Protocol (port 80)\n2. HTTPS = HTTP + SSL/TLS şifreleme (port 443)\n3. HTTP'de veri düz metin olarak iletilir\n4. HTTPS'de veri şifrelenir\n5. HTTPS, SSL/TLS sertifikası gerektirir\n6. SEO açısından HTTPS tercih edilir\n7. Modern web standartlarında HTTPS zorunlu hale geldi",
                "response": "HTTP (HyperText Transfer Protocol) ve HTTPS (HTTP Secure) arasındaki temel fark güvenliktir:\n\n- **HTTP**: Veri düz metin olarak iletilir, port 80 kullanır\n- **HTTPS**: SSL/TLS şifreleme ile veri korunur, port 443 kullanır\n\nHTTPS, veri bütünlüğü, gizlilik ve kimlik doğrulama sağlar. Günümüzde tüm web sitelerinde HTTPS kullanımı standart haline gelmiştir."
            },
        ]
    
    return pretrain_texts, sft_data, cot_data


def load_hf_data(args):
    """HuggingFace'den veri yükle."""
    print(f"\n📥 HuggingFace'den veri yükleniyor: {args.hf_dataset}")
    
    try:
        raw = load_huggingface_dataset(
            args.hf_dataset,
            split="train",
            max_samples=args.max_samples,
            streaming=args.streaming,
        )
        
        if not raw:
            print("⚠️ HuggingFace'den veri yüklenemedi, yerel veriler kullanılıyor.")
            return None, None, None
        
        print(f"  ✅ {len(raw)} örnek yüklendi")
        
        # Veri formatını belirle
        sample = raw[0]
        
        pretrain_texts = []
        sft_data = []
        cot_data = []
        
        if "messages" in sample:
            # Zaten chat formatında
            sft_data = raw
        elif "text" in sample:
            # Ham metin
            pretrain_texts = [item["text"] for item in raw if item.get("text")]
        elif "instruction" in sample:
            # Alpaca formatı → chat formatına dönüştür
            sft_data = convert_to_chat_format(
                raw,
                system_prompt="Sen yardımcı bir yapay zeka asistanısın.",
            )
        elif "question" in sample and "reasoning" in sample:
            # CoT formatı
            cot_data = convert_cot_format(raw)
        else:
            # Bilinen anahtarları bul
            keys = list(sample.keys())
            print(f"  ℹ️ Bilinmeyen format, anahtarlar: {keys}")
            # Metin içeren alanları pretrain'e ekle
            for item in raw:
                for key in ["content", "text", "input", "output"]:
                    if key in item and item[key]:
                        pretrain_texts.append(str(item[key]))
        
        return pretrain_texts or None, sft_data or None, cot_data or None
        
    except Exception as e:
        print(f"⚠️ HuggingFace veri yükleme hatası: {e}")
        return None, None, None


# ============================================================
# Eğitim Aşamaları
# ============================================================

def stage_pretrain(model, tokenizer, texts, train_config, args):
    """Aşama 1: Pre-training (Causal Language Modeling)."""
    print("\n" + "="*60)
    print("📚 AŞAMA 1: Pre-training (Causal LM)")
    print("="*60)
    
    if not texts:
        print("⚠️ Pre-training verisi yok, bu aşama atlanıyor.")
        return model
    
    print(f"  Metin sayısı: {len(texts)}")
    print(f"  Toplam karakter: {sum(len(t) for t in texts):,}")
    
    # Tokenizer eğitimi (eğer SentencePiece yoksa)
    if tokenizer._sp_model is None and len(texts) > 10:
        print("\n  🔤 Tokenizer eğitiliyor...")
        try:
            tokenizer.train(
                texts=texts,
                vocab_size=tokenizer.vocab_size,
                output_path=os.path.join(args.output_dir, "tokenizer"),
            )
            print("  ✅ Tokenizer eğitimi tamamlandı")
        except Exception as e:
            print(f"  ⚠️ Tokenizer eğitimi başarısız (byte-fallback devam): {e}")
    
    # Dataset oluştur
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=train_config.max_seq_length,
    )
    print(f"  Dataset boyutu: {len(dataset)} örnek")
    
    if len(dataset) < 2:
        print("  ⚠️ Yeterli veri yok, pre-training atlanıyor.")
        return model
    
    # Train/eval split
    eval_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    # Trainer oluştur
    pretrain_config = TrainingConfig(
        num_epochs=args.pretrain_epochs,
        batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_seq_length=train_config.max_seq_length,
        learning_rate=args.learning_rate or 3e-4,
        mixed_precision=train_config.mixed_precision,
        logging_steps=10,
        eval_steps=max(50, len(train_ds) // train_config.batch_size // 4),
        save_steps=max(100, len(train_ds) // train_config.batch_size // 2),
        output_dir=os.path.join(args.output_dir, "pretrain_checkpoints"),
        stage="pretrain",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    
    collate_fn = create_data_collator(tokenizer, train_config.max_seq_length)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_config=pretrain_config,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
    )
    
    print(f"\n  🚀 Pre-training başlıyor...")
    print(f"     Epochs: {pretrain_config.num_epochs}")
    print(f"     Batch size: {pretrain_config.batch_size}")
    print(f"     Gradient accumulation: {pretrain_config.gradient_accumulation_steps}")
    print(f"     Effective batch size: {pretrain_config.batch_size * pretrain_config.gradient_accumulation_steps}")
    print(f"     Learning rate: {pretrain_config.learning_rate}")
    print(f"     Mixed precision: {pretrain_config.mixed_precision}")
    
    trainer.train()
    
    print("  ✅ Pre-training tamamlandı!")
    return model


def stage_sft(model, tokenizer, sft_data, train_config, args):
    """Aşama 2: Supervised Fine-Tuning (SFT)."""
    print("\n" + "="*60)
    print("💬 AŞAMA 2: Supervised Fine-Tuning (SFT)")
    print("="*60)
    
    if not sft_data:
        print("⚠️ SFT verisi yok, bu aşama atlanıyor.")
        return model
    
    print(f"  Chat örnekleri: {len(sft_data)}")
    
    # Dataset oluştur
    dataset = ChatDataset(
        data=sft_data,
        tokenizer=tokenizer,
        max_length=train_config.max_seq_length,
        mask_user_tokens=True,
    )
    print(f"  Dataset boyutu: {len(dataset)} örnek")
    
    if len(dataset) < 2:
        print("  ⚠️ Yeterli veri yok, SFT atlanıyor.")
        return model
    
    # Train/eval split
    eval_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    # Trainer
    sft_config = TrainingConfig(
        num_epochs=args.sft_epochs,
        batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_seq_length=train_config.max_seq_length,
        learning_rate=args.learning_rate or 2e-5,
        mixed_precision=train_config.mixed_precision,
        logging_steps=5,
        eval_steps=max(20, len(train_ds) // train_config.batch_size // 4),
        save_steps=max(50, len(train_ds) // train_config.batch_size // 2),
        output_dir=os.path.join(args.output_dir, "sft_checkpoints"),
        stage="sft",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    
    collate_fn = create_data_collator(tokenizer, train_config.max_seq_length)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_config=sft_config,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
    )
    
    print(f"\n  🚀 SFT başlıyor...")
    print(f"     Epochs: {sft_config.num_epochs}")
    print(f"     Learning rate: {sft_config.learning_rate}")
    
    trainer.train()
    
    print("  ✅ SFT tamamlandı!")
    return model


def stage_cot(model, tokenizer, cot_data, train_config, args):
    """Aşama 3: Chain-of-Thought Training."""
    print("\n" + "="*60)
    print("🧠 AŞAMA 3: Chain-of-Thought (CoT) Eğitimi")
    print("="*60)
    
    if not cot_data:
        print("⚠️ CoT verisi yok, bu aşama atlanıyor.")
        return model
    
    print(f"  CoT örnekleri: {len(cot_data)}")
    
    # Dataset oluştur
    dataset = CoTDataset(
        data=cot_data,
        tokenizer=tokenizer,
        max_length=train_config.max_seq_length,
    )
    print(f"  Dataset boyutu: {len(dataset)} örnek")
    
    if len(dataset) < 2:
        print("  ⚠️ Yeterli veri yok, CoT atlanıyor.")
        return model
    
    # Train/eval split
    eval_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    # Trainer
    cot_config = TrainingConfig(
        num_epochs=args.cot_epochs,
        batch_size=max(1, train_config.batch_size // 2),  # CoT daha uzun, batch küçült
        gradient_accumulation_steps=train_config.gradient_accumulation_steps * 2,
        max_seq_length=train_config.max_seq_length,
        learning_rate=args.learning_rate or 1e-5,
        mixed_precision=train_config.mixed_precision,
        logging_steps=5,
        eval_steps=max(10, len(train_ds) // max(1, train_config.batch_size // 2) // 2),
        save_steps=max(20, len(train_ds) // max(1, train_config.batch_size // 2)),
        output_dir=os.path.join(args.output_dir, "cot_checkpoints"),
        stage="cot",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    
    collate_fn = create_data_collator(tokenizer, train_config.max_seq_length)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_config=cot_config,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
    )
    
    print(f"\n  🚀 CoT eğitimi başlıyor...")
    print(f"     Epochs: {cot_config.num_epochs}")
    print(f"     Learning rate: {cot_config.learning_rate}")
    
    trainer.train()
    
    print("  ✅ CoT eğitimi tamamlandı!")
    return model


# ============================================================
# Test ve Değerlendirme
# ============================================================

def evaluate_model(model, tokenizer, model_config, args):
    """Model değerlendirme ve örnek üretim."""
    print("\n" + "="*60)
    print("🔍 MODEL DEĞERLENDİRME")
    print("="*60)
    
    device = next(model.parameters()).device
    generator = TextGenerator(model, tokenizer, device=device)
    
    test_prompts = [
        "Yapay zeka nedir?",
        "Python'da for döngüsü nasıl kullanılır?",
        "Merhaba, nasılsın?",
        "Transformer mimarisi",
    ]
    
    print("\n📝 Örnek çıktılar (temperature=0.7):\n")
    
    for prompt in test_prompts:
        print(f"  Soru: {prompt}")
        try:
            response = generator.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
            )
            # İlk 200 karakteri göster
            display = response[:200].replace('\n', ' ')
            print(f"  Cevap: {display}")
        except Exception as e:
            print(f"  ⚠️ Üretim hatası: {e}")
        print()
    
    # CoT testi
    if model_config.cot_enabled:
        print("\n🧠 CoT (Düşünme) Testi:\n")
        try:
            cot_result = generator.generate_with_thinking(
                prompt="25 * 4 kaçtır?",
                max_new_tokens=200,
                temperature=0.5,
            )
            print(f"  Düşünce: {cot_result.get('thinking', 'N/A')[:200]}")
            print(f"  Cevap: {cot_result.get('response', 'N/A')[:200]}")
        except Exception as e:
            print(f"  ⚠️ CoT hatası: {e}")
    
    print("\n✅ Değerlendirme tamamlandı!")


def save_final_model(model, tokenizer, model_config, args):
    """Son modeli kaydet."""
    save_path = os.path.join(args.output_dir, "final_model")
    
    print(f"\n💾 Model kaydediliyor: {save_path}")
    
    # Model kaydet
    model.save_pretrained(save_path)
    print(f"  ✅ Model ağırlıkları kaydedildi")
    
    # Tokenizer kaydet
    tokenizer_path = os.path.join(save_path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"  ✅ Tokenizer kaydedildi")
    
    # Eğitim bilgilerini kaydet
    info = {
        "model_name": model_config.model_name,
        "model_version": model_config.model_version,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_kv_heads": model_config.num_kv_heads,
        "vocab_size": model_config.vocab_size,
        "parameters": count_parameters(model),
        "cot_enabled": model_config.cot_enabled,
        "training_device": str(next(model.parameters()).device),
        "pytorch_version": torch.__version__,
    }
    
    with open(os.path.join(save_path, "training_info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Eğitim bilgileri kaydedildi")
    
    print(f"\n🎉 Model başarıyla kaydedildi: {save_path}")


# ============================================================
# Ana Fonksiyon
# ============================================================

def parse_args():
    """Komut satırı argümanlarını parse et."""
    parser = argparse.ArgumentParser(
        description="Modern LLM Eğitim Script'i",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python train.py                                    # Otomatik GPU tespiti
  python train.py --model nano --epochs 5            # Nano model, 5 epoch
  python train.py --model small --gpu T4             # Small model, T4 ayarları
  python train.py --stage pretrain                   # Sadece pre-training
  python train.py --stage sft --sft_epochs 3         # Sadece SFT, 3 epoch
  python train.py --hf_dataset "alibayram/turkish_instructions_150k"  # HF veri
  python train.py --wandb --wandb_project my-llm     # Wandb ile loglama
        """,
    )
    
    # Model seçimi
    parser.add_argument("--model", type=str, default="auto",
                        choices=["auto", "nano", "small", "medium", "large", "xl"],
                        help="Model büyüklüğü (default: GPU'ya göre otomatik)")
    parser.add_argument("--gpu", type=str, default="auto",
                        help="GPU tipi (auto, T4, L4, A100, H100)")
    
    # Eğitim aşaması
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "pretrain", "sft", "cot"],
                        help="Eğitim aşaması (default: all)")
    
    # Epoch sayıları
    parser.add_argument("--epochs", type=int, default=None,
                        help="Tüm aşamalar için epoch sayısı")
    parser.add_argument("--pretrain_epochs", type=int, default=3,
                        help="Pre-training epoch sayısı")
    parser.add_argument("--sft_epochs", type=int, default=3,
                        help="SFT epoch sayısı")
    parser.add_argument("--cot_epochs", type=int, default=5,
                        help="CoT epoch sayısı")
    
    # Eğitim parametreleri
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch boyutu (default: GPU preset)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (default: aşamaya göre)")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Max sequence uzunluğu (default: GPU preset)")
    
    # Veri kaynağı
    parser.add_argument("--data_source", type=str, default="local",
                        choices=["local", "huggingface", "both"],
                        help="Veri kaynağı")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset adı")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="HF'den max örnek sayısı")
    parser.add_argument("--streaming", action="store_true",
                        help="HF streaming mode")
    
    # Çıktı
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Çıktı dizini")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Wandb logging")
    parser.add_argument("--wandb_project", type=str, default="modern-llm",
                        help="Wandb proje adı")
    
    # Çalıştırma modları
    parser.add_argument("--skip_eval", action="store_true",
                        help="Değerlendirmeyi atla")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint'tan devam et (path)")
    
    args = parser.parse_args()
    
    # --epochs belirtildiyse diğerlerini geçersiz kıl
    if args.epochs is not None:
        args.pretrain_epochs = args.epochs
        args.sft_epochs = args.epochs
        args.cot_epochs = args.epochs
    
    return args


def main():
    """Ana eğitim pipeline'ı."""
    args = parse_args()
    
    print_banner()
    
    # Seed
    set_seed(args.seed)
    print(f"🎲 Seed: {args.seed}")
    
    # ===== 1. GPU Tespiti =====
    print("\n" + "="*60)
    print("🖥️  GPU TESPİTİ")
    print("="*60)
    
    if args.gpu == "auto":
        gpu_type = detect_gpu()
    else:
        gpu_type = args.gpu.upper()
    
    print(f"\n  Seçilen GPU profili: {gpu_type}")
    
    # ===== 2. Konfigürasyon =====
    print("\n" + "="*60)
    print("⚙️  KONFİGÜRASYON")
    print("="*60)
    
    # Model konfigürasyonu
    if args.model == "auto":
        model_config, train_config = get_config_for_gpu(gpu_type)
        print(f"  Model: {model_config.model_name} (GPU'ya göre otomatik seçildi)")
    else:
        model_config = PRESET_CONFIGS[args.model]
        _, train_config = get_config_for_gpu(gpu_type)
        print(f"  Model: {model_config.model_name}")
    
    # Kullanıcı override'ları
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.max_seq_length:
        train_config.max_seq_length = args.max_seq_length
    
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Attention heads: {model_config.num_attention_heads} (KV: {model_config.num_kv_heads})")
    print(f"  Tahmini parametreler: {format_params(model_config.estimated_params)}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Max seq length: {train_config.max_seq_length}")
    print(f"  Mixed precision: {train_config.mixed_precision}")
    
    # ===== 3. Veri Hazırlığı =====
    print("\n" + "="*60)
    print("📊 VERİ HAZIRLIĞI")
    print("="*60)
    
    pretrain_texts, sft_data, cot_data = prepare_sample_data()
    
    # HuggingFace verisi
    if args.data_source in ("huggingface", "both") and args.hf_dataset:
        hf_pretrain, hf_sft, hf_cot = load_hf_data(args)
        if hf_pretrain:
            pretrain_texts.extend(hf_pretrain)
        if hf_sft:
            sft_data.extend(hf_sft)
        if hf_cot:
            cot_data.extend(hf_cot)
    
    print(f"\n  📚 Pre-training metinleri: {len(pretrain_texts)}")
    print(f"  💬 SFT (Chat) örnekleri: {len(sft_data)}")
    print(f"  🧠 CoT örnekleri: {len(cot_data)}")
    
    # ===== 4. Tokenizer =====
    print("\n" + "="*60)
    print("🔤 TOKENİZER")
    print("="*60)
    
    tokenizer = ModernTokenizer(vocab_size=model_config.vocab_size)
    
    # Tokenizer eğitimi
    if pretrain_texts and len(pretrain_texts) > 5:
        print("  Tokenizer eğitiliyor...")
        try:
            tokenizer.train(
                texts=pretrain_texts,
                vocab_size=model_config.vocab_size,
                output_path=os.path.join(args.output_dir, "tokenizer"),
            )
            print("  ✅ Tokenizer eğitimi tamamlandı")
        except Exception as e:
            print(f"  ⚠️ Tokenizer eğitimi başarısız, byte-fallback: {e}")
    else:
        print("  ℹ️ Byte-level fallback tokenizer kullanılıyor")
    
    # Test
    test_text = "Merhaba, ben bir yapay zeka modeliyim."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"  Test: '{test_text}'")
    print(f"  Tokens: {len(tokens)} token → '{decoded}'")
    
    # ===== 5. Model Oluşturma =====
    print("\n" + "="*60)
    print("🏗️  MODEL OLUŞTURMA")
    print("="*60)
    
    if args.resume:
        print(f"  Checkpoint'tan yükleniyor: {args.resume}")
        model = ModernLLMForCausalLM.from_pretrained(args.resume)
    else:
        model = ModernLLMForCausalLM(model_config)
    
    # Model özeti
    print_model_summary(model, model_config)
    
    # Device'a taşı
    device = get_device()
    model = model.to(device)
    print(f"\n  Device: {device}")
    
    # ===== 6. Eğitim =====
    start_time = time.time()
    
    if args.stage in ("all", "pretrain"):
        model = stage_pretrain(model, tokenizer, pretrain_texts, train_config, args)
    
    if args.stage in ("all", "sft"):
        model = stage_sft(model, tokenizer, sft_data, train_config, args)
    
    if args.stage in ("all", "cot"):
        model = stage_cot(model, tokenizer, cot_data, train_config, args)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    
    print(f"\n⏱️ Toplam eğitim süresi: {hours}s {mins}dk {secs}sn")
    
    # ===== 7. Değerlendirme =====
    if not args.skip_eval:
        evaluate_model(model, tokenizer, model_config, args)
    
    # ===== 8. Kaydetme =====
    save_final_model(model, tokenizer, model_config, args)
    
    # ===== 9. Sonuç =====
    print("\n" + "="*60)
    print("🎉 EĞİTİM TAMAMLANDI!")
    print("="*60)
    print(f"""
  Model: {model_config.model_name}
  Parametreler: {format_params(count_parameters(model)['total'])}
  Kayıt yolu: {os.path.join(args.output_dir, 'final_model')}
  Süre: {hours}s {mins}dk {secs}sn
    
  Modeli yüklemek için:
    from modern_llm.model.transformer import ModernLLMForCausalLM
    model = ModernLLMForCausalLM.from_pretrained("{os.path.join(args.output_dir, 'final_model')}")
    
  Metin üretmek için:
    from modern_llm.inference.generator import TextGenerator
    generator = TextGenerator(model, tokenizer)
    response = generator.generate("Merhaba!", max_new_tokens=100)
""")


if __name__ == "__main__":
    main()
