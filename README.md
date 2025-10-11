# Yapay Zeka Asistanı (GPT-4 Tarzı)

Bu proje, GPT-4 tarzı derin akıl yürütme (reasoning), akıllı kod üretimi ve web araştırması yapabilen etkileşimli bir komut satırı uygulamasıdır. Sistem; insan benzeri düşünme adımlarını simüle eder, matematik ve programlama sorularına açıklamalı yanıtlar üretir, gerektiğinde internette araştırma yaparak sonuçları özetler.

## Özellikler
- 🧠 Gelişmiş Reasoning: Derin analiz, çoklu perspektif, sentez ve context memory
- 💻 Akıllı Kod Üretimi: İsteğe göre kod yazma, iyileştirme önerileri, dosyaya kaydetme
- 🌐 Web Araştırması: Wikipedia/haber/specialized kaynaklarda arama ve özetleme
- 🤖 Chatbot: Matematik, Python, ML ve genel konularda açıklamalı yanıtlar
- 🧪 Demo Modülleri: LoRA, CoT dataset, RL tutor, Knowledge Distillation (opsiyonel)

## Dizin Yapısı (özet)
```
/ (proje kökü)
├─ main.py                         # Ana menü ve tüm akış
├─ README.md                       # Bu dosya
├─ requirements_web.txt            # Web araştırma modülü için kütüphaneler
├─ src/
│  ├─ advanced_reasoning.py        # Reasoning, Web ve Kod üretim motorları
│  ├─ web_research.py              # Web araştırma, bilgi tabanı, otomatik öğrenme
│  ├─ advanced_ai.py               # (Opsiyonel) ileri seviye AI demoları
│  └─ __init__.py                  # Model ve yardımcı sınıflar
├─ data/                           # Bilgi tabanı ve örnek veri (oluşturulabilir)
└─ generated/                      # Kod üretici çıktıları (otomatik oluşturulur)
```

## Gereksinimler
- Python 3.10+ (önerilir)
- Windows Terminal/PowerShell için UTF-8 desteği (emoji/Türkçe karakterler için)

## Kurulum
PowerShell ile önerilen kurulum adımları:

```powershell
# (İsteğe bağlı) Sanal ortam
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Web araştırması için gerekli paketler
pip install -r requirements_web.txt

# (Opsiyonel) İleri düzey demolar için gerekli paketler
pip install gymnasium
```

### Alternatif: Conda ile Kurulum (Windows)
Conda kurulu değilse Miniconda/Anaconda yükleyin ve yeni bir PowerShell oturumu açın.

```powershell
# Yeni ortam oluştur ve etkinleştir
conda create -y -n yapay python=3.11
conda activate yapay

# Gerekli paketler
pip install -r requirements.txt
pip install -r requirements_web.txt

# (Opsiyonel) İleri seviye demolar
pip install gymnasium
```

Not:
- `advanced_ai.py` içeren demolar bazı ek kütüphaneler ister; yüklü değilse program çalışır fakat ilgili demolar atlanır.
- `wikipedia`, `requests`, `beautifulsoup4`, `feedparser` paketleri web araştırma için gereklidir (requirements_web.txt içinde vardır).

## Çalıştırma
```powershell
python .\main.py
```
Uygulama açıldığında menüden aşağıdaki gibi özelliklere erişebilirsiniz:

- 🤖 Advanced AI Chatbot
- 🌐 Web Research Demo / Akıllı Web Chatbot / Bilgi Tabanı Yönetimi
- 🧠 “Senin Gibi Düşünen AI” (Thinking Clone)
- 💻 Gelişmiş Kod Üretici
- 🧪 LoRA, CoT, RL, Distillation demoları (opsiyonel)

## Hızlı Kullanım Örnekleri
- Chatbot (matematik): `sin(pi/2) hesapla`, `√16 + 2^3`, `27 × 15`
- Chatbot (Python): `Python'da for döngüsü`, `Liste dilimleme nedir?`
- Kod Üretici: `Flask ile basit API`, `ikili arama fonksiyonu`, `Pandas ile CSV analiz`
- Web Araştırması: `LLM evaluation yöntemleri nedir?`, `RAG mimarisi hakkında`

## Gelişmiş Kod Üretici
- İstediğiniz kodu tarif edin.
- Önizleme konsolda gösterilir; kaydetmeyi seçerseniz `generated/` klasörüne `codegen_{n}_{konu}.py` olarak yazılır.

## Sorun Giderme
- `No module named 'wikipedia'` veya benzeri: `pip install -r requirements_web.txt`
- `No module named 'gymnasium'`: İleri düzey AI demoları için `pip install gymnasium`
- Türkçe/Emoji bozuk çıkıyor: PowerShell’de UTF-8 etkin olduğundan emin olun (Windows Terminal önerilir).

## Notlar
- Web araştırma çıktıları `data/` altında bilgi tabanına kaydedilebilir.
- Kod üretimi sonucu dosyalar `generated/` klasörüne kaydedilir (otomatik oluşturulur).
- Jupyter notebooklar `notebooks/` klasöründe olabilir; bazı demoları oradan tetikleyebilirsiniz.

## Katkı ve Lisans
- PR ve önerilere açıktır. Lisans bilgisi eklenmediyse, lütfen kullanmadan önce proje sahibine danışın.
