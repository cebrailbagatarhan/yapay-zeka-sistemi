# Yapay Zeka Sistemi — ModernLLM ve Qwen Deneyleri

[![Python](https://img.shields.io/badge/Python-3.11+-green?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebrailbagatarhan/yapay-zeka-sistemi/blob/main/modern_llm_training.ipynb)

Bu depo, dil modeli mimarisi ve fine-tuning üzerine deneysel/öğrenme amaçlı çalışmalar içerir. Üretime hazır bir model, GPT-4/Claude/Gemini replikası veya doğrulanmış bir “reasoning modeli” sunmaz.

> **Durum özeti:** ModernLLM kodu mevcut ve H100 üzerinde kısmi eğitim denemeleri kaydedilmiş durumda; ancak depodaki notebook çıktıları tamamlanmış, tekrarlanabilir bir eğitim koşusu göstermiyor. Qwen dosyaları ise ağırlıklı olarak fine-tuning reçetesi/notebook niteliğinde ve tamamlanmış eğitim metriği içermiyor.

## Birbirinden Ayrı Çalışma Alanları

| Alan | Başlangıç noktası | Kod / notebook | Kanıtlanan durum |
| --- | --- | --- | --- |
| **ModernLLM** | PyTorch ile tanımlanan özel decoder-only Transformer | [`modern_llm/`](modern_llm/), [`modern_llm_training.ipynb`](modern_llm_training.ipynb) | Mimari ve trainer uygulanmış; H100 koşuları kısmi/kesintili |
| **Qwen fine-tuning** | Önceden eğitilmiş Qwen2.5 modelleri + LoRA/QLoRA | [`colab_runner.ipynb`](colab_runner.ipynb), [`turkish_h100_training.ipynb`](turkish_h100_training.ipynb) | Eğitim reçetesi mevcut; tamamlanmış koşu ve final metrik yok |
| **Web asistanı** | Önceden eğitilmiş Qwen + web kaynaklarından içerik toplama | [`qwen_deep_web.py`](qwen_deep_web.py), [`qwen_web_assistant.py`](qwen_web_assistant.py), [`web_app.py`](web_app.py) | Uygulama kodu mevcut; kalite/hız benchmarkı yok |
| **Eski/demonstrasyon deneyleri** | Kural tabanlı ve küçük örnek sistemler | [`src/`](src/), [`results/`](results/) | ModernLLM veya Qwen eğitim sonucu olarak değerlendirilmemeli |

Bu alanların sonuçları birbirinin yerine kullanılamaz. Özellikle `results/` altındaki küçük teacher/student ve RL JSON dosyaları ModernLLM eğitim metriği değildir.

## ModernLLM'de Uygulanan Mimari

- RMSNorm ve pre-normalization
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU ileri beslemeli katmanlar
- KV cache ve tied input/output embeddings
- Causal language-model loss
- Mixed precision, gradient accumulation, checkpoint ve değerlendirme döngüsü

### Attention uygulaması

ModernLLM, uygun PyTorch sürümünde [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) çağrısını kullanır; aksi durumda manuel attention yoluna düşer. Bu, depoda doğrudan FlashAttention/FlashAttention-2 paketinin kullanıldığı anlamına gelmez. `use_flash_attention` yapılandırma alanı tarihsel bir isimdir ve ModernLLM kodunda PyTorch SDPA yolunu seçer.

Qwen notebook'larında isteğe bağlı `flash_attention_2` kurulumu ayrı bir deney yoludur; ModernLLM uygulamasıyla karıştırılmamalıdır.

## CoT Verisinin Anlamı

Depoda `<think>...</think>` biçiminde toplam **133** örnek bulunuyor:

| Dosya | Örnek |
| --- | ---: |
| [`data/cot/cot_training_data.json`](data/cot/cot_training_data.json) | 33 |
| [`data/cot/turkish_cot_data.json`](data/cot/turkish_cot_data.json) | 10 |
| [`data/examples/cot_dataset.json`](data/examples/cot_dataset.json) | 90 |

Bu örnekler modele belirli bir çıktı biçimi göstermek için kullanılabilir. Tek başına örnek sayısı; genellenebilir akıl yürütme, matematik doğruluğu veya zincir içi düşünmenin güvenilirliği için kanıt değildir. Depoda şu anda held-out reasoning benchmarkı, karşılaştırmalı baseline veya insan değerlendirmesi yoktur.

## Model Presetleri

Aşağıdaki parametre sayıları [`ModelConfig.estimated_params`](modern_llm/config.py) formülüyle hesaplanan yapılandırma değerleridir. “Eğitildi” anlamına gelmez.

| Preset | Hesaplanan parametre | Hidden | Katman | Attention head | KV head | Maks. context | Durum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Nano | 39,068,160 | 512 | 8 | 8 | 2 | 2,048 | Yalnızca config |
| Small | 100,289,280 | 768 | 12 | 12 | 4 | 2,048 | Yalnızca config |
| Medium | 303,612,928 | 1,024 | 24 | 16 | 4 | 4,096 | Yalnızca config |
| Large | 1,129,416,704 | 2,048 | 24 | 32 | 8 | 4,096 | Oluşturuldu; kısmi H100 koşuları var |
| XL | 3,270,183,936 | 3,072 | 32 | 32 | 8 | 8,192 | Yalnızca config |

## Kaydedilmiş Eğitim Kanıtı

Bu tablo yalnızca depoya commit edilmiş notebook çıktısını özetler. Kesintili koşulardaki ara değerler final sonuç değildir.

| Koşu | Model / donanım | Veri ve plan | Gerçekleşen durum | Son gözlem |
| --- | --- | --- | --- | --- |
| İlk pre-training | ModernLLM Large / H100 79.2 GB | 7 blok, 2 epoch | `Total steps: 0`; optimizer adımı yok | `loss=0`, `PPL=1` bir eğitim sonucu değildir |
| SFT | ModernLLM Large / H100 | 31 örnek, 3 planlı adım | Çıktı eğitim başlangıcından sonra tamamlanmıyor | Final metrik yok |
| CoT fine-tuning | ModernLLM Large / H100 | 133 örnek, 37 planlı adım | CUDA OOM ile durdu | Final metrik yok |
| Extended run | ModernLLM Large / H100 | 18,370 planlı adım | Bir deneme 200. adımda kesildi | Ara eval loss `1.4176`, PPL `4.13` |
| Optimized 1K run | ModernLLM Large / H100 | En fazla 32,000 örnek, seq len 1,024, 1,000 planlı adım | 600. adım sonrasında kesildi | Ara train loss `0.0009`; ara eval loss `0.0001`, PPL `1.00` |
| Qwen fine-tuning | Qwen2.5 1.5B/3B/7B reçeteleri | GPU'ya göre LoRA/QLoRA planı | Tamamlanmış eğitim çıktısı yok | Final metrik yok |

Optimized 1K hücresinde eğitim ve değerlendirme indeksleri aynı corpus üzerinden bağımsız seçildiği için örtüşme mümkündür. Bu nedenle çok düşük ara eval loss değeri genelleme kanıtı olarak kullanılmamalıdır.

### Sonuç tablosu

| Koşu | Params | Eğitim tokenı | GPU-saat | Final loss | Final perplexity | tok/s | Sistem RAM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ModernLLM Large | 1,129,416,704 | Kaydedilmedi | Kaydedilmedi | Yok — koşu tamamlanmadı | Yok — koşu tamamlanmadı | Ölçülmedi | Ölçülmedi | Ölçülmedi; başarısız denemeler yaklaşık 79 GB'ı doldurdu |
| Qwen fine-tuning | Seçilen base modele bağlı | Kaydedilmedi | Kaydedilmedi | Yok — tamamlanmış koşu yok | Yok — tamamlanmış koşu yok | Ölçülmedi | Ölçülmedi | Ölçülmedi |

Eksik hücreler sıfır değildir; metriklerin mevcut kod/notebook tarafından güvenilir biçimde kaydedilmediğini gösterir.

## Hızlı Başlangıç

```bash
git clone https://github.com/cebrailbagatarhan/yapay-zeka-sistemi.git
cd yapay-zeka-sistemi

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

ModernLLM notebook'u:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cebrailbagatarhan/yapay-zeka-sistemi/blob/main/modern_llm_training.ipynb)

Qwen notebook'ları, ModernLLM'den ayrı çalıştırılmalıdır:

- [`colab_runner.ipynb`](colab_runner.ipynb): GPU'ya göre Qwen model/LoRA reçetesi
- [`turkish_h100_training.ipynb`](turkish_h100_training.ipynb): Türkçe veri indirme ve Qwen SFT planı; commit edilmiş çalıştırma çıktısı yok

## Tekrarlanabilir Sonuç İçin Eksikler

Yeni bir “tamamlandı” sonucu eklemeden önce aşağıdakiler kaydedilmelidir:

1. Sabit seed, commit SHA ve tam model/training config
2. Veri kaynağı, lisansı, temizleme adımları, split manifesti ve duplicate kontrolü
3. Gerçek eğitim tokenı ve optimizer adımı
4. GPU modeli/adedi, duvar saati ve GPU-saat
5. Final train/eval loss, perplexity ve held-out değerlendirme
6. tokens/s, sistem RAM ve peak allocated/reserved VRAM
7. Kesintisiz log ve yüklenebilir checkpoint
8. CoT için ayrı, veri sızıntısından arındırılmış reasoning benchmarkı

## Bilinen Sorunlar

- Commit edilmiş ana ModernLLM notebook'unda başarısız, yeniden denenmiş ve kesilmiş hücreler birlikte bulunuyor.
- Kaydedilmiş tokenizer çalıştırmasında SentencePiece eğitimi hata veriyor ve byte-level fallback kullanılıyor.
- Bazı harici veri setleri notebook çıktısında bulunamıyor, erişim gerektiriyor veya güncel `datasets` sürümüyle yüklenemiyor.
- Extended koşularda OOM, collator hataları, checkpointing hataları ve manuel kesintiler bulunuyor.
- Qwen notebook sonuçları, ModernLLM sonuç tablosuna dahil edilmemelidir.

## Proje Yapısı

```text
modern_llm/                  # Özel PyTorch decoder-only model ve trainer
modern_llm_training.ipynb    # ModernLLM deney notebook'u
colab_runner.ipynb           # Qwen fine-tuning reçetesi
turkish_h100_training.ipynb  # Türkçe Qwen SFT reçetesi
data/                        # Küçük yerel eğitim/demonstrasyon verileri
qwen_deep_web.py             # Qwen + web araştırma uygulaması
qwen_web_assistant.py        # Qwen web yardımcı kodu
src/                         # Eski/ayrı deneyler
results/                     # Eski/demonstrasyon JSON çıktıları
tests/                       # Mevcut temel testler
```

## Lisans

Bu depoda şu anda bir `LICENSE` dosyası bulunmuyor. Bu nedenle MIT veya başka bir lisans varsayılmamalıdır. Yeniden kullanım/dağıtım koşullarını netleştirmek için açık bir lisans dosyası eklenmelidir.

## Maintainer

[@cebrailbagatarhan](https://github.com/cebrailbagatarhan)
