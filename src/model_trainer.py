"""
🚀 AI MODEL TRAINER - TAM PAKET EĞİTİM SİSTEMİ
Birden fazla dataset ile otomatik model eğitimi
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# Temel kütüphaneler
import numpy as np
import pandas as pd

# Dataset yönetimi
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ datasets kütüphanesi bulunamadı. pip install datasets")

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# PyTorch (alternatif)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DatasetManager:
    """Dataset indirme ve yönetim sistemi"""
    
    def __init__(self, cache_dir="./datasets_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_dataset(self, dataset_name, split="train", max_samples=None):
        """Dataset'i indir veya cache'ten yükle"""
        if not DATASETS_AVAILABLE:
            print("❌ datasets kütüphanesi gerekli!")
            return None
            
        try:
            print(f"📥 {dataset_name} indiriliyor...")
            
            # Dataset'i yükle - farklı yöntemler dene
            try:
                if max_samples:
                    dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]", trust_remote_code=True)
                else:
                    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            except:
                # Alternatif: split belirtmeden dene
                print(f"⚠️ Split sorunlu, alternatif yöntem deneniyor...")
                full_dataset = load_dataset(dataset_name, trust_remote_code=True)
                if 'train' in full_dataset:
                    dataset = full_dataset['train']
                else:
                    # İlk mevcut split'i al
                    first_split = list(full_dataset.keys())[0]
                    dataset = full_dataset[first_split]
                
                # Max samples uygula
                if max_samples and len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
            
            print(f"✅ {len(dataset)} örnek yüklendi!")
            return dataset
            
        except Exception as e:
            print(f"❌ {dataset_name} yüklenemedi: {e}")
            print(f"💡 Bu dataset atlanıyor, devam ediliyor...")
            return None
    
    def load_multiple_datasets(self, dataset_configs, max_total=10000):
        """Birden fazla dataset'i yükle ve birleştir"""
        all_data = []
        successful_datasets = []
        failed_datasets = []
        
        for config in dataset_configs:
            name = config['name']
            max_samples = config.get('max_samples', 1000)
            split = config.get('split', 'train')
            
            print(f"\n{'='*60}")
            print(f"📚 Dataset: {name}")
            print(f"📊 Hedef: {max_samples} örnek")
            
            dataset = self.download_dataset(name, split=split, max_samples=max_samples)
            
            if dataset and len(dataset) > 0:
                # Dataset'i dict formatına çevir
                for item in dataset:
                    all_data.append(item)
                    
                print(f"✅ {len(dataset)} örnek eklendi")
                successful_datasets.append(name)
            else:
                print(f"⚠️ {name} atlandı")
                failed_datasets.append(name)
        
        print(f"\n{'='*60}")
        print(f"🎉 TOPLAM: {len(all_data)} örnek yüklendi!")
        print(f"✅ Başarılı: {len(successful_datasets)} dataset")
        if failed_datasets:
            print(f"⚠️ Atlanan: {len(failed_datasets)} dataset")
            for ds in failed_datasets:
                print(f"   • {ds}")
        return all_data


class SimpleTextModel:
    """Basit bir text generation modeli"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        
    def build_model(self):
        """Basit bir LSTM modeli oluştur"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow gerekli!")
            return None
            
        model = keras.Sequential([
            keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            keras.layers.LSTM(256, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(128),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Model oluşturuldu!")
        return model
    
    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """Modeli eğit"""
        if self.model is None:
            print("❌ Önce model oluşturun!")
            return None
            
        print(f"\n🚀 EĞİTİM BAŞLIYOR...")
        print(f"📊 Veri boyutu: {X_train.shape}")
        print(f"🔄 Epoch: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        print("\n✅ EĞİTİM TAMAMLANDI!")
        return history
    
    def save(self, filepath="./models/trained_model.h5"):
        """Modeli kaydet"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Model kaydedildi: {filepath}")


class ModelTrainer:
    """Ana eğitim sınıfı"""
    
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.model = None
        self.training_config = {
            'datasets': [
                {
                    'name': 'camel-ai/math',
                    'max_samples': 5000,
                    'weight': 0.3,
                    'split': 'train'
                },
                {
                    'name': 'yahma/alpaca-cleaned',  # Daha erişilebilir Alpaca versiyonu
                    'max_samples': 5000,
                    'weight': 0.3,
                    'split': 'train'
                },
                {
                    'name': 'tatsu-lab/alpaca',  # Orijinal Alpaca
                    'max_samples': 3000,
                    'weight': 0.2,
                    'split': 'train'
                },
                {
                    'name': 'garage-bAInd/Open-Platypus',  # Code + STEM dataset
                    'max_samples': 2000,
                    'weight': 0.2,
                    'split': 'train'
                }
            ],
            'max_total_samples': 15000,
            'test_mode': True  # İlk denemede küçük veri
        }
    
    def prepare_data(self):
        """Veriyi hazırla"""
        print("\n" + "="*60)
        print("📚 VERİ HAZIRLANIYOR...")
        print("="*60)
        
        if self.training_config['test_mode']:
            print("🧪 TEST MODU: Küçük veri seti ile çalışılıyor")
            # Test için sadece CAMEL Math kullan
            configs = [self.training_config['datasets'][0]]
            configs[0]['max_samples'] = 100  # Çok küçük test
        else:
            configs = self.training_config['datasets']
        
        # Dataset'leri yükle
        data = self.dataset_manager.load_multiple_datasets(configs)
        
        return data
    
    def start_training_demo(self):
        """Eğitim demo'su - hızlı test"""
        print("\n" + "🎯"*30)
        print("🚀 MODEL EĞİTİMİ BAŞLIYOR")
        print("🎯"*30)
        
        if not TF_AVAILABLE:
            print("\n❌ TensorFlow yüklü değil!")
            print("💡 Kurulum: pip install tensorflow")
            return
        
        # 1. Veriyi hazırla
        data = self.prepare_data()
        
        if not data or len(data) == 0:
            print("❌ Veri yüklenemedi!")
            return
        
        print(f"\n✅ {len(data)} örnek hazır!")
        
        # 2. Basit bir model oluştur
        print("\n🔨 Model oluşturuluyor...")
        model = SimpleTextModel(vocab_size=1000, max_length=50)
        model.build_model()
        
        # 3. Demo veri oluştur (gerçek tokenization yerine)
        print("\n🔄 Demo verisi hazırlanıyor...")
        X_train = np.random.randint(0, 1000, size=(len(data), 50))
        y_train = np.random.randint(0, 1000, size=(len(data),))
        
        # 4. Modeli eğit (1 epoch, hızlı test)
        print("\n" + "="*60)
        model.train(X_train, y_train, epochs=1, batch_size=16)
        
        # 5. Kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(f"./models/model_demo_{timestamp}.h5")
        
        print("\n" + "🎉"*30)
        print("✅ EĞİTİM DEMO'SU TAMAMLANDI!")
        print("🎉"*30)
        
        return model


def quick_test():
    """Hızlı test fonksiyonu"""
    print("\n" + "="*60)
    print("🧪 HIZLI TEST MODU")
    print("="*60)
    
    trainer = ModelTrainer()
    trainer.training_config['test_mode'] = True
    
    try:
        trainer.start_training_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️ Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()
