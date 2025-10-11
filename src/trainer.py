"""
Model Eğitim Sınıfı
Bu dosya, model eğitimi, validasyonu ve değerlendirmesi için gerekli araçları içerir.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, List, Optional
import os
import datetime

from .model import NeuralNetwork


class ModelTrainer:
    """
    Model eğitimi ve değerlendirmesi için kapsamlı araçlar
    """
    
    def __init__(self, model: NeuralNetwork):
        """
        Trainer'ı başlat
        
        Args:
            model: Eğitilecek NeuralNetwork modeli
        """
        self.model = model
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def prepare_data(self, X, y, test_size=0.2, validation_size=0.2, random_state=42):
        """
        Veriyi train/validation/test olarak böl
        
        Args:
            X: Özellikler
            y: Hedef değişken
            test_size: Test verisi oranı
            validation_size: Validation verisi oranı
            random_state: Rastgele tohum
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Önce train+val ve test'i ayır
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Sonra train ve validation'ı ayır
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Veri bölümü:")
        print(f"  Training: {X_train.shape[0]} örnek")
        print(f"  Validation: {X_val.shape[0]} örnek")
        print(f"  Test: {X_test.shape[0]} örnek")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_data(self, X_train, X_val=None, X_test=None):
        """
        Veriyi normalize et (Z-score normalization)
        
        Args:
            X_train: Eğitim verisi
            X_val: Validation verisi (opsiyonel)
            X_test: Test verisi (opsiyonel)
            
        Returns:
            Normalize edilmiş veriler ve normalizasyon parametreleri
        """
        # Eğitim verisinden istatistikleri hesapla
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        
        # Sıfıra bölme hatası için küçük epsilon ekle
        self.std = np.where(self.std == 0, 1e-8, self.std)
        
        # Normalize et
        X_train_norm = (X_train - self.mean) / self.std
        
        results = [X_train_norm]
        
        if X_val is not None:
            X_val_norm = (X_val - self.mean) / self.std
            results.append(X_val_norm)
        
        if X_test is not None:
            X_test_norm = (X_test - self.mean) / self.std
            results.append(X_test_norm)
        
        return tuple(results) if len(results) > 1 else X_train_norm
    
    def train_with_validation(self, X_train, y_train, X_val, y_val, 
                            epochs=1000, early_stopping_patience=50, 
                            min_delta=1e-4, verbose=True):
        """
        Validation ile model eğitimi (Early stopping dahil)
        
        Args:
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            X_val: Validation verisi
            y_val: Validation etiketleri
            epochs: Maksimum epoch sayısı
            early_stopping_patience: Early stopping için sabır
            min_delta: Minimum iyileşme miktarı
            verbose: İlerleme göster
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        print(f"Model eğitimine başlıyor...")
        print(f"Early stopping: {early_stopping_patience} epoch sabır")
        
        for epoch in range(epochs):
            # Training
            train_output = self.model.forward(X_train)
            train_loss = np.mean((train_output - y_train) ** 2)
            train_accuracy = self.model.evaluate(X_train, y_train)
            
            # Backpropagation
            self.model.backward(X_train, y_train, train_output)
            
            # Validation
            val_output = self.model.forward(X_val)
            val_loss = np.mean((val_output - y_val) ** 2)
            val_accuracy = self.model.evaluate(X_val, y_val)
            
            # Geçmişe kaydet
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_accuracy)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_accuracy)
            
            # Early stopping kontrolü
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # En iyi ağırlıkları sakla
                best_weights = [w.copy() for w in self.model.weights]
                best_biases = [b.copy() for b in self.model.biases]
            else:
                patience_counter += 1
            
            # İlerleme göster
            if verbose and (epoch % 100 == 0 or epoch < 10):
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")
                print("-" * 50)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                
                # En iyi ağırlıkları geri yükle
                if best_weights is not None:
                    self.model.weights = best_weights
                    self.model.biases = best_biases
                break
        
        print("\nEğitim tamamlandı!")
        
    def evaluate_model(self, X_test, y_test, class_names=None):
        """
        Modeli kapsamlı olarak değerlendir
        
        Args:
            X_test: Test verisi
            y_test: Test etiketleri
            class_names: Sınıf isimleri
        """
        print("\n" + "="*50)
        print("MODEL DEĞERLENDİRMESİ")
        print("="*50)
        
        # Tahminler
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Temel metrikler
        accuracy = np.mean(y_pred == y_test)
        print(f"\nDoğruluk (Accuracy): {accuracy:.4f}")
        
        # Classification report
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nKarışıklık Matrisi:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, save_path=None):
        """
        Eğitim geçmişini görselleştir
        
        Args:
            save_path: Grafik kayıt yolu (opsiyonel)
        """
        if not self.train_history['loss']:
            print("Henüz eğitim yapılmamış!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss grafiği
        epochs = range(1, len(self.train_history['loss']) + 1)
        ax1.plot(epochs, self.train_history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy grafiği
        ax2.plot(epochs, self.train_history['accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.train_history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """
        Confusion matrix'i görselleştir
        
        Args:
            cm: Confusion matrix
            class_names: Sınıf isimleri
            save_path: Kayıt yolu (opsiyonel)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix kaydedildi: {save_path}")
        
        plt.show()
    
    def save_training_report(self, save_dir, test_results=None):
        """
        Eğitim raporunu kaydet
        
        Args:
            save_dir: Kayıt dizini
            test_results: Test sonuçları
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model parametrelerini kaydet
        model_info = {
            'model_architecture': self.model.layers,
            'learning_rate': self.model.learning_rate,
            'total_epochs': len(self.train_history['loss']),
            'final_train_loss': self.train_history['loss'][-1] if self.train_history['loss'] else None,
            'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else None,
            'best_val_accuracy': max(self.train_history['val_accuracy']) if self.train_history['val_accuracy'] else None,
            'test_accuracy': test_results['accuracy'] if test_results else None,
            'timestamp': timestamp
        }
        
        # Raporu kaydet
        report_path = os.path.join(save_dir, f"training_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YAPAY ZEKA MODELİ EĞİTİM RAPORU\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Tarih: {timestamp}\n\n")
            f.write("MODEL MİMARİSİ:\n")
            f.write(f"  Katmanlar: {model_info['model_architecture']}\n")
            f.write(f"  Öğrenme Oranı: {model_info['learning_rate']}\n\n")
            f.write("EĞİTİM SONUÇLARI:\n")
            f.write(f"  Toplam Epoch: {model_info['total_epochs']}\n")
            f.write(f"  Son Train Loss: {model_info['final_train_loss']:.4f}\n")
            f.write(f"  Son Val Loss: {model_info['final_val_loss']:.4f}\n")
            f.write(f"  En İyi Val Accuracy: {model_info['best_val_accuracy']:.4f}\n")
            if test_results:
                f.write(f"  Test Accuracy: {model_info['test_accuracy']:.4f}\n")
        
        print(f"Eğitim raporu kaydedildi: {report_path}")
        
        # Modeli kaydet
        model_path = os.path.join(save_dir, f"model_{timestamp}.json")
        self.model.save_model(model_path)
        
        return report_path, model_path
