"""
Veri İşleme Modülü
Bu dosya, veri yükleme, preprocessing ve augmentation işlemlerini içerir.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris, load_digits
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os


class DataProcessor:
    """
    Veri işleme ve hazırlama için kapsamlı araçlar
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
    
    def generate_synthetic_data(self, n_samples=1000, n_features=10, n_classes=2, 
                              noise=0.1, random_state=42):
        """
        Sentetik veri üret
        
        Args:
            n_samples: Örnek sayısı
            n_features: Özellik sayısı
            n_classes: Sınıf sayısı
            noise: Gürültü seviyesi
            random_state: Rastgele tohum
            
        Returns:
            X, y: Özellikler ve hedef değişken
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            random_state=random_state,
            n_clusters_per_class=1
        )
        
        # Gürültü ekle
        X += np.random.normal(0, noise, X.shape)
        
        # Binary classification için reshape
        if n_classes == 2:
            y = y.reshape(-1, 1)
        
        print(f"Sentetik veri oluşturuldu:")
        print(f"  Örnekler: {X.shape[0]}")
        print(f"  Özellikler: {X.shape[1]}")
        print(f"  Sınıflar: {n_classes}")
        
        return X, y
    
    def load_iris_dataset(self):
        """
        Iris veri setini yükle ve binary classification için hazırla
        """
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Binary classification için sadece 2 sınıf al (setosa vs. non-setosa)
        binary_y = (y != 0).astype(int).reshape(-1, 1)
        
        self.feature_names = iris.feature_names
        self.target_names = ['Setosa', 'Non-Setosa']
        
        print(f"Iris veri seti yüklendi:")
        print(f"  Örnekler: {X.shape[0]}")
        print(f"  Özellikler: {X.shape[1]}")
        print(f"  Özellik isimleri: {self.feature_names}")
        
        return X, binary_y
    
    def load_digits_dataset(self, binary=True):
        """
        El yazısı rakam veri setini yükle
        
        Args:
            binary: True ise binary classification (0 vs diğerleri)
        """
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        if binary:
            # Binary classification: 0 vs diğerleri
            binary_y = (y == 0).astype(int).reshape(-1, 1)
            self.target_names = ['Rakam 0', 'Diğer Rakamlar']
            print(f"Digits veri seti yüklendi (Binary):")
            print(f"  Örnekler: {X.shape[0]}")
            print(f"  Özellikler: {X.shape[1]} (8x8 piksel)")
            return X, binary_y
        else:
            self.target_names = [f'Rakam {i}' for i in range(10)]
            print(f"Digits veri seti yüklendi (Multi-class):")
            print(f"  Örnekler: {X.shape[0]}")
            print(f"  Özellikler: {X.shape[1]} (8x8 piksel)")
            print(f"  Sınıflar: 10 (0-9 rakamları)")
            return X, y
    
    def load_csv_data(self, filepath, target_column, feature_columns=None):
        """
        CSV dosyasından veri yükle
        
        Args:
            filepath: CSV dosya yolu
            target_column: Hedef değişken sütunu
            feature_columns: Özellik sütunları (None ise hedef hariç hepsi)
        """
        try:
            df = pd.read_csv(filepath)
            print(f"CSV dosyası yüklendi: {filepath}")
            print(f"  Satırlar: {df.shape[0]}")
            print(f"  Sütunlar: {df.shape[1]}")
            
            # Hedef değişken
            y = df[target_column].values
            
            # Özellikler
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
            
            X = df[feature_columns].values
            self.feature_names = feature_columns
            
            # Kategorik hedef değişkeni encode et
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
                self.target_names = self.label_encoder.classes_.tolist()
            
            # Binary classification için reshape
            if len(np.unique(y)) == 2:
                y = y.reshape(-1, 1)
            
            print(f"  Özellikler: {X.shape[1]}")
            print(f"  Sınıflar: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            print(f"CSV yükleme hatası: {e}")
            return None, None
    
    def preprocess_data(self, X_train, X_val=None, X_test=None, normalize=True):
        """
        Veriyi ön işle
        
        Args:
            X_train: Eğitim verisi
            X_val: Validation verisi
            X_test: Test verisi
            normalize: Normalizasyon yapılsın mı
            
        Returns:
            İşlenmiş veriler
        """
        if normalize:
            # Eğitim verisine fit et
            X_train_processed = self.scaler.fit_transform(X_train)
            
            results = [X_train_processed]
            
            if X_val is not None:
                X_val_processed = self.scaler.transform(X_val)
                results.append(X_val_processed)
            
            if X_test is not None:
                X_test_processed = self.scaler.transform(X_test)
                results.append(X_test_processed)
            
            print("Veri normalizasyonu tamamlandı.")
            return tuple(results) if len(results) > 1 else X_train_processed
        else:
            return X_train, X_val, X_test
    
    def add_noise(self, X, noise_level=0.1):
        """
        Veriye gürültü ekle (data augmentation)
        
        Args:
            X: Orijinal veri
            noise_level: Gürültü seviyesi
            
        Returns:
            Gürültülü veri
        """
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        return X_noisy
    
    def create_feature_combinations(self, X, degree=2):
        """
        Özellik kombinasyonları oluştur (polynomial features)
        
        Args:
            X: Orijinal özellikler
            degree: Polinom derecesi
            
        Returns:
            Genişletilmiş özellik matrisi
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        print(f"Özellik kombinasyonları oluşturuldu:")
        print(f"  Orijinal özellikler: {X.shape[1]}")
        print(f"  Yeni özellikler: {X_poly.shape[1]}")
        
        return X_poly
    
    def visualize_data(self, X, y, title="Veri Dağılımı", save_path=None):
        """
        Veriyi görselleştir
        
        Args:
            X: Özellikler (ilk 2 özellik kullanılır)
            y: Hedef değişken
            title: Grafik başlığı
            save_path: Kayıt yolu
        """
        if X.shape[1] < 2:
            print("Görselleştirme için en az 2 özellik gerekli!")
            return
        
        plt.figure(figsize=(10, 6))
        
        # İlk iki özelliği kullan
        unique_labels = np.unique(y.flatten())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, label in enumerate(unique_labels):
            mask = (y.flatten() == label)
            label_name = self.target_names[int(label)] if self.target_names else f'Sınıf {label}'
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       label=label_name, 
                       alpha=0.7)
        
        plt.xlabel(self.feature_names[0] if self.feature_names else 'Özellik 1')
        plt.ylabel(self.feature_names[1] if self.feature_names else 'Özellik 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def get_data_info(self, X, y):
        """
        Veri hakkında bilgi ver
        
        Args:
            X: Özellikler
            y: Hedef değişken
        """
        print("\n" + "="*40)
        print("VERİ BİLGİLERİ")
        print("="*40)
        print(f"Örnek sayısı: {X.shape[0]}")
        print(f"Özellik sayısı: {X.shape[1]}")
        print(f"Hedef değişken şekli: {y.shape}")
        
        # Sınıf dağılımı
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nSınıf dağılımı:")
        for label, count in zip(unique, counts):
            label_name = self.target_names[int(label)] if self.target_names else f'Sınıf {label}'
            percentage = (count / len(y)) * 100
            print(f"  {label_name}: {count} (%{percentage:.1f})")
        
        # Özellik istatistikleri
        print(f"\nÖzellik istatistikleri:")
        print(f"  Ortalama: {np.mean(X, axis=0)[:3]}..." if X.shape[1] > 3 else f"  Ortalama: {np.mean(X, axis=0)}")
        print(f"  Standart sapma: {np.std(X, axis=0)[:3]}..." if X.shape[1] > 3 else f"  Standart sapma: {np.std(X, axis=0)}")
        print(f"  Min: {np.min(X, axis=0)[:3]}..." if X.shape[1] > 3 else f"  Min: {np.min(X, axis=0)}")
        print(f"  Max: {np.max(X, axis=0)[:3]}..." if X.shape[1] > 3 else f"  Max: {np.max(X, axis=0)}")


def create_sample_datasets():
    """
    Örnek veri setleri oluştur ve kaydet
    """
    processor = DataProcessor()
    
    # Örnek klasörü oluştur
    examples_dir = "data/examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # 1. Basit XOR verisi
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    xor_data = pd.DataFrame(X_xor, columns=['X1', 'X2'])
    xor_data['y'] = y_xor.flatten()
    xor_data.to_csv(os.path.join(examples_dir, "xor_data.csv"), index=False)
    
    # 2. Daha karmaşık sentetik veri
    X_complex, y_complex = processor.generate_synthetic_data(
        n_samples=1000, n_features=5, n_classes=2, random_state=42
    )
    
    complex_data = pd.DataFrame(X_complex, columns=[f'feature_{i}' for i in range(5)])
    complex_data['target'] = y_complex.flatten()
    complex_data.to_csv(os.path.join(examples_dir, "synthetic_data.csv"), index=False)
    
    # 3. Iris verisi
    X_iris, y_iris = processor.load_iris_dataset()
    iris_data = pd.DataFrame(X_iris, columns=processor.feature_names)
    iris_data['is_setosa'] = y_iris.flatten()
    iris_data.to_csv(os.path.join(examples_dir, "iris_binary.csv"), index=False)
    
    print("Örnek veri setleri oluşturuldu:")
    print(f"  - {examples_dir}/xor_data.csv")
    print(f"  - {examples_dir}/synthetic_data.csv")
    print(f"  - {examples_dir}/iris_binary.csv")


if __name__ == "__main__":
    # Örnek kullanım
    processor = DataProcessor()
    
    # Sentetik veri oluştur
    X, y = processor.generate_synthetic_data(n_samples=500, n_features=2)
    processor.get_data_info(X, y)
    processor.visualize_data(X, y, "Sentetik Veri")
    
    # Örnek veri setlerini oluştur
    create_sample_datasets()
