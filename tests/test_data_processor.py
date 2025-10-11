"""
Data Processor Testleri
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """DataProcessor sınıfı için kapsamlı testler"""
    
    def setUp(self):
        """Test kurulumu"""
        self.processor = DataProcessor()
        
    def test_synthetic_data_generation(self):
        """Sentetik veri üretimi detaylı testleri"""
        # Farklı parametrelerle test
        test_cases = [
            (100, 3, 2),
            (500, 5, 2),
            (50, 2, 2)
        ]
        
        for n_samples, n_features, n_classes in test_cases:
            with self.subTest(samples=n_samples, features=n_features):
                X, y = self.processor.generate_synthetic_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    random_state=42
                )
                
                # Boyut kontrolleri
                self.assertEqual(X.shape, (n_samples, n_features))
                self.assertEqual(y.shape, (n_samples, 1))
                
                # Sınıf sayısı kontrolü
                unique_classes = np.unique(y)
                self.assertEqual(len(unique_classes), n_classes)
                
                # Veri tipı kontrolü
                self.assertTrue(np.issubdtype(X.dtype, np.number))
                self.assertTrue(np.issubdtype(y.dtype, np.integer))
                
    def test_iris_dataset_loading(self):
        """Iris veri seti yükleme testleri"""
        X, y = self.processor.load_iris_dataset()
        
        # Boyut kontrolleri
        self.assertEqual(X.shape[1], 4)  # 4 özellik
        self.assertEqual(y.shape[1], 1)  # Binary classification
        self.assertEqual(X.shape[0], 150)  # 150 örnek
        
        # Sınıf kontrolü
        unique_classes = np.unique(y)
        self.assertEqual(len(unique_classes), 2)  # Binary
        self.assertIn(0, unique_classes)
        self.assertIn(1, unique_classes)
        
        # Feature names kontrolü
        self.assertIsNotNone(self.processor.feature_names)
        self.assertEqual(len(self.processor.feature_names), 4)
        
    def test_digits_dataset_loading(self):
        """Digits veri seti yükleme testleri"""
        # Binary classification
        X_bin, y_bin = self.processor.load_digits_dataset(binary=True)
        self.assertEqual(X_bin.shape[1], 64)  # 8x8 = 64 piksel
        self.assertEqual(y_bin.shape[1], 1)
        self.assertEqual(len(np.unique(y_bin)), 2)  # Binary
        
        # Multi-class classification
        X_multi, y_multi = self.processor.load_digits_dataset(binary=False)
        self.assertEqual(X_multi.shape[1], 64)
        self.assertEqual(len(np.unique(y_multi)), 10)  # 10 sınıf (0-9)
        
    def test_csv_data_loading(self):
        """CSV veri yükleme testleri"""
        # Geçici CSV dosyası oluştur
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Test verisi yaz
            f.write("feature1,feature2,feature3,target\n")
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,5.0,6.0,1\n")
            f.write("7.0,8.0,9.0,0\n")
            f.write("10.0,11.0,12.0,1\n")
            temp_file = f.name
            
        try:
            # CSV'yi yükle
            X, y = self.processor.load_csv_data(temp_file, 'target')
            
            # Kontroller
            self.assertEqual(X.shape, (4, 3))  # 4 satır, 3 özellik
            self.assertEqual(y.shape, (4, 1))  # 4 satır, 1 hedef
            
            # Veri değerlerini kontrol et
            expected_X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
            expected_y = np.array([[0], [1], [0], [1]])
            
            np.testing.assert_array_equal(X, expected_X)
            np.testing.assert_array_equal(y, expected_y)
            
        finally:
            # Geçici dosyayı sil
            os.unlink(temp_file)
            
    def test_data_preprocessing(self):
        """Veri ön işleme testleri"""
        # Test verisi oluştur
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8], [9, 10]])
        
        # Normalizasyon ile
        X_train_norm, X_test_norm = self.processor.preprocess_data(
            X_train, X_test=X_test, normalize=True
        )
        
        # Eğitim setinin ortalaması ~0, std ~1 olmalı
        np.testing.assert_allclose(np.mean(X_train_norm, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_train_norm, axis=0), 1, atol=1e-10)
        
        # Normalizasyon olmadan
        X_train_raw, X_test_raw, _ = self.processor.preprocess_data(
            X_train, X_test=X_test, normalize=False
        )
        
        np.testing.assert_array_equal(X_train_raw, X_train)
        np.testing.assert_array_equal(X_test_raw, X_test)
        
    def test_noise_addition(self):
        """Gürültü ekleme testi"""
        X_original = np.array([[1, 2], [3, 4], [5, 6]])
        X_noisy = self.processor.add_noise(X_original, noise_level=0.1)
        
        # Boyut aynı kalmalı
        self.assertEqual(X_noisy.shape, X_original.shape)
        
        # Biraz farklı olmalı (gürültü eklenmiş)
        self.assertFalse(np.array_equal(X_original, X_noisy))
        
        # Ama çok farklı olmamalı
        diff = np.abs(X_original - X_noisy)
        self.assertTrue(np.all(diff < 1.0))  # Gürültü sınırlı olmalı
        
    def test_feature_combinations(self):
        """Özellik kombinasyonu testi"""
        X = np.array([[1, 2], [3, 4]])
        
        X_poly = self.processor.create_feature_combinations(X, degree=2)
        
        # Polinom özellikleri daha fazla sütun oluşturmalı
        self.assertGreater(X_poly.shape[1], X.shape[1])
        
        # Satır sayısı aynı kalmalı
        self.assertEqual(X_poly.shape[0], X.shape[0])
        
    def test_data_info_output(self):
        """Veri bilgisi çıktısı testi"""
        X, y = self.processor.generate_synthetic_data(n_samples=100, n_features=3)
        
        # Bu fonksiyon exception fırlatmamalı
        try:
            self.processor.get_data_info(X, y)
        except Exception as e:
            self.fail(f"get_data_info fonksiyonu hata verdi: {e}")
            
    def test_data_visualization(self):
        """Veri görselleştirme testi"""
        X, y = self.processor.generate_synthetic_data(n_samples=50, n_features=2)
        
        # Görselleştirme exception fırlatmamalı
        try:
            self.processor.visualize_data(X, y, title="Test")
        except Exception as e:
            # Grafik kütüphanesi yoksa veya display problemi varsa geç
            if "display" in str(e).lower() or "backend" in str(e).lower():
                self.skipTest("Grafik display problemi")
            else:
                self.fail(f"Görselleştirme hatası: {e}")
                
    def test_invalid_csv_loading(self):
        """Geçersiz CSV yükleme testi"""
        # Var olmayan dosya
        X, y = self.processor.load_csv_data("nonexistent.csv", "target")
        self.assertIsNone(X)
        self.assertIsNone(y)


def run_data_tests():
    """Veri işleme testlerini çalıştır"""
    print("📊 Veri İşleme Testleri Başlatılıyor...")
    print("=" * 50)
    
    # Test suite oluştur
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataProcessor)
    
    # Testleri çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Sonuçları yazdır
    print("\n" + "=" * 50)
    print("DATA PROCESSOR TEST SONUÇLARI:")
    print(f"✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Başarısız: {len(result.failures)}")
    print(f"💥 Hata: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 Tüm veri işleme testleri başarılı!")
    else:
        print("\n⚠️ Bazı veri işleme testleri başarısız!")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    run_data_tests()
