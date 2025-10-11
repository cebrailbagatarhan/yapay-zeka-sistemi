"""
Neural Network Model Testleri
"""

import unittest
import numpy as np
import sys
import os

# Proje modüllerini import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import NeuralNetwork, ModelTrainer, DataProcessor


class TestNeuralNetwork(unittest.TestCase):
    """Neural Network sınıfı için testler"""
    
    def setUp(self):
        """Test kurulumu"""
        self.model = NeuralNetwork([2, 3, 1], activation='sigmoid', learning_rate=0.1)
        
    def test_model_initialization(self):
        """Model başlatma testi"""
        self.assertEqual(len(self.model.weights), 2)
        self.assertEqual(len(self.model.biases), 2)
        self.assertEqual(self.model.layers, [2, 3, 1])
        
    def test_forward_pass(self):
        """İleri beslenme testi"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        output = self.model.forward(X)
        
        self.assertEqual(output.shape, (4, 1))
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        
    def test_xor_learning(self):
        """XOR problemini öğrenebilme testi"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Model eğit
        losses = self.model.train(X, y, epochs=1000, verbose=False)
        
        # Doğruluk testi
        accuracy = self.model.evaluate(X, y)
        self.assertGreater(accuracy, 0.8, "Model XOR problemini yeterince öğrenemedi")
        
    def test_prediction(self):
        """Tahmin fonksiyonu testi"""
        X = np.array([[0.5, 0.5]])
        
        # Olasılık tahmini
        proba = self.model.predict_proba(X)
        self.assertEqual(proba.shape, (1, 1))
        self.assertTrue(0 <= proba[0][0] <= 1)
        
        # Sınıf tahmini
        pred = self.model.predict(X)
        self.assertIn(pred[0][0], [0, 1])


class TestModelTrainer(unittest.TestCase):
    """ModelTrainer sınıfı için testler"""
    
    def setUp(self):
        """Test kurulumu"""
        self.model = NeuralNetwork([2, 4, 1])
        self.trainer = ModelTrainer(self.model)
        
        # Test verisi
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        
    def test_data_preparation(self):
        """Veri hazırlama testi"""
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
            self.X, self.y, test_size=0.25, validation_size=0.25
        )
        
        # Boyut kontrolleri
        total_samples = len(self.X)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total_samples)
        
    def test_normalization(self):
        """Normalizasyon testi"""
        X_normalized = self.trainer.normalize_data(self.X)
        
        # Normalize edilmiş verinin ortalama ve standart sapması
        mean = np.mean(X_normalized, axis=0)
        std = np.std(X_normalized, axis=0)
        
        # Ortalama ~0, std ~1 olmalı
        np.testing.assert_allclose(mean, 0, atol=1e-10)
        np.testing.assert_allclose(std, 1, atol=1e-10)


class TestDataProcessor(unittest.TestCase):
    """DataProcessor sınıfı için testler"""
    
    def setUp(self):
        """Test kurulumu"""
        self.processor = DataProcessor()
        
    def test_synthetic_data_generation(self):
        """Sentetik veri üretimi testi"""
        X, y = self.processor.generate_synthetic_data(
            n_samples=100, n_features=5, n_classes=2
        )
        
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(y.shape, (100, 1))
        self.assertEqual(len(np.unique(y)), 2)
        
    def test_iris_dataset_loading(self):
        """Iris veri seti yükleme testi"""
        X, y = self.processor.load_iris_dataset()
        
        self.assertEqual(X.shape[1], 4)  # 4 özellik
        self.assertEqual(y.shape[1], 1)  # Binary classification
        self.assertEqual(len(np.unique(y)), 2)  # 2 sınıf
        
    def test_data_info_function(self):
        """Veri bilgisi fonksiyonu testi"""
        X, y = self.processor.generate_synthetic_data(n_samples=50, n_features=3)
        
        # Bu fonksiyon hata vermemeli
        try:
            self.processor.get_data_info(X, y)
        except Exception as e:
            self.fail(f"get_data_info fonksiyonu hata verdi: {e}")


class TestModelSaveLoad(unittest.TestCase):
    """Model kaydetme/yükleme testleri"""
    
    def setUp(self):
        """Test kurulumu"""
        self.model = NeuralNetwork([2, 3, 1])
        self.test_file = "test_model.json"
        
    def tearDown(self):
        """Test temizleme"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def test_save_and_load_model(self):
        """Model kaydetme ve yükleme testi"""
        # Orijinal ağırlıkları kaydet
        original_weights = [w.copy() for w in self.model.weights]
        original_biases = [b.copy() for b in self.model.biases]
        
        # Modeli kaydet
        self.model.save_model(self.test_file)
        self.assertTrue(os.path.exists(self.test_file))
        
        # Yeni model oluştur ve yükle
        new_model = NeuralNetwork([2, 3, 1])
        new_model.load_model(self.test_file)
        
        # Ağırlıkların aynı olduğunu kontrol et
        for orig_w, new_w in zip(original_weights, new_model.weights):
            np.testing.assert_array_equal(orig_w, new_w)
            
        for orig_b, new_b in zip(original_biases, new_model.biases):
            np.testing.assert_array_equal(orig_b, new_b)


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("🧪 Neural Network Testleri Başlatılıyor...")
    print("=" * 50)
    
    # Test suite oluştur
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Test sınıflarını ekle
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelSaveLoad))
    
    # Testleri çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Sonuçları yazdır
    print("\n" + "=" * 50)
    print("TEST SONUÇLARI:")
    print(f"✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Başarısız: {len(result.failures)}")
    print(f"💥 Hata: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 Tüm testler başarılı!")
    else:
        print("\n⚠️ Bazı testler başarısız!")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
