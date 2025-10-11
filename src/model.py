"""
Basit Neural Network Modeli
Bu dosya, sıfırdan yazılmış bir sinir ağı implementasyonu içerir.
"""

import numpy as np
import json
from typing import List, Tuple, Optional


class Activation:
    """Aktivasyon fonksiyonları"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid aktivasyon fonksiyonu"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid fonksiyonunun türevi"""
        return x * (1 - x)
    
    @staticmethod
    def relu(x):
        """ReLU aktivasyon fonksiyonu"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU fonksiyonunun türevi"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def tanh(x):
        """Tanh aktivasyon fonksiyonu"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Tanh fonksiyonunun türevi"""
        return 1 - x**2


class NeuralNetwork:
    """
    Basit Feed-Forward Neural Network
    
    Bu sınıf, çok katmanlı perceptron (MLP) implementasyonu içerir.
    Backpropagation algoritması ile eğitim yapar.
    """
    
    def __init__(self, layers: List[int], activation='sigmoid', learning_rate=0.1):
        """
        Neural Network'ü başlatır
        
        Args:
            layers: Her katmandaki nöron sayısı [input, hidden1, hidden2, ..., output]
            activation: Aktivasyon fonksiyonu ('sigmoid', 'relu', 'tanh')
            learning_rate: Öğrenme oranı
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Aktivasyon fonksiyonunu seç
        if activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        else:
            raise ValueError("Desteklenen aktivasyon fonksiyonları: 'sigmoid', 'relu', 'tanh'")
        
        # Ağırlıkları ve bias'ları başlat
        self.weights = []
        self.biases = []
        
        # Xavier/Glorot initialization
        for i in range(self.num_layers - 1):
            # Ağırlıkları rastgele başlat
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            self.weights.append(w)
            
            # Bias'ları sıfır olarak başlat
            b = np.zeros((1, layers[i+1]))
            self.biases.append(b)
    
    def forward(self, X):
        """
        İleri beslenme (Forward propagation)
        
        Args:
            X: Giriş verisi (n_samples, n_features)
            
        Returns:
            Çıkış tahminleri
        """
        self.a = [X]  # Aktivasyonları sakla
        self.z = []   # Ağırlıklı toplamları sakla
        
        current_input = X
        
        for i in range(self.num_layers - 1):
            # Ağırlıklı toplam: z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z.append(z)
            
            # Aktivasyon fonksiyonunu uygula
            if i == self.num_layers - 2:  # Son katman
                # Son katmanda sigmoid kullan (binary classification için)
                a = Activation.sigmoid(z)
            else:
                a = self.activation(z)
            
            self.a.append(a)
            current_input = a
        
        return self.a[-1]
    
    def backward(self, X, y, output):
        """
        Geri beslenme (Backpropagation)
        
        Args:
            X: Giriş verisi
            y: Gerçek etiketler
            output: Model çıkışı
        """
        m = X.shape[0]  # Örnek sayısı
        
        # Çıkış katmanından başlayarak hata hesapla
        delta = output - y
        
        # Gradyanları sakla
        dW = []
        db = []
        
        # Geriye doğru ilerle
        for i in range(self.num_layers - 2, -1, -1):
            # Ağırlık gradyanı
            dw = np.dot(self.a[i].T, delta) / m
            dW.insert(0, dw)
            
            # Bias gradyanı
            d_b = np.sum(delta, axis=0, keepdims=True) / m
            db.insert(0, d_b)
            
            if i > 0:  # İlk katman değilse
                # Bir önceki katmana hata propogasyonu
                if i == self.num_layers - 2:  # Son katmandan geliyorsa
                    delta = np.dot(delta, self.weights[i].T) * Activation.sigmoid_derivative(self.a[i])
                else:
                    delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.a[i])
        
        # Ağırlıkları güncelle
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Modeli eğit
        
        Args:
            X: Eğitim verisi
            y: Eğitim etiketleri
            epochs: Epoch sayısı
            verbose: İlerleme göster
            
        Returns:
            Eğitim kayıpları (loss history)
        """
        losses = []
        
        for epoch in range(epochs):
            # İleri beslenme
            output = self.forward(X)
            
            # Kayıp hesapla (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # Geri beslenme
            self.backward(X, y, output)
            
            # İlerleme göster
            if verbose and epoch % 100 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, X):
        """Tahmin yap"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Olasılık tahmini"""
        return self.forward(X)
    
    def evaluate(self, X, y):
        """Model performansını değerlendir"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        model_data = {
            'layers': self.layers,
            'learning_rate': self.learning_rate,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """Modeli yükle"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.layers = model_data['layers']
        self.learning_rate = model_data['learning_rate']
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.num_layers = len(self.layers)
        
        print(f"Model yüklendi: {filepath}")


if __name__ == "__main__":
    # Basit test
    print("Neural Network Test")
    print("=" * 30)
    
    # XOR problemi - klasik test
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Model oluştur
    model = NeuralNetwork([2, 4, 1], activation='sigmoid', learning_rate=0.5)
    
    print("Eğitim öncesi tahminler:")
    initial_predictions = model.predict_proba(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Prediction: {initial_predictions[i][0]:.4f}, Expected: {y[i][0]}")
    
    # Modeli eğit
    print("\nModel eğitiliyor...")
    losses = model.train(X, y, epochs=1000, verbose=False)
    
    print("\nEğitim sonrası tahminler:")
    final_predictions = model.predict_proba(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Prediction: {final_predictions[i][0]:.4f}, Expected: {y[i][0]}")
    
    # Doğruluk hesapla
    accuracy = model.evaluate(X, y)
    print(f"\nDoğruluk: {accuracy:.4f}")
