"""
Yardımcı Fonksiyonlar
Bu dosya, proje genelinde kullanılan yardımcı fonksiyonları içerir.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import List, Dict, Any
import json


def progress_bar(current, total, bar_length=50, prefix="İlerleme"):
    """
    Terminal için ilerleme çubuğu
    
    Args:
        current: Mevcut ilerleme
        total: Toplam değer
        bar_length: Çubuk uzunluğu
        prefix: Önek metin
    """
    percent = float(current) * 100 / total
    arrow = '█' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    print(f'\r{prefix}: [{arrow}{spaces}] {percent:.1f}%', end='', flush=True)
    
    if current == total:
        print()  # Yeni satıra geç


def plot_decision_boundary(model, X, y, resolution=100, title="Karar Sınırı"):
    """
    2D veri için karar sınırını çiz
    
    Args:
        model: Eğitilmiş model
        X: Özellikler (2D)
        y: Hedef değişken
        resolution: Grid çözünürlüğü
        title: Grafik başlığı
    """
    if X.shape[1] != 2:
        print("Karar sınırı çizimi sadece 2D veri için mümkün!")
        return
    
    # Grid oluştur
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Grid noktalarında tahmin yap
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Grafik çiz
    plt.figure(figsize=(10, 8))
    
    # Karar sınırı
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Tahmin Olasılığı')
    
    # Veri noktaları
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), 
                         cmap='RdYlBu', edgecolors='black', linewidth=1)
    
    plt.title(title)
    plt.xlabel('Özellik 1')
    plt.ylabel('Özellik 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Detaylı performans metrikleri hesapla
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        y_proba: Tahmin olasılıkları (opsiyonel)
        
    Returns:
        Metrik sözlüğü
    """
    # Temel metrikler
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Hesaplamalar
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    }
    
    # AUC hesapla (eğer olasılık verilmişse)
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_proba)
            metrics['auc'] = auc
        except:
            pass
    
    return metrics


def print_metrics(metrics):
    """
    Metrikleri güzel formatta yazdır
    
    Args:
        metrics: calculate_metrics() fonksiyonundan dönen metrikler
    """
    print("\n" + "="*40)
    print("PERFORMANS METRİKLERİ")
    print("="*40)
    print(f"Doğruluk (Accuracy):    {metrics['accuracy']:.4f}")
    print(f"Kesinlik (Precision):   {metrics['precision']:.4f}")
    print(f"Duyarlılık (Recall):    {metrics['recall']:.4f}")
    print(f"F1 Skoru:               {metrics['f1_score']:.4f}")
    print(f"Özgüllük (Specificity): {metrics['specificity']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC Skoru:              {metrics['auc']:.4f}")
    
    print("\nKarışıklık Matrisi:")
    cm = metrics['confusion_matrix']
    print(f"  TP: {cm['tp']:<4} FP: {cm['fp']}")
    print(f"  FN: {cm['fn']:<4} TN: {cm['tn']}")


def plot_learning_curves(train_scores, val_scores, metric_name="Loss", 
                        save_path=None):
    """
    Öğrenme eğrilerini çiz
    
    Args:
        train_scores: Eğitim skorları
        val_scores: Validation skorları
        metric_name: Metrik ismi
        save_path: Kayıt yolu
    """
    epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', linewidth=2)
    plt.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', linewidth=2)
    
    plt.title(f'Learning Curves - {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # En iyi noktayı işaretle
    if metric_name.lower() == 'loss':
        best_epoch = np.argmin(val_scores) + 1
        best_score = min(val_scores)
    else:
        best_epoch = np.argmax(val_scores) + 1
        best_score = max(val_scores)
    
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    plt.text(best_epoch, best_score, f'  En iyi: Epoch {best_epoch}', 
             verticalalignment='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.tight_layout()
    plt.show()


def compare_models(models_results: Dict[str, Dict], metric='accuracy'):
    """
    Birden fazla modeli karşılaştır
    
    Args:
        models_results: Model ismi -> sonuçlar sözlüğü
        metric: Karşılaştırma metriği
    """
    model_names = list(models_results.keys())
    scores = [models_results[name][metric] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    
    # En iyi modeli vurgula
    best_idx = np.argmax(scores)
    bars[best_idx].set_color('darkgreen')
    
    plt.title(f'Model Karşılaştırması - {metric.title()}')
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)
    
    # Değerleri çubukların üzerine yaz
    for i, (name, score) in enumerate(zip(model_names, scores)):
        plt.text(i, score + 0.01, f'{score:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # En iyi modeli yazdır
    best_model = model_names[best_idx]
    best_score = scores[best_idx]
    print(f"\nEn iyi model: {best_model} ({metric}: {best_score:.4f})")


def save_experiment_results(results: Dict[str, Any], filepath: str):
    """
    Deney sonuçlarını JSON formatında kaydet
    
    Args:
        results: Sonuçlar sözlüğü
        filepath: Kayıt dosya yolu
    """
    # NumPy array'leri liste'ye çevir
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"Deney sonuçları kaydedildi: {filepath}")


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """
    Deney sonuçlarını JSON'dan yükle
    
    Args:
        filepath: Dosya yolu
        
    Returns:
        Sonuçlar sözlüğü
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Deney sonuçları yüklendi: {filepath}")
        return results
    except Exception as e:
        print(f"Dosya yükleme hatası: {e}")
        return {}


def create_model_summary(model, X_sample):
    """
    Model özeti oluştur
    
    Args:
        model: Neural network modeli
        X_sample: Örnek veri (boyut tespiti için)
    """
    print("\n" + "="*50)
    print("MODEL ÖZETİ")
    print("="*50)
    
    print(f"Model Mimarisi:")
    total_params = 0
    
    for i, (layer_size, next_layer_size) in enumerate(zip(model.layers[:-1], model.layers[1:])):
        weights_count = layer_size * next_layer_size
        bias_count = next_layer_size
        layer_params = weights_count + bias_count
        total_params += layer_params
        
        print(f"  Katman {i+1}: {layer_size} -> {next_layer_size}")
        print(f"    Ağırlıklar: {weights_count}")
        print(f"    Bias: {bias_count}")
        print(f"    Toplam: {layer_params}")
        print()
    
    print(f"Toplam Parametre Sayısı: {total_params:,}")
    print(f"Giriş Boyutu: {X_sample.shape[1:]}")
    print(f"Çıkış Boyutu: {model.layers[-1]}")
    print(f"Öğrenme Oranı: {model.learning_rate}")


class Timer:
    """
    Zaman ölçümü için yardımcı sınıf
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Zamanlayıcıyı başlat"""
        self.start_time = time.time()
        print("Zamanlayıcı başlatıldı...")
    
    def stop(self):
        """Zamanlayıcıyı durdur"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"Geçen süre: {elapsed:.2f} saniye")
        return elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def ensure_directory(path):
    """
    Dizinin var olduğundan emin ol, yoksa oluştur
    
    Args:
        path: Dizin yolu
    """
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Yardımcı fonksiyonların test edilmesi
    print("Yardımcı fonksiyonlar test ediliyor...")
    
    # Timer testi
    with Timer():
        time.sleep(1)
    
    # Metrik hesaplama testi
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics)
    
    print("Test tamamlandı!")
