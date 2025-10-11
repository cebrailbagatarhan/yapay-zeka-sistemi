"""
Yapay Zeka Modeli Paketi
Bu paket, sıfırdan yapay zeka modeli oluşturmak için gerekli tüm araçları içerir.
"""

from .model import NeuralNetwork, Activation
from .trainer import ModelTrainer
from .data_processor import DataProcessor, create_sample_datasets
from .utils import (
    plot_decision_boundary, 
    calculate_metrics, 
    print_metrics,
    plot_learning_curves,
    compare_models,
    save_experiment_results,
    load_experiment_results,
    create_model_summary,
    Timer,
    ensure_directory
)

__version__ = "1.0.0"
__author__ = "Yapay Zeka Öğrencisi"

# Ana bileşenleri dışa aktar
__all__ = [
    'NeuralNetwork',
    'Activation', 
    'ModelTrainer',
    'DataProcessor',
    'create_sample_datasets',
    'plot_decision_boundary',
    'calculate_metrics',
    'print_metrics',
    'plot_learning_curves',
    'compare_models',
    'save_experiment_results',
    'load_experiment_results',
    'create_model_summary',
    'Timer',
    'ensure_directory'
]
