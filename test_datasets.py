from src.model_trainer import ModelTrainer

print("="*60)
print("📚 YAPAY ZEKA EĞİTİM SİSTEMİ - DATASET LİSTESİ")
print("="*60)

trainer = ModelTrainer()

print(f"\n✅ Toplam Dataset: {len(trainer.training_config['datasets'])}")
print("\n📋 Dataset Detayları:\n")

for i, dataset in enumerate(trainer.training_config['datasets'], 1):
    print(f"{i}. {dataset['name']}")
    print(f"   • Max Samples: {dataset['max_samples']}")
    print(f"   • Weight: {dataset['weight']}")
    print(f"   • Split: {dataset['split']}")
    print()

print("="*60)
print("🎯 Bu dataset'ler eğitimde kullanılacak!")
print("="*60)
