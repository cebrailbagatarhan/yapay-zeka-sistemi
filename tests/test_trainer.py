"""ModernLLM Trainer eğitim döngüsü testleri."""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modern_llm.config import ModelConfig, TrainingConfig
from modern_llm.model.transformer import ModernLLMForCausalLM
from modern_llm.training.trainer import Trainer


def tiny_model():
    config = ModelConfig(
        vocab_size=300,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    return ModernLLMForCausalLM(config)


def token_dataset(size):
    return [
        {
            "input_ids": torch.tensor([14, 15, 16, 17], dtype=torch.long),
            "labels": torch.tensor([14, 15, 16, 17], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
        }
        for _ in range(size)
    ]


class TrainerLoopTests(unittest.TestCase):
    def build_trainer(self, dataset_size, batch_size, accumulation):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                num_epochs=1,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                max_seq_length=8,
                mixed_precision="fp32",
                num_workers=0,
                logging_steps=100,
                eval_steps=100,
                save_steps=100,
                output_dir=tmpdir,
            )
            trainer = Trainer(
                tiny_model(),
                token_dataset(dataset_size),
                training_config=config,
            )
        # Dosya sistemi testin odağı değil; checkpoint/log yazımını atla.
        trainer.save_checkpoint = lambda name: None
        trainer._save_training_log = lambda: None
        return trainer

    def run_silently(self, trainer):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()

    def test_partial_accumulation_window_is_flushed(self):
        trainer = self.build_trainer(dataset_size=3, batch_size=1, accumulation=2)
        self.assertEqual(trainer.total_steps, 2)
        self.run_silently(trainer)
        self.assertEqual(trainer.global_step, 2)

    def test_dataset_smaller_than_batch_still_trains(self):
        trainer = self.build_trainer(dataset_size=2, batch_size=4, accumulation=1)
        self.assertEqual(trainer.total_steps, 1)
        self.run_silently(trainer)
        self.assertEqual(trainer.global_step, 1)

    def test_empty_dataset_is_rejected(self):
        config = TrainingConfig(
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            mixed_precision="fp32",
            num_workers=0,
        )
        with self.assertRaises(ValueError):
            Trainer(tiny_model(), [], training_config=config)


if __name__ == "__main__":
    unittest.main()
