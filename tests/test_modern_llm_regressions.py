import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from modern_llm.config import ModelConfig, TrainingConfig
from modern_llm.model.attention import GroupedQueryAttention
from modern_llm.tokenizer import CHAT_TEMPLATE, ModernTokenizer
from modern_llm.training.dataset import ChatDataset, TextDataset
from modern_llm.training.trainer import Trainer


class ModernTokenizerRegressionTests(unittest.TestCase):
    def test_byte_fallback_round_trips_unicode(self):
        tokenizer = ModernTokenizer(vocab_size=32256)
        text = "Türkçe 😀 你好 — test"

        token_ids = tokenizer.encode(text)

        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertTrue(all(0 <= token_id < len(tokenizer) for token_id in token_ids))
        self.assertEqual(len(tokenizer), len(tokenizer.special_tokens) + 256)


class DatasetRegressionTests(unittest.TestCase):
    def test_text_dataset_does_not_shift_labels(self):
        tokenizer = ModernTokenizer()
        dataset = TextDataset(
            ["Bu bir causal language model test metnidir."],
            tokenizer,
            max_length=16,
        )

        example = dataset[0]

        self.assertTrue(torch.equal(example["input_ids"], example["labels"]))

    def test_chat_dataset_masks_every_non_assistant_turn(self):
        tokenizer = ModernTokenizer()
        messages = [
            {"role": "system", "content": "Sistem"},
            {"role": "user", "content": "Birinci soru"},
            {"role": "assistant", "content": "Birinci cevap"},
            {"role": "user", "content": "İkinci soru"},
            {"role": "assistant", "content": "İkinci cevap"},
        ]
        dataset = ChatDataset(
            [{"messages": messages}],
            tokenizer,
            max_length=256,
            mask_user_tokens=True,
        )
        example = dataset[0]
        input_ids = example["input_ids"].tolist()
        labels = example["labels"].tolist()

        self.assertEqual(labels[0], -100)  # BOS
        cursor = 1
        for message in messages:
            segment = CHAT_TEMPLATE[message["role"]].format(
                content=message["content"]
            )
            segment_ids = tokenizer.encode(segment)
            segment_labels = labels[cursor:cursor + len(segment_ids)]
            if message["role"] == "assistant":
                self.assertEqual(
                    segment_labels,
                    input_ids[cursor:cursor + len(segment_ids)],
                )
            else:
                self.assertTrue(all(label == -100 for label in segment_labels))
            cursor += len(segment_ids)

        self.assertEqual(labels[cursor], tokenizer.eos_token_id)
        self.assertTrue(all(label == -100 for label in labels[cursor + 1:]))


class AttentionRegressionTests(unittest.TestCase):
    def test_sdpa_and_manual_paths_share_causal_sliding_window_mask(self):
        config = ModelConfig(
            vocab_size=64,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
            attention_dropout=0.0,
            use_flash_attention=True,
            sliding_window=2,
        )
        attention = GroupedQueryAttention(config).eval()
        hidden_states = torch.randn(1, 4, config.hidden_size)

        sdpa_output, _, _ = attention(
            hidden_states,
            attention_mask=None,
            output_attentions=False,
        )
        manual_output, weights, _ = attention(
            hidden_states,
            attention_mask=None,
            output_attentions=True,
        )

        self.assertTrue(torch.allclose(sdpa_output, manual_output, atol=1e-5))
        for query_position in range(4):
            for key_position in range(4):
                allowed = (
                    key_position <= query_position
                    and key_position >= query_position - 1
                )
                if not allowed:
                    self.assertLess(
                        weights[0, :, query_position, key_position]
                        .abs()
                        .max()
                        .item(),
                        1e-6,
                    )


class _TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        value = torch.tensor([float(index + 1)])
        return {
            "input_ids": value.long(),
            "labels": value.long(),
            "attention_mask": torch.ones_like(value, dtype=torch.long),
        }


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, attention_mask=None, labels=None):
        target = labels.float().mean()
        return SimpleNamespace(loss=(self.weight - target).square())


class TrainerRegressionTests(unittest.TestCase):
    def test_partial_gradient_accumulation_group_is_stepped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = TrainingConfig(
                num_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=2,
                mixed_precision="fp32",
                num_workers=0,
                logging_steps=100,
                eval_steps=100,
                save_steps=100,
                output_dir=output_dir,
                use_wandb=False,
            )
            trainer = Trainer(
                model=_TinyModel(),
                train_dataset=_TinyDataset(),
                training_config=config,
            )

            trainer.train()

            self.assertEqual(trainer.total_steps, 3)
            self.assertEqual(trainer.global_step, 3)


if __name__ == "__main__":
    unittest.main()
