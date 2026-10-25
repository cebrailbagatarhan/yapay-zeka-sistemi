"""Tokenizer ve veri pipeline'ı regresyon testleri."""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modern_llm.tokenizer import CHAT_TEMPLATE, ModernTokenizer
from modern_llm.training.dataset import (
    ChatDataset,
    TextDataset,
    create_data_collator,
)
from modern_llm.utils import chunk_text


class ByteFallbackTokenizerTests(unittest.TestCase):
    def test_unicode_newline_and_control_characters_round_trip(self):
        tokenizer = ModernTokenizer()
        text = "Merhaba\nİstanbul 🌍\tAnkara"
        ids = tokenizer.encode(text)
        self.assertEqual(
            tokenizer.decode(ids, skip_special_tokens=False),
            text,
        )
        self.assertEqual(tokenizer.decode(ids), text)

    def test_thinking_tokens_round_trip_when_requested(self):
        tokenizer = ModernTokenizer()
        text = "<think>iç düşünce</think>sonuç"
        ids = tokenizer.encode(text)
        self.assertEqual(
            tokenizer.decode(ids, skip_special_tokens=False),
            text,
        )
        self.assertEqual(tokenizer.decode(ids), "iç düşüncesonuç")

    def test_save_and_load_fallback_tokenizer(self):
        tokenizer = ModernTokenizer()
        text = "Türkçe 🌍\nveri"
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(tmpdir)
            loaded = ModernTokenizer(model_path=tmpdir)
            self.assertEqual(
                loaded.decode(loaded.encode(text), skip_special_tokens=False),
                text,
            )

    def test_missing_tokenizer_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            ModernTokenizer(model_path="does-not-exist-tokenizer")

    def test_sentencepiece_training_and_reload(self):
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            self.skipTest("sentencepiece kurulu değil")

        texts = [
            "Türkçe yapay zeka ve dil modelleri üzerine bir çalışma. " * 10,
            "Makine öğrenmesi veriyle öğrenir; derin öğrenme çok katmanlıdır. " * 10,
            "Doğal dil işleme metni anlamlandırmayı amaçlar. " * 10,
        ]
        text = "Merhaba dünya 🌍\nTürkçe model."
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = ModernTokenizer(vocab_size=300)
            trained = tokenizer.train(texts, vocab_size=300, output_path=tmpdir)
            self.assertTrue(trained)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer.model")))
            self.assertEqual(
                tokenizer.decode(
                    tokenizer.encode(text),
                    skip_special_tokens=False,
                ),
                text,
            )

            loaded = ModernTokenizer(model_path=tmpdir)
            self.assertEqual(
                loaded.decode(loaded.encode(text), skip_special_tokens=False),
                text,
            )


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ModernTokenizer()

    def test_text_dataset_labels_are_not_double_shifted(self):
        dataset = TextDataset(["abcdef"], self.tokenizer, max_length=4)
        self.assertGreater(len(dataset), 0)
        item = dataset[0]
        self.assertTrue(torch.equal(item["input_ids"], item["labels"]))

    def test_chat_dataset_masks_every_non_assistant_segment(self):
        messages = [
            {"role": "system", "content": "Sistem"},
            {"role": "user", "content": "ilk soru"},
            {"role": "assistant", "content": "ilk cevap"},
            {"role": "user", "content": "ikinci soru"},
            {"role": "assistant", "content": "son cevap"},
        ]
        dataset = ChatDataset(
            [{"messages": messages}],
            self.tokenizer,
            max_length=512,
        )
        item = dataset[0]
        full_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(messages),
            add_bos=True,
            add_eos=True,
        )

        for role, content in [("system", "Sistem"), ("user", "ikinci soru")]:
            segment = self.tokenizer.encode(
                CHAT_TEMPLATE[role].format(content=content)
            )
            start = next(
                index
                for index in range(len(full_ids) - len(segment) + 1)
                if full_ids[index:index + len(segment)] == segment
            )
            self.assertTrue(
                torch.all(item["labels"][start:start + len(segment)] == -100)
            )

        self.assertGreater(int((item["labels"] != -100).sum()), 0)
        self.assertEqual(item["labels"][0].item(), -100)

    def test_collator_preserves_integer_dtypes(self):
        collate = create_data_collator(self.tokenizer, max_length=8)
        batch = collate([
            {
                "input_ids": torch.tensor([14, 15], dtype=torch.long),
                "labels": torch.tensor([14, 15], dtype=torch.long),
                "attention_mask": torch.ones(2, dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([16], dtype=torch.long),
                "labels": torch.tensor([16], dtype=torch.long),
                "attention_mask": torch.ones(1, dtype=torch.long),
            },
        ])
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["attention_mask"].dtype, torch.long)

    def test_collator_rejects_empty_batch(self):
        collate = create_data_collator(self.tokenizer)
        with self.assertRaises(ValueError):
            collate([])

    def test_chunk_text_rejects_infinite_loop_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", max_length=2, overlap=2)
        with self.assertRaises(ValueError):
            chunk_text("abc", max_length=0)


if __name__ == "__main__":
    unittest.main()
