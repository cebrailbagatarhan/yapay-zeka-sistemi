"""ModernLLM çekirdek davranış testleri."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modern_llm.config import ModelConfig, TrainingConfig, get_config_for_gpu
from modern_llm.inference.generator import (
    ChatInterface,
    GenerationConfig,
    TextGenerator,
)
from modern_llm.model.transformer import ModernLLMModel, ModernLLMForCausalLM
from modern_llm.tokenizer import ModernTokenizer


def tiny_config(**overrides):
    values = {
        "vocab_size": 300,
        "hidden_size": 16,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
    }
    values.update(overrides)
    return ModelConfig(**values)


class ModelCorrectnessTests(unittest.TestCase):
    def test_causal_mask_only_allows_past_and_current_tokens(self):
        model = ModernLLMModel(tiny_config())
        mask = model._make_causal_mask(
            (1, 4), torch.float32, torch.device("cpu")
        )
        allowed = mask[0, 0] == 0
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(allowed, expected))
        self.assertEqual(mask.dtype, torch.float32)

    def test_cached_decode_matches_full_forward(self):
        torch.manual_seed(7)
        model = ModernLLMForCausalLM(tiny_config()).eval()
        input_ids = torch.tensor([[14, 15, 16, 17]])

        with torch.no_grad():
            full_logits = model(input_ids).logits
            prefix = model(input_ids[:, :2], use_cache=True)
            suffix = model(
                input_ids[:, 2:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(
            suffix.logits,
            full_logits[:, 2:],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_save_and_load_preserves_logits(self):
        torch.manual_seed(11)
        model = ModernLLMForCausalLM(tiny_config()).eval()
        input_ids = torch.tensor([[14, 15, 16]])
        with torch.no_grad():
            expected = model(input_ids).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = ModernLLMForCausalLM.from_pretrained(tmpdir).eval()
            with torch.no_grad():
                actual = loaded(input_ids).logits

        torch.testing.assert_close(actual, expected)

    def test_generation_smoke_with_modern_tokenizer(self):
        torch.manual_seed(3)
        tokenizer = ModernTokenizer()
        generator = TextGenerator(
            ModernLLMForCausalLM(tiny_config()).eval(),
            tokenizer,
            device="cpu",
        )
        output = generator.generate("Merhaba", max_new_tokens=2, do_sample=False)
        self.assertIsInstance(output, str)

    def test_zero_temperature_uses_greedy_decoding(self):
        tokenizer = ModernTokenizer()
        generator = TextGenerator(
            ModernLLMForCausalLM(tiny_config()).eval(),
            tokenizer,
            device="cpu",
        )
        logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])
        config = GenerationConfig(temperature=0.0, top_k=0, top_p=1.0)
        token = generator._sample_token(logits, torch.tensor([[14]]), config)
        self.assertEqual(token.item(), 1)

    def test_generate_with_thinking_keeps_think_tags(self):
        tokenizer = ModernTokenizer()
        prefix_ids = tokenizer.encode("ok")
        response_ids = tokenizer.encode("cevap")
        targets = (
            [tokenizer.think_start_id]
            + [tokenizer.special_tokens["\n"]]
            + prefix_ids
            + [tokenizer.think_end_id]
            + response_ids
            + [tokenizer.eos_token_id]
        )

        class StubModel:
            def __init__(self):
                self.step = 0

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, input_ids, past_key_values=None, use_cache=True):
                seq_len = input_ids.shape[1]
                logits = torch.zeros(1, seq_len, 300)
                target = targets[min(self.step, len(targets) - 1)]
                logits[0, -1, target] = 20.0
                self.step += 1
                return SimpleNamespace(
                    logits=logits,
                    past_key_values=(torch.empty(0),),
                )

        generator = TextGenerator(StubModel(), tokenizer, device="cpu")
        result = generator.generate_with_thinking(
            "Soru",
            max_new_tokens=len(targets),
        )
        self.assertEqual(result["thinking"], "ok")
        self.assertEqual(result["response"], "cevap")
        self.assertNotIn("<|im_end|>", result["full"])

    def test_chat_stream_does_not_apply_chat_template_twice(self):
        tokenizer = ModernTokenizer()
        model = ModernLLMForCausalLM(tiny_config()).eval()
        chat = ChatInterface(model, tokenizer)
        captured = {}

        def fake_stream(input_ids, config):
            captured["ids"] = input_ids[0].tolist()
            return iter(())

        chat.generator._stream_tokens = fake_stream
        list(chat.stream_chat("Merhaba"))

        expected_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": chat.system_prompt},
                {"role": "user", "content": "Merhaba"},
            ],
            add_generation_prompt=True,
        )
        expected_ids = tokenizer.encode(expected_text)
        self.assertEqual(captured["ids"], expected_ids)
        self.assertEqual(captured["ids"].count(tokenizer.special_tokens["<|user|>"]), 1)


class ConfigAndValidationTests(unittest.TestCase):
    def test_invalid_head_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            tiny_config(hidden_size=15)

    def test_invalid_training_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(logging_steps=0)

    def test_generation_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            GenerationConfig(temperature=-1.0)
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=0.0)

    def test_gpu_preset_is_not_shared_between_calls(self):
        first, first_train = get_config_for_gpu("T4")
        first.hidden_size = 1
        first_train.batch_size = 1
        second, second_train = get_config_for_gpu("T4")
        self.assertEqual(second.hidden_size, 512)
        self.assertEqual(second_train.batch_size, 4)

    def test_beam_search_is_not_silently_ignored(self):
        generator = TextGenerator(
            ModernLLMForCausalLM(tiny_config()).eval(),
            ModernTokenizer(),
            device="cpu",
        )
        config = GenerationConfig(num_beams=2)
        with self.assertRaises(NotImplementedError):
            generator._generate_tokens(torch.tensor([[14]]), config)


if __name__ == "__main__":
    unittest.main()
