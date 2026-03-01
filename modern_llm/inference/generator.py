"""
Text Generator ve Chat Interface
=================================

Modern LLM için inference modülü.

Özellikler:
- Temperature sampling
- Top-k ve Top-p (nucleus) sampling
- Repetition penalty
- Beam search (opsiyonel)
- KV-Cache ile hızlı üretim
- CoT (Chain-of-Thought) entegrasyonu
- Streaming text generation
- İnteraktif sohbet arayüzü
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
import time
import sys


@dataclass
class GenerationConfig:
    """Text generation ayarları."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1  # 1 = greedy/sampling, >1 = beam search
    
    # CoT
    enable_thinking: bool = True
    max_thinking_tokens: int = 1024
    
    # Durma koşulları
    stop_tokens: List[int] = field(default_factory=list)
    
    # Streaming
    stream: bool = False


class TextGenerator:
    """
    Modern LLM Text Generator.
    
    KV-Cache kullanarak verimli autoregressive üretim.
    Temperature, top-k, top-p, repetition penalty destekli.
    
    Kullanım:
        generator = TextGenerator(model, tokenizer)
        text = generator.generate("Merhaba, nasılsın?")
    """
    
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Metin üret.
        
        Args:
            prompt: Kullanıcı girdisi
            config: Üretim ayarları
            system_prompt: Sistem mesajı
            max_new_tokens: Üretilecek max token (kısayol)
            temperature: Sampling sıcaklığı (kısayol)
            top_k: Top-k filtre (kısayol)
            top_p: Top-p nucleus filtre (kısayol)
            repetition_penalty: Tekrar cezası (kısayol)
            do_sample: Sampling aç/kapa (kısayol)
            
        Returns:
            Üretilen metin
        """
        config = config or GenerationConfig()
        
        # Kısayol parametreleri config'e uygula
        if max_new_tokens is not None: config.max_new_tokens = max_new_tokens
        if temperature is not None: config.temperature = temperature
        if top_k is not None: config.top_k = top_k
        if top_p is not None: config.top_p = top_p
        if repetition_penalty is not None: config.repetition_penalty = repetition_penalty
        if do_sample is not None: config.do_sample = do_sample
        
        # Chat template uygula
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Tokenize
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        else:
            input_ids = self.tokenizer.encode(prompt)
        
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Üretim
        output_ids = self._generate_tokens(input_ids, config)
        
        # Decode
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens.tolist())
        
        return response
    
    @torch.no_grad()
    def generate_with_thinking(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        CoT destekli metin üret.
        
        Args:
            prompt: Kullanıcı girdisi
            max_new_tokens: Üretilecek max token (kısayol)
            temperature: Sampling sıcaklığı (kısayol)
        
        Returns:
            {
                "thinking": "...",  # İç düşünce süreci
                "response": "...",  # Nihai yanıt
                "full": "...",      # Komple çıktı
            }
        """
        config = config or GenerationConfig(enable_thinking=True)
        if max_new_tokens is not None: config.max_new_tokens = max_new_tokens
        if temperature is not None: config.temperature = temperature
        
        # Thinking prompt ekle
        think_system = system_prompt or (
            "Sen düşünen bir yapay zeka asistanısın. "
            "Her soruya önce <think> etiketleri arasında düşün, "
            "sonra yanıtını ver."
        )
        
        full_response = self.generate(prompt, config, think_system)
        
        # Thinking ve response ayır
        thinking = ""
        response = full_response
        
        if "<think>" in full_response and "</think>" in full_response:
            think_start = full_response.index("<think>") + len("<think>")
            think_end = full_response.index("</think>")
            thinking = full_response[think_start:think_end].strip()
            response = full_response[think_end + len("</think>"):].strip()
        
        return {
            "thinking": thinking,
            "response": response,
            "full": full_response,
        }
    
    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming text generation.
        
        Token token üretir (generator/yield).
        
        Kullanım:
            for token in generator.stream_generate("Merhaba"):
                print(token, end="", flush=True)
        """
        config = config or GenerationConfig(stream=True)
        
        # Chat template uygula
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Tokenize
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        else:
            input_ids = self.tokenizer.encode(prompt)
        
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Token token üret
        past_key_values = None
        generated = input_ids
        
        for _ in range(config.max_new_tokens):
            # Forward
            if past_key_values is not None:
                # Sadece son token
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            
            # Sampling
            next_token = self._sample_token(logits, generated, config)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            
            # EOS kontrolü
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Stop tokens kontrolü
            if next_token.item() in config.stop_tokens:
                break
            
            # Decode ve yield
            token_text = self.tokenizer.decode([next_token.item()])
            yield token_text
    
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Token üretim döngüsü.
        
        KV-Cache kullanarak verimli autoregressive üretim.
        """
        generated = input_ids
        past_key_values = None
        
        for step in range(config.max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # KV-cache varsa sadece son token gönder
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]  # Son token logits
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            
            # Token sampling
            next_token = self._sample_token(logits, generated, config)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            
            # EOS kontrolü
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Stop tokens
            if next_token.item() in config.stop_tokens:
                break
        
        return generated
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Token sampling stratejileri.
        
        1. Repetition penalty uygula
        2. Temperature uygula
        3. Top-k filtrele
        4. Top-p (nucleus) filtrele
        5. Sample veya argmax
        """
        # (1) Repetition penalty
        if config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated, config.repetition_penalty)
        
        # Greedy decoding
        if not config.do_sample:
            return logits.argmax(dim=-1)
        
        # (2) Temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # (3) Top-k filtering
        if config.top_k > 0:
            logits = self._top_k_filter(logits, config.top_k)
        
        # (4) Top-p (nucleus) filtering
        if config.top_p < 1.0:
            logits = self._top_p_filter(logits, config.top_p)
        
        # (5) Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_token
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Tekrar cezası uygula."""
        # Daha önce üretilen token'lara ceza
        for token_id in set(generated[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k filtering: En yüksek k token dışındakileri -inf yap."""
        top_k = min(k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_value, torch.tensor(float('-inf'), device=logits.device), logits)
        return logits
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Top-p (nucleus) filtering: Kümülatif olasılık p'yi geçene kadar token al."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # p'yi geçen tokenleri filtrele
        sorted_indices_to_remove = cumulative_probs > p
        # İlk token her zaman kalsın
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        
        return logits


class ChatInterface:
    """
    İnteraktif sohbet arayüzü.
    
    Konuşma geçmişi tutarak, çok turlu sohbet yapar.
    CoT (Chain-of-Thought) destekli.
    
    Kullanım:
        chat = ChatInterface(model, tokenizer)
        chat.start()
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.generator = TextGenerator(model, tokenizer, device)
        self.tokenizer = tokenizer
        
        self.system_prompt = system_prompt or (
            "Sen yardımcı bir yapay zeka asistanısın. "
            "Sorulara net, doğru ve faydalı yanıtlar verirsin. "
            "Karmaşık sorularda adım adım düşünürsün."
        )
        
        self.generation_config = generation_config or GenerationConfig()
        
        # Konuşma geçmişi
        self.history: List[Dict[str, str]] = []
        self.max_history_turns = 20
    
    def chat(self, user_message: str, show_thinking: bool = False) -> str:
        """
        Tek tur sohbet.
        
        Args:
            user_message: Kullanıcı mesajı
            show_thinking: Düşünce sürecini göster
            
        Returns:
            Asistan yanıtı
        """
        # Geçmişe ekle
        self.history.append({"role": "user", "content": user_message})
        
        # Geçmiş çok uzunsa kes
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]
        
        # Mesajları oluştur
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        
        # Chat template uygula
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_text = self._format_messages(messages)
            input_ids = self.tokenizer.encode(prompt_text)
        
        input_ids = torch.tensor(
            [input_ids], dtype=torch.long, device=self.generator.device
        )
        
        # Üretim
        output_ids = self.generator._generate_tokens(input_ids, self.generation_config)
        
        # Decode
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens.tolist())
        
        # Thinking extract
        thinking = ""
        clean_response = response
        
        if "<think>" in response and "</think>" in response:
            think_start = response.index("<think>") + len("<think>")
            think_end = response.index("</think>")
            thinking = response[think_start:think_end].strip()
            clean_response = response[think_end + len("</think>"):].strip()
        
        # Geçmişe ekle
        self.history.append({"role": "assistant", "content": clean_response})
        
        if show_thinking and thinking:
            return f"💭 Düşünce:\n{thinking}\n\n📝 Yanıt:\n{clean_response}"
        
        return clean_response
    
    def stream_chat(
        self,
        user_message: str,
        show_thinking: bool = True,
    ) -> Generator[str, None, None]:
        """
        Streaming sohbet.
        
        Token token yanıt üretir.
        """
        self.history.append({"role": "user", "content": user_message})
        
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            prompt_text = self._format_messages(messages)
        
        full_response = ""
        in_thinking = False
        
        for token in self.generator.stream_generate(
            prompt_text,
            self.generation_config,
        ):
            full_response += token
            
            # Thinking durumunu takip et
            if "<think>" in full_response and not in_thinking:
                in_thinking = True
                if show_thinking:
                    yield "\n💭 Düşünüyor...\n"
            
            if "</think>" in full_response and in_thinking:
                in_thinking = False
                if show_thinking:
                    yield "\n📝 Yanıt:\n"
                continue
            
            if not in_thinking or show_thinking:
                yield token
        
        self.history.append({"role": "assistant", "content": full_response})
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Basit mesaj formatlama (chat template yoksa)."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    
    def reset(self):
        """Konuşma geçmişini sıfırla."""
        self.history = []
        print("🔄 Sohbet geçmişi sıfırlandı.")
    
    def start(self):
        """
        İnteraktif sohbet başlat (terminal).
        
        Komutlar:
            /quit - Çıkış
            /reset - Geçmişi sıfırla
            /think - Düşünme modunu aç/kapa
            /config - Mevcut ayarları göster
            /temp <float> - Temperature değiştir
        """
        print("=" * 60)
        print("🤖 Modern LLM - İnteraktif Sohbet")
        print("=" * 60)
        print(f"Sistem: {self.system_prompt[:80]}...")
        print("Komutlar: /quit, /reset, /think, /config, /temp <değer>")
        print("=" * 60)
        
        show_thinking = True
        
        while True:
            try:
                user_input = input("\n👤 Sen: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Güle güle!")
                break
            
            if not user_input:
                continue
            
            # Komutları işle
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                
                if cmd[0] == "/quit":
                    print("👋 Güle güle!")
                    break
                elif cmd[0] == "/reset":
                    self.reset()
                    continue
                elif cmd[0] == "/think":
                    show_thinking = not show_thinking
                    status = "açık" if show_thinking else "kapalı"
                    print(f"💭 Düşünme modu: {status}")
                    continue
                elif cmd[0] == "/config":
                    self._show_config()
                    continue
                elif cmd[0] == "/temp" and len(cmd) > 1:
                    try:
                        self.generation_config.temperature = float(cmd[1])
                        print(f"🌡️ Temperature: {self.generation_config.temperature}")
                    except ValueError:
                        print("❌ Geçersiz değer")
                    continue
                elif cmd[0] == "/stream":
                    # Streaming mode
                    print("\n🤖 Asistan: ", end="", flush=True)
                    for token in self.stream_chat(user_input[8:], show_thinking):
                        print(token, end="", flush=True)
                    print()
                    continue
                else:
                    print(f"❌ Bilinmeyen komut: {cmd[0]}")
                    continue
            
            # Normal sohbet
            print("\n🤖 Asistan: ", end="", flush=True)
            
            if self.generation_config.stream:
                for token in self.stream_chat(user_input, show_thinking):
                    print(token, end="", flush=True)
                print()
            else:
                response = self.chat(user_input, show_thinking)
                print(response)
    
    def _show_config(self):
        """Mevcut ayarları göster."""
        print("\n⚙️ Mevcut Ayarlar:")
        print(f"  Temperature: {self.generation_config.temperature}")
        print(f"  Top-k: {self.generation_config.top_k}")
        print(f"  Top-p: {self.generation_config.top_p}")
        print(f"  Max tokens: {self.generation_config.max_new_tokens}")
        print(f"  Repetition penalty: {self.generation_config.repetition_penalty}")
        print(f"  Sampling: {self.generation_config.do_sample}")
        print(f"  Thinking: {self.generation_config.enable_thinking}")
        print(f"  Geçmiş tur: {len(self.history) // 2}")


def quick_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "auto",
) -> str:
    """
    Hızlı text generation (tek fonksiyon).
    
    Kullanım:
        text = quick_generate(model, tokenizer, "Merhaba!")
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generator = TextGenerator(model, tokenizer, torch.device(device))
    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    
    return generator.generate(prompt, config)
