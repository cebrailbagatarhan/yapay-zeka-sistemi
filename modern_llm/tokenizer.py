"""
Tokenizer Modülü
================

Modern LLM için BPE (Byte-Pair Encoding) tokenizer.

İki mod:
1. Sıfırdan eğitim: SentencePiece ile BPE tokenizer eğit
2. Yükleme: Mevcut tokenizer'ı yükle

Özel tokenlar:
- <pad>, <bos>, <eos>, <unk>
- <|im_start|>, <|im_end|> (chat format)
- <think>, </think> (CoT reasoning)
- <|system|>, <|user|>, <|assistant|>
"""

import os
import json
import re
from typing import List, Optional, Dict, Any


# Özel token tanımları
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<|im_start|>": 4,
    "<|im_end|>": 5,
    "<think>": 6,
    "</think>": 7,
    "<|system|>": 8,
    "<|user|>": 9,
    "<|assistant|>": 10,
    "<|tool|>": 11,
    "<|tool_result|>": 12,
    "\n": 13,
}

# Chat format şablonları
CHAT_TEMPLATE = {
    "system": "<|im_start|><|system|>\n{content}<|im_end|>\n",
    "user": "<|im_start|><|user|>\n{content}<|im_end|>\n",
    "assistant": "<|im_start|><|assistant|>\n{content}<|im_end|>\n",
    "assistant_with_thinking": "<|im_start|><|assistant|>\n<think>\n{thinking}\n</think>\n{content}<|im_end|>\n",
    "generation_prefix": "<|im_start|><|assistant|>\n",
}


class ModernTokenizer:
    """
    Modern LLM Tokenizer.
    
    SentencePiece BPE tabanlı tokenizer ile özel token desteği.
    Tokenizer eğitimi ve yükleme fonksiyonları içerir.
    """
    
    def __init__(
        self,
        vocab_size: int = 32256,
        model_path: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS.copy()
        self._sp_model = None
        self._sp_model_path: Optional[str] = None
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._build_special_pattern()

        if model_path:
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Tokenizer dizini bulunamadı: {model_path}")
            self.load(model_path)
        else:
            # Varsayılan olarak byte-level fallback tokenizer
            self._init_byte_fallback()

    def _build_special_pattern(self):
        """Özel tokenları en uzun eşleşme önceliğiyle bulan regex."""
        alternatives = sorted(self.special_tokens, key=len, reverse=True)
        self._special_pattern = re.compile(
            "|".join(re.escape(token) for token in alternatives)
        )
        self._special_offset = len(self.special_tokens)

    def _init_byte_fallback(self):
        """Tüm Unicode metinleri kayıpsız taşıyan byte-level tokenizer."""
        self._sp_model = None
        self._sp_model_path = None
        self._token_to_id = {}
        self._id_to_token = {}

        # Özel tokenlar 0..N-1 aralığını kullanır.
        for token, idx in self.special_tokens.items():
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token

        # UTF-8 byte'ları N..N+255 aralığına eşlenir. Böylece emoji, CJK,
        # kontrol karakterleri ve tüm Türkçe karakterler round-trip edilir.
        offset = self._special_offset
        for byte_value in range(256):
            token = f"<0x{byte_value:02X}>"
            self._token_to_id[token] = offset + byte_value
            self._id_to_token[offset + byte_value] = token

    def _encode_plain_text(self, text: str) -> List[int]:
        """Özel olmayan metin parçasını token ID'lerine çevir."""
        if not text:
            return []

        if self._sp_model is not None:
            ids = []
            unk_id = self._sp_model.unk_id()
            for sp_id in self._sp_model.encode(text, out_type=int):
                if sp_id == unk_id:
                    ids.append(self.special_tokens["<unk>"])
                else:
                    ids.append(sp_id + self._special_offset)
            return ids

        # Byte-level fallback: her Unicode karakteri UTF-8 byte'larına ayır.
        return [self._special_offset + byte for byte in text.encode("utf-8")]
    
    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        output_path: str = "tokenizer",
    ):
        """
        SentencePiece ile BPE tokenizer eğit.
        
        Args:
            texts: Eğitim metinleri
            vocab_size: Vocabulary boyutu
            output_path: Kayıt yolu
        """
        try:
            import sentencepiece as spm
        except ImportError:
            print("⚠️ sentencepiece yüklü değil. pip install sentencepiece")
            print("   Byte-level fallback tokenizer kullanılıyor.")
            return False

        if not texts:
            raise ValueError("Tokenizer eğitimi için en az bir metin gerekli.")

        if vocab_size is None:
            vocab_size = self.vocab_size

        os.makedirs(output_path, exist_ok=True)

        # Eğitim verisini dosyaya yaz
        train_file = os.path.join(output_path, "train_data.txt")
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')

        # Özel tokenlar SentencePiece'in dışında yönetilir. Böylece sabit
        # 0..13 ID'leri korunur ve SP vocabulary'si ile çakışmaz.
        model_prefix = os.path.join(output_path, "tokenizer")
        sp_vocab_size = vocab_size - len(self.special_tokens)
        if sp_vocab_size < 256:
            raise ValueError(
                "SentencePiece + byte fallback için vocab_size en az "
                f"{len(self.special_tokens) + 256} olmalı."
            )

        try:
            spm.SentencePieceTrainer.train(
                input=train_file,
                model_prefix=model_prefix,
                vocab_size=sp_vocab_size,
                model_type="bpe",
                character_coverage=0.9999,
                num_threads=os.cpu_count() or 4,
                split_digits=True,
                byte_fallback=True,
                max_sentence_length=8192,
                pad_id=-1,
                bos_id=-1,
                eos_id=-1,
                unk_id=0,
                hard_vocab_limit=False,
                normalization_rule_name="identity",
                minloglevel=2,
            )

            self._sp_model = spm.SentencePieceProcessor()
            self._sp_model.load(model_prefix + ".model")
            self._sp_model_path = model_prefix + ".model"
        finally:
            # Başarısız eğitimden kalan geçici dosyayı da temizle.
            if os.path.exists(train_file):
                os.remove(train_file)

        # Token map güncelle
        self._update_token_maps()

        # Config kaydet
        config = {
            "vocab_size": vocab_size,
            "model_type": "bpe",
            "special_tokens": self.special_tokens,
            "sentencepiece_special_tokens_external": True,
        }
        with open(os.path.join(output_path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        actual_vocab = len(self._id_to_token)
        print(f"✅ Tokenizer eğitildi: {output_path}")
        print(f"   Vocabulary: {actual_vocab} (hedef: {vocab_size})")
        return True
    
    def _update_token_maps(self):
        """SentencePiece modelinden token map güncelle."""
        if self._sp_model is None:
            return
        
        self._token_to_id = {}
        self._id_to_token = {}
        
        # Özel tokenlar
        for token, idx in self.special_tokens.items():
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        
        # SentencePiece tokenları (özel token adlarıyla çakışanları atla)
        for i in range(self._sp_model.get_piece_size()):
            token = self._sp_model.id_to_piece(i)
            idx = i + self._special_offset
            if token in self.special_tokens:
                continue
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Metni token ID'lerine dönüştür.
        
        Args:
            text: Girdi metni
            add_bos: Başlangıca BOS token ekle
            add_eos: Sona EOS token ekle
        
        Returns:
            Token ID listesi
        """
        ids = []

        if add_bos:
            ids.append(self.special_tokens["<bos>"])

        # Özel tokenları normal metinden ayır; böylece <think>, <|im_end|>
        # gibi işaretler alt kelime parçalarına bölünmez.
        cursor = 0
        for match in self._special_pattern.finditer(text):
            ids.extend(self._encode_plain_text(text[cursor:match.start()]))
            ids.append(self.special_tokens[match.group(0)])
            cursor = match.end()
        ids.extend(self._encode_plain_text(text[cursor:]))

        if add_eos:
            ids.append(self.special_tokens["<eos>"])

        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Token ID'lerini metne dönüştür.
        
        Args:
            ids: Token ID listesi
            skip_special_tokens: Özel tokenları atla
        
        Returns:
            Decoded metin
        """
        special_ids = set(self.special_tokens.values())
        newline_id = self.special_tokens.get("\n")
        parts: List[str] = []

        if self._sp_model is not None:
            sp_buffer: List[int] = []

            def flush_sp():
                if sp_buffer:
                    parts.append(self._sp_model.decode(sp_buffer))
                    sp_buffer.clear()

            for token_id in ids:
                if token_id in special_ids:
                    flush_sp()
                    if token_id == newline_id:
                        parts.append("\n")
                    elif not skip_special_tokens:
                        parts.append(self._id_to_token.get(token_id, ""))
                elif token_id >= self._special_offset:
                    sp_id = token_id - self._special_offset
                    if sp_id < self._sp_model.get_piece_size():
                        sp_buffer.append(sp_id)
                    else:
                        flush_sp()
                elif not skip_special_tokens:
                    parts.append(self._id_to_token.get(token_id, ""))

            flush_sp()
            return "".join(parts)

        # Byte-level fallback decode: ardışık byte tokenlarını birlikte
        # çözerek çok baytlı UTF-8 karakterleri geri kur.
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id in special_ids:
                flush_bytes()
                if token_id == newline_id:
                    parts.append("\n")
                elif not skip_special_tokens:
                    parts.append(self._id_to_token.get(token_id, ""))
            elif self._special_offset <= token_id < self._special_offset + 256:
                byte_buffer.append(token_id - self._special_offset)
            elif not skip_special_tokens:
                parts.append(self._id_to_token.get(token_id, ""))

        flush_bytes()
        return "".join(parts)
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        thinking: Optional[str] = None,
    ) -> str:
        """
        Chat mesajlarını model formatına dönüştür.
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            add_generation_prompt: Assistant prefix ekle
            thinking: CoT düşünme metni
        
        Returns:
            Formatlanmış chat string
        """
        result = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                result += CHAT_TEMPLATE["system"].format(content=content)
            elif role == "user":
                result += CHAT_TEMPLATE["user"].format(content=content)
            elif role == "assistant":
                if thinking and msg == messages[-1]:
                    result += CHAT_TEMPLATE["assistant_with_thinking"].format(
                        thinking=thinking,
                        content=content,
                    )
                else:
                    result += CHAT_TEMPLATE["assistant"].format(content=content)
        
        if add_generation_prompt:
            result += CHAT_TEMPLATE["generation_prefix"]
        
        return result
    
    def save(self, path: str):
        """Tokenizer'ı kaydet."""
        os.makedirs(path, exist_ok=True)
        
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "has_sp_model": self._sp_model is not None,
            "sentencepiece_special_tokens_external": True,
        }

        with open(os.path.join(path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        if self._sp_model is not None:
            # load/save aynı modeli serileştirsin; dizin API'sine bağımlı olma.
            sp_model_path = os.path.join(path, "tokenizer.model")
            with open(sp_model_path, "wb") as f:
                f.write(self._sp_model.serialized_model_proto())
            self._sp_model_path = sp_model_path

    def load(self, path: str):
        """Tokenizer'ı yükle."""
        self._sp_model = None
        self._sp_model_path = None
        self._token_to_id = {}
        self._id_to_token = {}

        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.vocab_size = config.get("vocab_size", self.vocab_size)
            self.special_tokens = config.get("special_tokens", self.special_tokens)
        self._build_special_pattern()

        # SentencePiece model yükle
        sp_model_path = os.path.join(path, "tokenizer.model")
        if os.path.exists(sp_model_path):
            try:
                import sentencepiece as spm
                self._sp_model = spm.SentencePieceProcessor()
                self._sp_model.load(sp_model_path)
                self._sp_model_path = sp_model_path
                self._update_token_maps()
            except ImportError:
                print("⚠️ sentencepiece yüklü değil, byte-level fallback kullanılıyor")
                self._init_byte_fallback()
        else:
            self._init_byte_fallback()
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<bos>"]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]
    
    @property
    def think_start_id(self) -> int:
        return self.special_tokens["<think>"]
    
    @property
    def think_end_id(self) -> int:
        return self.special_tokens["</think>"]
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"ModernTokenizer(vocab_size={self.vocab_size}, special_tokens={len(self.special_tokens)})"
