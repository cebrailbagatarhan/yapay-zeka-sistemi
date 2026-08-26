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
        self.requested_vocab_size = vocab_size
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS.copy()
        self._sp_model = None
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            # Varsayılan olarak byte-level fallback tokenizer
            self._init_byte_fallback()
    
    def _init_byte_fallback(self):
        """Tüm Unicode metinleri UTF-8 byte'ları üzerinden kayıpsız kodla."""
        self._sp_model = None
        self._token_to_id = {}
        self._id_to_token = {}
        
        for token, idx in self.special_tokens.items():
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        
        byte_offset = len(self.special_tokens)
        for byte in range(256):
            token = f"<0x{byte:02X}>"
            token_id = byte_offset + byte
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        
        # Fallback model yalnızca gerçekten kodlayabildiği vocabulary'yi raporlar.
        self.vocab_size = len(self.special_tokens) + 256
    
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
        except ImportError as exc:
            raise RuntimeError(
                "Tokenizer eğitimi için sentencepiece gereklidir: "
                "pip install sentencepiece"
            ) from exc
        
        if vocab_size is None:
            vocab_size = self.requested_vocab_size
        
        minimum_vocab_size = len(self.special_tokens) + 256
        if vocab_size < minimum_vocab_size:
            raise ValueError(
                f"byte_fallback için vocab_size en az {minimum_vocab_size} olmalıdır"
            )
        
        os.makedirs(output_path, exist_ok=True)
        
        # Eğitim verisini dosyaya yaz
        train_file = os.path.join(output_path, "train_data.txt")
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')
        
        # İlk dört token SentencePiece meta tokenlarıdır. Kalan tokenlar
        # sözlükteki sırayla 4..N ID'lerini alır.
        user_defined_symbols = [
            token
            for token, token_id in sorted(
                self.special_tokens.items(),
                key=lambda item: item[1],
            )
            if token_id >= 4
        ]
        
        # SentencePiece modeli eğit
        model_prefix = os.path.join(output_path, "tokenizer")
        
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9999,  # Türkçe için yüksek coverage
            num_threads=os.cpu_count() or 4,
            split_digits=True,
            byte_fallback=True,
            max_sentence_length=8192,
            user_defined_symbols=user_defined_symbols,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            normalization_rule_name="identity",  # NFD/NFKC yapma
            hard_vocab_limit=False,
        )
        
        # Modeli yükle
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.load(model_prefix + ".model")
        
        # Token map güncelle
        self._update_token_maps()
        
        # Config kaydet
        config = {
            "vocab_size": self.vocab_size,
            "requested_vocab_size": vocab_size,
            "model_type": "bpe",
            "special_tokens": self.special_tokens,
            "has_sp_model": True,
        }
        with open(os.path.join(output_path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Tokenizer eğitildi: {output_path}")
        print(f"   Vocabulary: {self.vocab_size}")
        
        # Temizlik
        if os.path.exists(train_file):
            os.remove(train_file)
    
    def _update_token_maps(self):
        """SentencePiece ID'lerini offset uygulamadan doğrudan kullan."""
        if self._sp_model is None:
            return
        
        self._token_to_id = {}
        self._id_to_token = {}
        
        for token, expected_id in self.special_tokens.items():
            actual_id = self._sp_model.piece_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Özel token ID uyumsuzluğu: {token} "
                    f"beklenen={expected_id}, alınan={actual_id}"
                )
        
        for token_id in range(self._sp_model.get_piece_size()):
            token = self._sp_model.id_to_piece(token_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        
        self.vocab_size = self._sp_model.get_piece_size()
    
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
        
        if self._sp_model is not None:
            ids.extend(self._sp_model.encode(text, out_type=int))
        else:
            # Özel tokenları koru; diğer tüm karakterleri UTF-8 byte'larına çevir.
            remaining = text
            byte_offset = len(self.special_tokens)
            sorted_special_tokens = sorted(
                self.special_tokens.keys(),
                key=len,
                reverse=True,
            )
            
            while remaining:
                matched_token = next(
                    (
                        token
                        for token in sorted_special_tokens
                        if remaining.startswith(token)
                    ),
                    None,
                )
                if matched_token is not None:
                    ids.append(self.special_tokens[matched_token])
                    remaining = remaining[len(matched_token):]
                    continue
                
                char = remaining[0]
                ids.extend(byte_offset + byte for byte in char.encode("utf-8"))
                remaining = remaining[1:]
        
        if add_eos:
            ids.append(self.special_tokens["<eos>"])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Token ID'lerini metne dönüştür."""
        special_ids = set(self.special_tokens.values())
        
        if self._sp_model is not None:
            if skip_special_tokens:
                clean_ids = [token_id for token_id in ids if token_id not in special_ids]
                return self._sp_model.decode(clean_ids)
            
            parts = []
            sp_buffer = []
            for token_id in ids:
                if token_id in special_ids:
                    if sp_buffer:
                        parts.append(self._sp_model.decode(sp_buffer))
                        sp_buffer = []
                    parts.append(self._id_to_token.get(token_id, "<unk>"))
                else:
                    sp_buffer.append(token_id)
            if sp_buffer:
                parts.append(self._sp_model.decode(sp_buffer))
            return "".join(parts)
        
        byte_offset = len(self.special_tokens)
        parts = []
        byte_buffer = bytearray()
        
        def flush_bytes():
            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()
        
        for token_id in ids:
            if token_id in special_ids:
                flush_bytes()
                if not skip_special_tokens:
                    parts.append(self._id_to_token.get(token_id, "<unk>"))
            elif byte_offset <= token_id < byte_offset + 256:
                byte_buffer.append(token_id - byte_offset)
            else:
                flush_bytes()
                parts.append("\uFFFD")
        
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
            "requested_vocab_size": self.requested_vocab_size,
            "special_tokens": self.special_tokens,
            "has_sp_model": self._sp_model is not None,
        }
        
        with open(os.path.join(path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        if self._sp_model is not None:
            sp_model_path = os.path.join(path, "tokenizer.model")
            with open(sp_model_path, "wb") as model_file:
                model_file.write(self._sp_model.serialized_model_proto())
    
    def load(self, path: str):
        """Tokenizer'ı yükle."""
        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.requested_vocab_size = config.get(
                "requested_vocab_size",
                config.get("vocab_size", self.requested_vocab_size),
            )
            loaded_special_tokens = config.get("special_tokens", self.special_tokens)
            # Eski config'lerde newline ayrı special token olarak tutuluyordu.
            loaded_special_tokens.pop("\n", None)
            self.special_tokens = loaded_special_tokens
        
        # SentencePiece model yükle
        sp_model_path = os.path.join(path, "tokenizer.model")
        if os.path.exists(sp_model_path):
            try:
                import sentencepiece as spm
                self._sp_model = spm.SentencePieceProcessor()
                self._sp_model.load(sp_model_path)
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
