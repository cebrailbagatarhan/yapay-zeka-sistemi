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
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            # Varsayılan olarak byte-level fallback tokenizer
            self._init_byte_fallback()
    
    def _init_byte_fallback(self):
        """Byte-level fallback tokenizer (SentencePiece olmadan çalışır)."""
        # Özel tokenlar
        for token, idx in self.special_tokens.items():
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        
        # ASCII ve genişletilmiş karakter seti
        offset = len(self.special_tokens)
        for i in range(256):
            char = chr(i) if i >= 32 else f"<0x{i:02X}>"
            self._token_to_id[char] = offset + i
            self._id_to_token[offset + i] = char
        
        # Türkçe karakterler (ayrıca ekle)
        turkish_chars = "çÇğĞıİöÖşŞüÜâÂîÎûÛ"
        for i, char in enumerate(turkish_chars):
            idx = offset + 256 + i
            if char not in self._token_to_id:
                self._token_to_id[char] = idx
                self._id_to_token[idx] = char
    
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
            return
        
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        os.makedirs(output_path, exist_ok=True)
        
        # Eğitim verisini dosyaya yaz
        train_file = os.path.join(output_path, "train_data.txt")
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')
        
        # Özel token listesi
        user_defined_symbols = [t for t in self.special_tokens.keys() if t != '\n']
        
        # SentencePiece modeli eğit
        model_prefix = os.path.join(output_path, "tokenizer")
        
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size - len(self.special_tokens),
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
        )
        
        # Modeli yükle
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.load(model_prefix + ".model")
        
        # Token map güncelle
        self._update_token_maps()
        
        # Config kaydet
        config = {
            "vocab_size": vocab_size,
            "model_type": "bpe",
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(output_path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Tokenizer eğitildi: {output_path}")
        print(f"   Vocabulary: {vocab_size}")
        
        # Temizlik
        if os.path.exists(train_file):
            os.remove(train_file)
    
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
        
        # SentencePiece tokenları
        for i in range(self._sp_model.get_piece_size()):
            token = self._sp_model.id_to_piece(i)
            idx = i + len(self.special_tokens)
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
        
        if self._sp_model is not None:
            # SentencePiece ile tokenize et
            sp_ids = self._sp_model.encode(text, out_type=int)
            # Offset ekle (özel tokenlar için)
            ids.extend([i + len(self.special_tokens) for i in sp_ids])
        else:
            # Byte-level fallback
            # Önce özel tokenları kontrol et
            remaining = text
            while remaining:
                found = False
                for token in sorted(self.special_tokens.keys(), key=len, reverse=True):
                    if remaining.startswith(token):
                        ids.append(self.special_tokens[token])
                        remaining = remaining[len(token):]
                        found = True
                        break
                
                if not found:
                    char = remaining[0]
                    if char in self._token_to_id:
                        ids.append(self._token_to_id[char])
                    else:
                        # Byte fallback
                        for byte in char.encode('utf-8'):
                            byte_token = f"<0x{byte:02X}>"
                            if byte_token in self._token_to_id:
                                ids.append(self._token_to_id[byte_token])
                            else:
                                ids.append(self.special_tokens["<unk>"])
                    remaining = remaining[1:]
        
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
        if self._sp_model is not None:
            # Özel tokenları ayır
            special_ids = set(self.special_tokens.values())
            
            if skip_special_tokens:
                sp_ids = [i - len(self.special_tokens) for i in ids if i not in special_ids]
            else:
                parts = []
                sp_buffer = []
                
                for token_id in ids:
                    if token_id in special_ids:
                        if sp_buffer:
                            parts.append(self._sp_model.decode(sp_buffer))
                            sp_buffer = []
                        parts.append(self._id_to_token.get(token_id, ""))
                    else:
                        sp_buffer.append(token_id - len(self.special_tokens))
                
                if sp_buffer:
                    parts.append(self._sp_model.decode(sp_buffer))
                
                return "".join(parts)
            
            return self._sp_model.decode(sp_ids)
        else:
            # Byte-level fallback decode
            tokens = []
            for token_id in ids:
                if skip_special_tokens and token_id in self.special_tokens.values():
                    continue
                token = self._id_to_token.get(token_id, "")
                tokens.append(token)
            return "".join(tokens)
    
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
        }
        
        with open(os.path.join(path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        if self._sp_model is not None:
            # SentencePiece model dosyasını kopyala
            import shutil
            sp_model_path = os.path.join(path, "tokenizer.model")
            self._sp_model.save_model(path)
    
    def load(self, path: str):
        """Tokenizer'ı yükle."""
        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.vocab_size = config.get("vocab_size", self.vocab_size)
            self.special_tokens = config.get("special_tokens", self.special_tokens)
        
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
