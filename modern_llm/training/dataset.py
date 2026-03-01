"""
Dataset Modülü
==============

Modern LLM eğitimi için veri pipeline'ları.

Dataset Türleri:
1. TextDataset: Ham metin ön-eğitim (causal LM)
2. ChatDataset: Chat format SFT (instruction-following)
3. CoTDataset: Chain-of-Thought reasoning eğitimi

Veri Formatları:
- HuggingFace datasets entegrasyonu
- JSON/JSONL dosya desteği
- Streaming desteği (büyük datasetler için)
"""

import os
import json
import random
from typing import List, Dict, Optional, Any, Union
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Ham metin dataset'i (pre-training için).
    
    Metinleri sabit uzunlukta bloklara böler ve
    next-token prediction için hazırlar.
    
    Kullanım:
        dataset = TextDataset(texts, tokenizer, max_length=2048)
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
        stride: int = None,
    ):
        """
        Args:
            texts: Ham metin listesi
            tokenizer: ModernTokenizer instance
            max_length: Max sequence uzunluğu
            stride: Sliding window stride (None = max_length)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        self.examples = []
        
        # Metinleri tokenize et ve bloklara böl
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            all_ids.extend(ids)
        
        # Sabit uzunlukta bloklar oluştur
        for i in range(0, len(all_ids) - max_length, self.stride):
            block = all_ids[i:i + max_length + 1]  # +1 for labels shift
            if len(block) == max_length + 1:
                self.examples.append(block)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        block = self.examples[idx]
        input_ids = torch.tensor(block[:-1], dtype=torch.long)
        labels = torch.tensor(block[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


class ChatDataset(Dataset):
    """
    Chat format dataset (SFT için).
    
    Messages formatında veriyi alır ve model chat template'ine
    dönüştürerek tokenize eder.
    
    Format:
    [
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048,
        mask_user_tokens: bool = True,
    ):
        """
        Args:
            data: Chat mesaj listesi
            tokenizer: ModernTokenizer
            max_length: Max sequence uzunluğu
            mask_user_tokens: User/system tokenlarını loss'tan hariç tut
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_user_tokens = mask_user_tokens
        self.examples = []
        
        for item in data:
            example = self._process_item(item)
            if example is not None:
                self.examples.append(example)
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Tek bir chat örneğini işle."""
        messages = item.get("messages", [])
        if not messages:
            return None
        
        # Chat template uygula
        formatted = self.tokenizer.apply_chat_template(messages)
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted, add_bos=True, add_eos=True)
        
        # Max length'e kırp
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Labels oluştur
        labels = input_ids.copy()
        
        # User/system tokenlarını maskele (sadece assistant'ın cevabını eğit)
        if self.mask_user_tokens:
            labels = self._mask_non_assistant_tokens(messages, labels)
        
        # Padding
        pad_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
        labels = labels + [-100] * pad_length  # -100 = ignore in loss
        attention_mask = [1] * (self.max_length - pad_length) + [0] * pad_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def _mask_non_assistant_tokens(
        self,
        messages: List[Dict[str, str]],
        labels: List[int],
    ) -> List[int]:
        """Assistant olmayan tokenları -100 ile maskele."""
        # Basit yaklaşım: <|assistant|> tokenından sonraki kısımları eğit
        # Daha sofistike yaklaşım her mesajın sınırlarını bulur
        
        # Her mesajı ayrı tokenize edip sınırları bul
        current_pos = 0
        masked_labels = [-100] * len(labels)
        
        is_assistant = False
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "assistant":
                # Assistant mesajının tam formatını tokenize et
                from modern_llm.tokenizer import CHAT_TEMPLATE
                formatted = CHAT_TEMPLATE["assistant"].format(content=content)
                msg_ids = self.tokenizer.encode(formatted)
                
                # Bu mesajın başlangıç pozisyonunu bul
                # (basitleştirilmiş - gerçek implementasyonda daha robust olmalı)
                for i in range(current_pos, len(labels)):
                    if i < len(labels):
                        masked_labels[i] = labels[i]
                
                current_pos += len(msg_ids)
            else:
                # System/user mesajlarını tokenize et
                if role == "system":
                    from modern_llm.tokenizer import CHAT_TEMPLATE
                    formatted = CHAT_TEMPLATE["system"].format(content=content)
                else:
                    from modern_llm.tokenizer import CHAT_TEMPLATE
                    formatted = CHAT_TEMPLATE["user"].format(content=content)
                
                msg_ids = self.tokenizer.encode(formatted)
                current_pos += len(msg_ids)
        
        return masked_labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class CoTDataset(Dataset):
    """
    Chain-of-Thought dataset.
    
    CoT format:
    [
        {
            "instruction": "15 * 23 kaçtır?",
            "thinking": "15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345",
            "response": "15 * 23 = 345"
        },
        ...
    ]
    
    Model formatı:
    <|im_start|><|system|>
    ...
    <|im_end|>
    <|im_start|><|user|>
    15 * 23 kaçtır?
    <|im_end|>
    <|im_start|><|assistant|>
    <think>
    15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345
    </think>
    15 * 23 = 345
    <|im_end|>
    """
    
    SYSTEM_PROMPT = """Sen gelişmiş bir yapay zeka asistanısın. Sorulara cevap vermeden önce <think> etiketleri içinde adım adım düşünürsün. Düşünme sürecin mantıklı, detaylı ve doğrulanabilir olmalıdır."""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            data: CoT örnekleri listesi
            tokenizer: ModernTokenizer
            max_length: Max sequence uzunluğu
            system_prompt: Sistem promptu
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.examples = []
        
        for item in data:
            example = self._process_item(item)
            if example is not None:
                self.examples.append(example)
    
    def _process_item(self, item: Dict[str, str]) -> Optional[Dict[str, torch.Tensor]]:
        """CoT örneğini işle."""
        instruction = item.get("instruction", item.get("question", ""))
        thinking = item.get("thinking", item.get("reasoning", ""))
        response = item.get("response", item.get("answer", ""))
        
        if not instruction or not response:
            return None
        
        # Chat mesajları oluştur
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]
        
        # Assistant cevabı (thinking + response)
        if thinking:
            assistant_content = f"<think>\n{thinking}\n</think>\n{response}"
        else:
            assistant_content = response
        
        messages.append({"role": "assistant", "content": assistant_content})
        
        # Chat template uygula
        formatted = self.tokenizer.apply_chat_template(messages)
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted, add_bos=True, add_eos=True)
        
        # Max length kontrol
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Labels = input_ids (tüm sequecne'ı eğit, thinking dahil)
        labels = input_ids.copy()
        
        # Padding
        pad_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
        labels = labels + [-100] * pad_length
        attention_mask = [1] * (self.max_length - pad_length) + [0] * pad_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_dataset_from_json(path: str) -> List[Dict]:
    """JSON dosyasından dataset yükle."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        return [data]


def load_dataset_from_jsonl(path: str) -> List[Dict]:
    """JSONL dosyasından dataset yükle."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> List[Dict]:
    """
    HuggingFace'den dataset yükle.
    
    Args:
        dataset_name: HF dataset adı (ör: "alibayram/turkish_instructions_150k")
        split: Dataset split'i
        max_samples: Max örnek sayısı
        streaming: Streaming modda yükle
    
    Returns:
        Veri listesi
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠️ datasets kütüphanesi yüklü değil: pip install datasets")
        return []
    
    try:
        if streaming:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            data = []
            for i, item in enumerate(ds):
                if max_samples and i >= max_samples:
                    break
                data.append(dict(item))
            return data
        else:
            ds = load_dataset(dataset_name, split=split)
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))
            return [dict(item) for item in ds]
    except Exception as e:
        print(f"⚠️ Dataset yüklenemedi: {dataset_name} - {e}")
        return []


def convert_to_chat_format(
    data: List[Dict],
    instruction_key: str = "instruction",
    output_key: str = "output",
    input_key: Optional[str] = "input",
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Alpaca/instruction formatını chat formatına dönüştür.
    
    Alpaca format:
        {"instruction": "...", "input": "...", "output": "..."}
    
    Chat format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    chat_data = []
    
    for item in data:
        instruction = item.get(instruction_key, "")
        output = item.get(output_key, "")
        
        if not instruction or not output:
            continue
        
        # Input varsa instruction'a ekle
        input_text = item.get(input_key, "") if input_key else ""
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})
        
        chat_data.append({"messages": messages})
    
    return chat_data


def convert_cot_format(
    data: List[Dict],
    question_key: str = "question",
    reasoning_key: str = "reasoning",
    answer_key: str = "answer",
) -> List[Dict[str, str]]:
    """
    Mevcut CoT verisini standart formata dönüştür.
    
    Input format:
        {"question": "...", "reasoning": "...", "answer": "..."}
    
    Output format:
        {"instruction": "...", "thinking": "...", "response": "..."}
    """
    converted = []
    
    for item in data:
        question = item.get(question_key, "")
        reasoning = item.get(reasoning_key, "")
        answer = item.get(answer_key, "")
        
        if not question:
            continue
        
        converted.append({
            "instruction": question,
            "thinking": reasoning,
            "response": answer,
        })
    
    return converted


def create_data_collator(tokenizer, max_length: int = 2048):
    """
    DataLoader için collate fonksiyonu oluştur.
    Dynamic padding ile batch oluşturma.
    """
    def collate_fn(batch):
        # Batch'teki en uzun sequence'ı bul
        max_len = min(
            max(len(item["input_ids"]) for item in batch),
            max_length,
        )
        
        input_ids = []
        labels = []
        attention_masks = []
        
        for item in batch:
            ids = item["input_ids"][:max_len]
            lbl = item["labels"][:max_len]
            mask = item["attention_mask"][:max_len]
            
            # Padding
            pad_len = max_len - len(ids)
            ids = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100)])
            mask = torch.cat([mask, torch.zeros(pad_len)])
            
            input_ids.append(ids)
            labels.append(lbl)
            attention_masks.append(mask)
        
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks),
        }
    
    return collate_fn
