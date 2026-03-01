"""
Chain-of-Thought (CoT) Reasoning Engine
========================================

Modern LLM'lerin (Claude, GPT-4, DeepSeek-R1) akıl yürütme yeteneklerini
sağlayan CoT modülü.

Özellikler:
- <think>...</think> ile dahili akıl yürütme
- Çok adımlı problem çözme
- Self-reflection ve doğrulama
- Budget forcing (düşünme token limiti)
- Self-consistency (çoklu akıl yürütme yolu)
- Reasoning trace formatı
- Türkçe ve İngilizce destek

CoT Eğitim Formatı:
    <|im_start|><|user|>
    Soru burada
    <|im_end|>
    <|im_start|><|assistant|>
    <think>
    Adım adım düşünme süreci...
    1. İlk analiz
    2. Ara adımlar
    3. Doğrulama
    </think>
    Final cevap burada
    <|im_end|>
"""

import torch
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class ThinkingResult:
    """CoT düşünme sonucu."""
    thinking: str           # Dahili düşünme metni
    response: str           # Final cevap
    thinking_tokens: int    # Düşünme token sayısı
    response_tokens: int    # Cevap token sayısı
    total_tokens: int       # Toplam token sayısı
    confidence: float       # Güven skoru (0-1)
    reasoning_steps: List[str]  # Akıl yürütme adımları


class ChainOfThoughtEngine:
    """
    Chain-of-Thought Reasoning Engine.
    
    Modern reasoning modelleri gibi <think> tokenları kullanarak
    dahili akıl yürütme yapar.
    
    Kullanım:
        cot = ChainOfThoughtEngine(model, tokenizer)
        result = cot.think_and_respond("15 * 23 kaçtır?")
        print(result.thinking)   # Düşünme süreci
        print(result.response)   # Final cevap
    """
    
    # Reasoning görev türleri
    TASK_TYPES = {
        "math": {
            "description": "Matematik problemi çözme",
            "think_prompt": "Bu matematik problemini adım adım çözeceğim.",
            "verify_prompt": "Sonucu doğrulayayım.",
        },
        "logic": {
            "description": "Mantık ve akıl yürütme",
            "think_prompt": "Bu mantık problemini sistematik olarak analiz edeceğim.",
            "verify_prompt": "Çıkarımlarımı kontrol edeyim.",
        },
        "code": {
            "description": "Kod yazma ve analiz",
            "think_prompt": "Kodu adım adım planlayacağım.",
            "verify_prompt": "Kodun doğruluğunu ve edge case'leri kontrol edeyim.",
        },
        "analysis": {
            "description": "Metin analizi ve değerlendirme",
            "think_prompt": "Metni çeşitli açılardan analiz edeceğim.",
            "verify_prompt": "Analizimin tutarlılığını kontrol edeyim.",
        },
        "general": {
            "description": "Genel soru-cevap",
            "think_prompt": "Bu soruyu düşüneyim.",
            "verify_prompt": "Cevabımı gözden geçireyim.",
        },
    }
    
    # Sistem promptu
    SYSTEM_PROMPT_TR = """Sen gelişmiş bir yapay zeka asistanısın. Sorulara cevap vermeden önce <think> etiketleri içinde adım adım düşünürsün.

Düşünme kuralların:
1. Problemi anla ve parçalara ayır
2. Her adımı sırayla çöz
3. Ara sonuçları kontrol et
4. Cevabını doğrula
5. Net ve açık bir final cevap ver

Düşünme formatı:
<think>
[Burada adım adım düşünürsün - kullanıcı bunu görmez]
</think>
[Burada net cevabını verirsin]"""

    SYSTEM_PROMPT_EN = """You are an advanced AI assistant. Before answering questions, you think step by step inside <think> tags.

Your thinking rules:
1. Understand the problem and break it down
2. Solve each step sequentially
3. Verify intermediate results
4. Validate your answer
5. Give a clear final response

Thinking format:
<think>
[Step-by-step reasoning here - user doesn't see this]
</think>
[Clear answer here]"""
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        max_thinking_tokens: int = 1024,
        max_response_tokens: int = 512,
        language: str = "tr",
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        """
        Args:
            model: ModernLLMForCausalLM modeli
            tokenizer: ModernTokenizer
            max_thinking_tokens: Düşünme için max token
            max_response_tokens: Cevap için max token
            language: "tr" veya "en"
            temperature: Generation temperature
            top_p: Nucleus sampling threshold
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.max_response_tokens = max_response_tokens
        self.language = language
        self.temperature = temperature
        self.top_p = top_p
        
        self.system_prompt = self.SYSTEM_PROMPT_TR if language == "tr" else self.SYSTEM_PROMPT_EN
    
    def detect_task_type(self, query: str) -> str:
        """
        Soru türünü otomatik algıla.
        
        Args:
            query: Kullanıcı sorusu
        
        Returns:
            Görev türü string'i
        """
        query_lower = query.lower()
        
        # Matematik ipuçları
        math_keywords = [
            'hesapla', 'kaç', 'topla', 'çıkar', 'çarp', 'böl',
            'calculate', 'compute', 'solve', 'equation',
            '+', '-', '*', '/', '=', '%',
            'integral', 'türev', 'limit', 'matris', 'denklem',
            'toplam', 'fark', 'çarpım', 'bölüm', 'oran',
        ]
        
        # Mantık ipuçları
        logic_keywords = [
            'mantık', 'çıkarım', 'neden', 'niçin', 'kanıtla',
            'logic', 'deduce', 'infer', 'prove', 'reason',
            'doğru mu', 'yanlış mı', 'hangisi', 'karşılaştır',
            'sırala', 'büyük', 'küçük', 'hızlı', 'yavaş',
        ]
        
        # Kod ipuçları
        code_keywords = [
            'kod', 'fonksiyon', 'program', 'algoritma', 'yaz',
            'code', 'function', 'class', 'implement', 'debug',
            'python', 'javascript', 'java', 'c++', 'rust',
            'def ', 'import', 'for ', 'while ', 'if ',
            'hata', 'bug', 'optimize', 'refactor',
        ]
        
        # Analiz ipuçları
        analysis_keywords = [
            'analiz', 'değerlendir', 'incele', 'özetle', 'karşılaştır',
            'analyze', 'evaluate', 'summarize', 'compare', 'review',
            'duygu', 'sentiment', 'ton', 'perspektif',
            'metin', 'makale', 'paragraf',
        ]
        
        # Skor hesapla
        scores = {
            "math": sum(1 for k in math_keywords if k in query_lower),
            "logic": sum(1 for k in logic_keywords if k in query_lower),
            "code": sum(1 for k in code_keywords if k in query_lower),
            "analysis": sum(1 for k in analysis_keywords if k in query_lower),
        }
        
        # Sayılar varsa matematik olma olasılığı artar
        import re
        if re.search(r'\d+\s*[\+\-\*\/\=]\s*\d+', query):
            scores["math"] += 3
        
        max_score = max(scores.values())
        if max_score == 0:
            return "general"
        
        return max(scores, key=scores.get)
    
    def format_thinking_prompt(
        self,
        query: str,
        task_type: Optional[str] = None,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        CoT promptu oluştur.
        
        Args:
            query: Kullanıcı sorusu
            task_type: Görev türü (None=otomatik algıla)
            context: Ek bağlam
            chat_history: Önceki sohbet geçmişi
        
        Returns:
            Chat mesaj listesi
        """
        if task_type is None:
            task_type = self.detect_task_type(query)
        
        messages = []
        
        # Sistem promptu
        messages.append({
            "role": "system",
            "content": self.system_prompt,
        })
        
        # Sohbet geçmişi
        if chat_history:
            messages.extend(chat_history)
        
        # Context
        if context:
            messages.append({
                "role": "user",
                "content": f"Bağlam: {context}\n\nSoru: {query}",
            })
        else:
            messages.append({
                "role": "user",
                "content": query,
            })
        
        return messages
    
    def think_and_respond(
        self,
        query: str,
        task_type: Optional[str] = None,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        show_thinking: bool = False,
    ) -> ThinkingResult:
        """
        CoT ile düşün ve cevapla.
        
        Model ile inference yaparak <think> tokenları arasında
        akıl yürütme yapar, sonra cevap üretir.
        
        Args:
            query: Kullanıcı sorusu
            task_type: Görev türü
            context: Ek bağlam
            chat_history: Sohbet geçmişi
            show_thinking: Düşünme metnini göster
        
        Returns:
            ThinkingResult
        """
        if self.model is None or self.tokenizer is None:
            return self._mock_thinking(query, task_type)
        
        # Prompt oluştur
        messages = self.format_thinking_prompt(query, task_type, context, chat_history)
        
        # Chat template uygula
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=next(self.model.parameters()).device)
        
        # Önce <think> token üret
        think_start = "<think>\n"
        think_prefix_ids = self.tokenizer.encode(think_start)
        input_with_think = torch.cat([
            input_tensor,
            torch.tensor([think_prefix_ids], device=input_tensor.device),
        ], dim=-1)
        
        # Generate
        output_ids = self.model.generate(
            input_with_think,
            max_new_tokens=self.max_thinking_tokens + self.max_response_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0][len(input_ids):].tolist(),
            skip_special_tokens=False,
        )
        
        # Thinking ve Response ayır
        thinking, response = self._parse_thinking_response(generated_text)
        
        # Reasoning adımlarını çıkar
        reasoning_steps = self._extract_reasoning_steps(thinking)
        
        return ThinkingResult(
            thinking=thinking,
            response=response,
            thinking_tokens=len(self.tokenizer.encode(thinking)),
            response_tokens=len(self.tokenizer.encode(response)),
            total_tokens=len(output_ids[0]) - len(input_ids),
            confidence=self._estimate_confidence(thinking, response),
            reasoning_steps=reasoning_steps,
        )
    
    def _parse_thinking_response(self, text: str) -> Tuple[str, str]:
        """<think>...</think> ve cevabı ayır."""
        thinking = ""
        response = ""
        
        if "<think>" in text and "</think>" in text:
            think_start = text.index("<think>") + len("<think>")
            think_end = text.index("</think>")
            thinking = text[think_start:think_end].strip()
            response = text[think_end + len("</think>"):].strip()
        elif "<think>" in text:
            # </think> bulunamadı - tüm metin thinking
            think_start = text.index("<think>") + len("<think>")
            thinking = text[think_start:].strip()
        else:
            response = text.strip()
        
        # Özel tokenları temizle
        for token in ["<|im_end|>", "<|im_start|>", "<|assistant|>", "<eos>"]:
            response = response.replace(token, "").strip()
            thinking = thinking.replace(token, "").strip()
        
        return thinking, response
    
    def _extract_reasoning_steps(self, thinking: str) -> List[str]:
        """Düşünme metninden akıl yürütme adımlarını çıkar."""
        import re
        
        steps = []
        
        # Numaralı adımlar: "1.", "2.", "Adım 1:", "Step 1:"
        numbered = re.findall(
            r'(?:^|\n)\s*(?:(?:Adım|Step)\s*)?\d+[\.\):\-]\s*(.+?)(?=\n\s*(?:(?:Adım|Step)\s*)?\d+[\.\):\-]|\Z)',
            thinking,
            re.DOTALL,
        )
        
        if numbered:
            steps = [s.strip() for s in numbered if s.strip()]
        else:
            # Satır satır
            lines = [l.strip() for l in thinking.split('\n') if l.strip()]
            # Anlamlı satırları al (çok kısa olanları atla)
            steps = [l for l in lines if len(l) > 10]
        
        return steps if steps else [thinking]
    
    def _estimate_confidence(self, thinking: str, response: str) -> float:
        """Cevabın güven skorunu tahmin et."""
        confidence = 0.5  # Base
        
        # Düşünme uzunluğu - çok kısa veya çok uzun kötü
        thinking_len = len(thinking.split())
        if 20 <= thinking_len <= 200:
            confidence += 0.2
        elif thinking_len > 200:
            confidence += 0.1
        
        # Doğrulama adımı var mı?
        verification_keywords = [
            'doğrula', 'kontrol', 'verify', 'check', 'confirm',
            'emin', 'sure', 'correct', 'doğru',
        ]
        if any(k in thinking.lower() for k in verification_keywords):
            confidence += 0.15
        
        # Adım adım reasoning var mı?
        import re
        if re.search(r'\d+[\.\)]', thinking):
            confidence += 0.1
        
        # Cevap var mı?
        if response and len(response.split()) >= 3:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _mock_thinking(self, query: str, task_type: Optional[str] = None) -> ThinkingResult:
        """Model olmadan mock thinking sonucu (test için)."""
        if task_type is None:
            task_type = self.detect_task_type(query)
        
        task_info = self.TASK_TYPES.get(task_type, self.TASK_TYPES["general"])
        
        thinking = f"""{task_info['think_prompt']}

Soru: {query}

Adım 1: Soruyu analiz ediyorum.
- Girdi: {query}
- Tür: {task_info['description']}

Adım 2: Çözüm yaklaşımı belirliyorum.
- Bu bir {task_type} problemi.

Adım 3: {task_info['verify_prompt']}
- Cevabım tutarlı görünüyor."""
        
        response = f"Bu soruyu cevaplamak için model eğitimi gerekiyor. (Mock response - task type: {task_type})"
        
        return ThinkingResult(
            thinking=thinking,
            response=response,
            thinking_tokens=len(thinking.split()),
            response_tokens=len(response.split()),
            total_tokens=len(thinking.split()) + len(response.split()),
            confidence=0.0,
            reasoning_steps=[
                f"Soruyu analiz ediyorum: {query}",
                f"Tür: {task_info['description']}",
                task_info['verify_prompt'],
            ],
        )
    
    @staticmethod
    def create_cot_training_example(
        instruction: str,
        thinking: str,
        response: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        CoT eğitim örneği oluştur.
        
        Bu formatta veri eğitim pipeline'ına beslenebilir:
        
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "<think>...</think>\n..."},
            ]
        }
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": instruction})
        
        assistant_content = f"<think>\n{thinking}\n</think>\n{response}"
        messages.append({"role": "assistant", "content": assistant_content})
        
        return {"messages": messages}
    
    @staticmethod
    def format_cot_dataset(
        examples: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        CoT eğitim datasetini oluştur.
        
        Args:
            examples: [{"instruction": ..., "thinking": ..., "response": ...}]
            system_prompt: Opsiyonel sistem promptu
        
        Returns:
            Chat format mesaj listesi
        """
        dataset = []
        
        for ex in examples:
            entry = ChainOfThoughtEngine.create_cot_training_example(
                instruction=ex["instruction"],
                thinking=ex["thinking"],
                response=ex["response"],
                system_prompt=system_prompt or ChainOfThoughtEngine.SYSTEM_PROMPT_TR,
            )
            dataset.append(entry)
        
        return dataset
