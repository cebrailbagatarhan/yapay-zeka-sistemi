"""
🌐🤖 QWEN + WEB SCRAPING = Akıllı Web Asistanı

Bu modül Qwen modelini web scraping ile birleştirerek
gerçek zamanlı web araştırması yapan akıllı bir asistan oluşturur.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import os
import time


class QwenWebAssistant:
    """Qwen modeli ile web araştırma yapan asistan"""
    
    def __init__(self, model_path="./qwen-model"):
        """
        Args:
            model_path: Qwen model dizini
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.conversation_history = []
        
        print("🚀 Qwen Web Asistanı Başlatılıyor...")
        self.load_model()
    
    def load_model(self):
        """Qwen modelini yükle"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model bulunamadı: {self.model_path}")
        
        print("📂 Model yükleniyor...")
        start = time.time()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
            self.model = self.model.to(self.device)
        
        load_time = time.time() - start
        print(f"✅ Model yüklendi! ({load_time:.1f}s, Cihaz: {self.device.upper()})")
    
    def fetch_webpage(self, url, timeout=10):
        """
        Web sayfasını indir ve içeriği çıkar
        
        Args:
            url: Web sayfası URL'i
            timeout: Maksimum bekleme süresi
            
        Returns:
            dict: {'success': bool, 'title': str, 'text': str, 'error': str}
        """
        try:
            print(f"🔍 Sayfa indiriliyor: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Başlık
            title = soup.find('title')
            title = title.get_text().strip() if title else "Başlık bulunamadı"
            
            # Gereksiz etiketleri kaldır
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Metni çıkar
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())
            
            # İlk 5000 karakter
            text = text[:5000]
            
            print(f"✅ Sayfa indirildi: {title[:50]}...")
            
            return {
                'success': True,
                'title': title,
                'text': text,
                'error': None
            }
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            return {
                'success': False,
                'title': None,
                'text': None,
                'error': str(e)
            }
    
    def generate_response(self, prompt, max_tokens=512, temperature=0.7):
        """
        Qwen ile yanıt üret
        
        Args:
            prompt: Kullanıcı girdisi
            max_tokens: Maksimum token sayısı
            temperature: Yaratıcılık seviyesi
            
        Returns:
            str: Qwen'in yanıtı
        """
        messages = [
            {"role": "system", "content": "Sen yardımcı bir araştırma asistanısın. Web içeriklerini analiz edip kullanıcıya özetler sunarsın."},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Yanıt üret
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def analyze_webpage(self, url, question=None):
        """
        Web sayfasını indir ve Qwen ile analiz et
        
        Args:
            url: Web sayfası URL'i
            question: Opsiyonel soru
            
        Returns:
            str: Analiz sonucu
        """
        # Sayfa indir
        result = self.fetch_webpage(url)
        
        if not result['success']:
            return f"❌ Sayfa indirilemedi: {result['error']}"
        
        # Prompt hazırla
        if question:
            prompt = f"""Bu web sayfasının içeriğine dayanarak soruyu cevapla:

Başlık: {result['title']}

İçerik:
{result['text']}

Soru: {question}

Sadece sayfa içeriğindeki bilgilere dayanarak cevap ver:"""
        else:
            prompt = f"""Bu web sayfasının içeriğini özetle:

Başlık: {result['title']}

İçerik:
{result['text']}

Ana noktaları çıkararak kısa bir özet yaz (maksimum 200 kelime):"""
        
        print("🤖 Qwen analiz yapıyor...")
        return self.generate_response(prompt, max_tokens=400, temperature=0.5)
    
    def research_topic(self, topic, num_sources=3):
        """
        Bir konuyu araştır (Wikipedia + Google)
        
        Args:
            topic: Araştırma konusu
            num_sources: Kaç kaynak kullanılacak
            
        Returns:
            str: Araştırma özeti
        """
        print(f"\n🔍 '{topic}' konusu araştırılıyor...")
        
        # Wikipedia'dan başla
        wiki_url = f"https://tr.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
        sources = []
        result = self.fetch_webpage(wiki_url)
        
        if result['success']:
            sources.append({
                'url': wiki_url,
                'title': result['title'],
                'text': result['text'][:2000]
            })
            print(f"✅ Wikipedia kaynağı eklendi")
        
        # Tüm kaynakları birleştir
        if not sources:
            return f"❌ '{topic}' hakkında bilgi bulunamadı."
        
        # Özet oluştur
        combined_text = "\n\n".join([
            f"KAYNAK {i+1}: {s['title']}\n{s['text']}"
            for i, s in enumerate(sources)
        ])
        
        prompt = f"""Bu kaynaklara dayanarak '{topic}' hakkında kapsamlı bir özet hazırla:

{combined_text}

Özet:
1. Ana tanım nedir?
2. Önemli noktalar neler?
3. Pratik uygulamalar var mı?

Kısa ve öz bir özet yaz:"""
        
        print("🤖 Qwen özet hazırlıyor...")
        return self.generate_response(prompt, max_tokens=500, temperature=0.6)
    
    def interactive_mode(self):
        """İnteraktif sohbet modu"""
        print("\n" + "="*60)
        print("💬 QWEN WEB ASİSTANI - İNTERAKTİF MOD")
        print("="*60)
        print("\n📋 Komutlar:")
        print("  • web: [URL] - Web sayfasını analiz et")
        print("  • soru: [URL] | [Soru] - Sayfa hakkında soru sor")
        print("  • araştır: [Konu] - Konuyu araştır")
        print("  • q - Çıkış")
        print("\n💡 Veya direkt soru sorun!\n")
        
        while True:
            user_input = input("👤 Siz: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit', 'çıkış']:
                print("👋 Görüşmek üzere!")
                break
            
            if not user_input:
                continue
            
            # Komut kontrolü
            if user_input.startswith("web:"):
                url = user_input[4:].strip()
                response = self.analyze_webpage(url)
                print(f"🤖 Qwen: {response}\n")
                
            elif user_input.startswith("soru:"):
                parts = user_input[5:].split("|")
                if len(parts) == 2:
                    url = parts[0].strip()
                    question = parts[1].strip()
                    response = self.analyze_webpage(url, question)
                    print(f"🤖 Qwen: {response}\n")
                else:
                    print("❌ Format: soru: [URL] | [Soru]\n")
                    
            elif user_input.startswith("araştır:"):
                topic = user_input[8:].strip()
                response = self.research_topic(topic)
                print(f"🤖 Qwen: {response}\n")
                
            else:
                # Normal sohbet
                response = self.generate_response(user_input)
                print(f"🤖 Qwen: {response}\n")


def main():
    """Ana program"""
    print("\n" + "🌐"*30)
    print("🤖 QWEN WEB ASİSTANI 🌐")
    print("🌐"*30)
    
    try:
        assistant = QwenWebAssistant()
        
        # Demo örnekler
        print("\n📝 HIZLI DEMO")
        print("="*60)
        
        print("\n1️⃣ Wikipedia Analizi:")
        result = assistant.analyze_webpage("https://tr.wikipedia.org/wiki/Yapay_zeka")
        print(f"🤖 Sonuç: {result[:300]}...\n")
        
        print("\n2️⃣ Konu Araştırması:")
        result = assistant.research_topic("Python programlama")
        print(f"🤖 Sonuç: {result[:300]}...\n")
        
        # İnteraktif mod
        assistant.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
