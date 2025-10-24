"""
🔍🤖 QWEN + DERİN WEB ARAŞTIRMA
Ultra Kapsamlı Akıllı Web Asistanı

Bu sistem:
- Derin web taraması (DuckDuckGo, Wikipedia, güvenilir kaynaklar)
- Çoklu kaynak toplama ve birleştirme
- Qwen 2.5-1.5B ile akıllı analiz ve özet
- Gerçek zamanlı bilgi toplama
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.deep_web_researcher import DeepWebResearcher
import os
import sys


class QwenDeepWebAssistant:
    """Qwen + Derin Web Entegre Asistan"""
    
    def __init__(self, model_path="./qwen-model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.researcher = None
        
        print("\n" + "🔍"*40)
        print("🤖 QWEN + DERİN WEB ASİSTANI BAŞLATILIYOR")
        print("🔍"*40)
        
        self.load_qwen()
        self.init_researcher()
    
    def load_qwen(self):
        """Qwen modelini yükle"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model bulunamadı: {self.model_path}")
        
        print("\n📂 Qwen modeli yükleniyor...")
        
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
        
        print(f"✅ Qwen yüklendi! (Cihaz: {self.device.upper()})")
    
    def init_researcher(self):
        """Derin web araştırmacıyı başlat"""
        print("🔍 Derin web sistemi başlatılıyor...")
        self.researcher = DeepWebResearcher()
        print("✅ Derin web sistemi hazır!")
    
    def ask_qwen(self, prompt, max_tokens=512):
        """Qwen'e soru sor"""
        messages = [
            {"role": "system", "content": "Sen kapsamlı web araştırması yapan uzman bir asistansın."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response
    
    def deep_research(self, topic, max_sources=10, verbose=True):
        """
        Derin web araştırması + Qwen analizi
        
        Args:
            topic: Araştırma konusu
            max_sources: Maksimum kaynak sayısı
            verbose: Detaylı çıktı
            
        Returns:
            dict: Araştırma sonuçları ve Qwen özeti
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 DERİN ARAŞTIRMA: {topic}")
            print(f"{'='*60}\n")
        
        # 1. Derin web taraması
        if verbose:
            print("📡 Derin web taraması başladı...")
        
        results = self.researcher.deep_research(topic, max_sources=max_sources)
        
        if not results:
            return {
                'success': False,
                'error': 'Hiçbir kaynak bulunamadı',
                'sources': [],
                'summary': None
            }
        
        if verbose:
            print(f"\n✅ {len(results)} kaynak bulundu!\n")
            print("📚 BULUNAN KAYNAKLAR:")
            print("-"*60)
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. {result['title']}")
                print(f"   🔗 {result.get('url', 'URL yok')[:80]}")
                print(f"   📝 {result.get('snippet', '')[:100]}...")
                print()
        
        # 2. Bilgileri birleştir
        combined_info = f"'{topic}' konusu hakkında {len(results)} kaynaktan toplanan bilgiler:\n\n"
        
        for i, result in enumerate(results[:10], 1):
            combined_info += f"KAYNAK {i} - {result['source']}:\n"
            combined_info += f"Başlık: {result['title']}\n"
            
            if result.get('snippet'):
                combined_info += f"Özet: {result['snippet']}\n"
            
            if result.get('content'):
                combined_info += f"İçerik: {result['content'][:500]}\n"
            
            combined_info += "\n"
        
        # 3. Qwen ile analiz
        if verbose:
            print("🤖 Qwen tüm kaynakları analiz ediyor...\n")
        
        prompt = f"""{combined_info}

Yukarıdaki {len(results)} kaynağa dayanarak '{topic}' hakkında kapsamlı bir özet hazırla:

1. Ana tanım ve genel bakış
2. Önemli özellikler ve noktalar
3. Güncel gelişmeler (varsa)
4. Pratik uygulamalar
5. Sonuç ve değerlendirme

Detaylı, bilgilendirici ve yapılandırılmış bir özet yaz:"""
        
        summary = self.ask_qwen(prompt, max_tokens=700)
        
        if verbose:
            print("="*60)
            print("📊 QWEN ANALİZİ VE ÖZETİ")
            print("="*60)
            print(summary)
            print("\n" + "="*60)
        
        return {
            'success': True,
            'sources': results,
            'source_count': len(results),
            'summary': summary,
            'topic': topic
        }
    
    def interactive_mode(self):
        """İnteraktif sohbet modu"""
        print("\n" + "="*60)
        print("💬 İNTERAKTİF DERIN ARAŞTIRMA MODU")
        print("="*60)
        print("\n📋 Komutlar:")
        print("  • araştır: [Konu] - Orta araştırma (10 kaynak)")
        print("  • hızlı: [Konu] - Hızlı araştırma (5 kaynak)")
        print("  • tam: [Konu] - Tam araştırma (15+ kaynak)")
        print("  • soru: [Soru] - Direkt soru (araştırma yok)")
        print("  • q - Çıkış")
        print("\n💡 Veya direkt konu yazın!")
        print("\n📝 Örnekler:")
        print("  → Machine Learning")
        print("  → araştır: Quantum Computing")
        print("  → tam: Deep Learning gelişmeleri\n")
        
        while True:
            user_input = input("👤 Siz: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit', 'çıkış']:
                print("\n👋 Görüşmek üzere!")
                break
            
            if not user_input:
                continue
            
            try:
                if user_input.startswith("araştır:"):
                    topic = user_input[8:].strip()
                    self.deep_research(topic, max_sources=10)
                
                elif user_input.startswith("hızlı:"):
                    topic = user_input[6:].strip()
                    self.deep_research(topic, max_sources=5)
                
                elif user_input.startswith("tam:"):
                    topic = user_input[4:].strip()
                    self.deep_research(topic, max_sources=15)
                
                elif user_input.startswith("soru:"):
                    question = user_input[5:].strip()
                    print("\n🤖 Qwen: ", end="", flush=True)
                    response = self.ask_qwen(question)
                    print(response + "\n")
                
                else:
                    # Otomatik araştırma
                    self.deep_research(user_input, max_sources=8)
            
            except KeyboardInterrupt:
                print("\n\n⚠️ İşlem iptal edildi.\n")
            except Exception as e:
                print(f"\n❌ Hata: {e}\n")


def main():
    """Ana program"""
    try:
        # Asistan oluştur
        assistant = QwenDeepWebAssistant()
        
        print("\n🎯 SİSTEM HAZIR!")
        print("\n📊 Yetenekler:")
        print("  ✅ DuckDuckGo web araması")
        print("  ✅ Wikipedia entegrasyonu")
        print("  ✅ Güvenilir kaynaklardan toplama")
        print("  ✅ Qwen ile akıllı analiz")
        print("  ✅ Çoklu kaynak birleştirme")
        
        # Hızlı demo
        print("\n" + "="*60)
        print("🎬 HIZLI DEMO")
        print("="*60)
        
        demo_topic = "Python programlama"
        print(f"\n📝 Demo Araştırma: {demo_topic}")
        
        result = assistant.deep_research(demo_topic, max_sources=5)
        
        if result['success']:
            print(f"\n✅ Demo tamamlandı! {result['source_count']} kaynak kullanıldı.")
        
        # İnteraktif mod
        print("\n\n🚀 İnteraktif moda geçiliyor...\n")
        assistant.interactive_mode()
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("💡 Önce 'python download_qwen.py' çalıştırın.")
    except ImportError as e:
        print(f"\n❌ Gerekli modüller yok: {e}")
        print("💡 Yükleyin: pip install transformers torch requests beautifulsoup4 wikipedia")
    except KeyboardInterrupt:
        print("\n\n👋 Program sonlandırıldı. Hoşça kalın!")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
