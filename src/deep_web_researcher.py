"""
🔍 DERİN WEB ARAŞTIRMACI
Geniş kapsamlı web tarama ve özet çıkarma sistemi
"""

import os
import time
import json
from datetime import datetime
from urllib.parse import urlparse, urljoin
import re

# Web scraping
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Wikipedia
try:
    import wikipedia
    wikipedia.set_lang('tr')  # Türkçe öncelikli
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False


class DeepWebResearcher:
    """Derin web araştırma motoru"""
    
    def __init__(self):
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.visited_urls = set()
        self.results = []
        
        # Arama motorları ve kaynaklar
        self.search_engines = {
            'DuckDuckGo': 'https://html.duckduckgo.com/html/?q=',
            'Google Scholar': 'https://scholar.google.com/scholar?q=',
        }
        
        # Güvenilir kaynaklar
        self.trusted_sources = [
            'wikipedia.org',
            'github.com',
            'stackoverflow.com',
            'python.org',
            'medium.com',
            'towardsdatascience.com',
            'arxiv.org',
            'researchgate.net',
            'springer.com',
            'sciencedirect.com',
            'news.ycombinator.com',
            'reddit.com',
            'bbc.com',
            'cnn.com',
            'nytimes.com',
            'theguardian.com',
            'techcrunch.com',
            'wired.com',
            'forbes.com',
            'economist.com'
        ]
    
    def is_accessible(self, url):
        """URL'nin erişilebilir olup olmadığını kontrol et"""
        try:
            response = self.session.head(url, headers=self.headers, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def extract_text_from_url(self, url, max_length=2000):
        """URL'den metin çıkar"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Script ve style etiketlerini kaldır
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Ana içeriği bul
            main_content = soup.find(['article', 'main', 'div'], class_=re.compile('content|article|post|entry'))
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Temizle
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = ' '.join(lines)
            
            # Çok uzunsa kısalt
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
            
        except Exception as e:
            return f"[İçerik çıkarılamadı: {str(e)}]"
    
    def extract_links_from_page(self, url, base_url):
        """Sayfadan linkleri çıkar"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                # Tam URL'ye çevir
                full_url = urljoin(base_url, href)
                
                # Sadece http/https
                if full_url.startswith(('http://', 'https://')):
                    links.append(full_url)
            
            return list(set(links))[:20]  # İlk 20 benzersiz link
            
        except Exception as e:
            return []
    
    def search_duckduckgo(self, query, max_results=10):
        """DuckDuckGo ile arama yap"""
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            response = self.session.get(search_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:max_results]:
                try:
                    title_tag = result.find('a', class_='result__a')
                    snippet_tag = result.find('a', class_='result__snippet')
                    
                    if title_tag and title_tag.get('href'):
                        title = title_tag.get_text(strip=True)
                        url = title_tag['href']
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                        
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'DuckDuckGo'
                        })
                except:
                    continue
            
            return results
            
        except Exception as e:
            print(f"⚠️ DuckDuckGo araması başarısız: {e}")
            return []
    
    def search_wikipedia(self, query, max_results=5):
        """Wikipedia'da ara"""
        if not WIKIPEDIA_AVAILABLE:
            return []
        
        try:
            search_results = wikipedia.search(query, results=max_results)
            results = []
            
            for title in search_results[:max_results]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    results.append({
                        'title': page.title,
                        'url': page.url,
                        'snippet': page.summary[:300] + "...",
                        'source': 'Wikipedia'
                    })
                except:
                    continue
            
            return results
            
        except Exception as e:
            return []
    
    def deep_research(self, topic, max_sources=15, include_links=True):
        """Derin araştırma yap"""
        print("\n" + "🔍"*30)
        print(f"🌐 DERİN WEB ARAŞTIRMASI: {topic}")
        print("🔍"*30)
        
        all_results = []
        
        # 1. Wikipedia'da ara
        print("\n📚 Wikipedia araştırılıyor...")
        wiki_results = self.search_wikipedia(topic, max_results=3)
        all_results.extend(wiki_results)
        print(f"✅ {len(wiki_results)} Wikipedia sonucu bulundu")
        
        # 2. DuckDuckGo'da ara
        print("\n🔍 DuckDuckGo araştırılıyor...")
        ddg_results = self.search_duckduckgo(topic, max_results=max_sources - len(all_results))
        all_results.extend(ddg_results)
        print(f"✅ {len(ddg_results)} web sonucu bulundu")
        
        # 3. Her sonuç için içerik çıkar
        print("\n📄 İçerikler çıkarılıyor...")
        for i, result in enumerate(all_results[:max_sources], 1):
            print(f"  {i}/{len(all_results[:max_sources])} {result['url'][:50]}...")
            
            # İçeriği çıkar
            content = self.extract_text_from_url(result['url'], max_length=500)
            result['content'] = content
            
            # Linkleri çıkar (isteğe bağlı)
            if include_links:
                links = self.extract_links_from_page(result['url'], result['url'])
                result['related_links'] = links[:5]  # İlk 5 link
            
            time.sleep(0.5)  # Rate limiting
        
        self.results = all_results[:max_sources]
        
        print("\n" + "="*60)
        print(f"🎉 TOPLAM {len(self.results)} KAYNAK TARANDIÇIKTI!")
        print("="*60)
        
        return self.results
    
    def generate_ai_summary(self, max_depth=3):
        """AI destekli derin özet oluştur - linklere girerek"""
        if not self.results:
            return "Henüz araştırma yapılmadı."
        
        print("\n" + "🧠"*30)
        print("AI DESTEKLI DERİN ANALİZ BAŞLIYOR")
        print("🧠"*30)
        
        all_contents = []
        processed_urls = set()
        
        # 1. Ana sonuçları işle
        print("\n📊 Aşama 1: Ana kaynaklar analiz ediliyor...")
        for i, result in enumerate(self.results, 1):
            print(f"  {i}/{len(self.results)} {result['title'][:50]}...")
            
            url = result['url']
            if url not in processed_urls:
                content_data = {
                    'url': url,
                    'title': result['title'],
                    'source': result.get('source', 'Web'),
                    'content': result.get('content', ''),
                    'snippet': result.get('snippet', ''),
                    'depth': 0
                }
                all_contents.append(content_data)
                processed_urls.add(url)
        
        # 2. İlgili linkleri de tara (derin analiz)
        if max_depth > 0:
            print(f"\n🔍 Aşama 2: İlgili linkler taranıyor (derinlik: {max_depth})...")
            
            links_to_process = []
            for result in self.results:
                if 'related_links' in result:
                    links_to_process.extend(result['related_links'][:2])  # Her kaynaktan 2 link
            
            # Benzersiz linkler
            links_to_process = list(set(links_to_process))[:max_depth * 3]
            
            for i, link in enumerate(links_to_process, 1):
                if link in processed_urls:
                    continue
                
                print(f"  {i}/{len(links_to_process)} {link[:60]}...")
                
                try:
                    content = self.extract_text_from_url(link, max_length=1000)
                    if content and len(content) > 100:
                        content_data = {
                            'url': link,
                            'title': self.extract_title_from_url(link),
                            'source': 'Related Link',
                            'content': content,
                            'snippet': content[:200],
                            'depth': 1
                        }
                        all_contents.append(content_data)
                        processed_urls.add(link)
                    time.sleep(0.5)
                except:
                    continue
        
        # 3. Tüm içerikleri birleştir ve özet oluştur
        print(f"\n🧠 Aşama 3: AI özeti oluşturuluyor...")
        print(f"📊 Toplam işlenen kaynak: {len(all_contents)}")
        
        # Ana özet
        summary = []
        summary.append("\n" + "🌟"*30)
        summary.append("AI DESTEKLI KAPSAMLI ARAŞTIRMA ÖZETİ")
        summary.append("🌟"*30 + "\n")
        
        # İstatistikler
        summary.append(f"📊 ARAŞTIRMA İSTATİSTİKLERİ:")
        summary.append(f"  • Toplam Kaynak: {len(all_contents)}")
        summary.append(f"  • Ana Kaynaklar: {len([c for c in all_contents if c['depth'] == 0])}")
        summary.append(f"  • İlgili Linkler: {len([c for c in all_contents if c['depth'] > 0])}")
        
        # Kaynak türlerine göre dağılım
        source_types = {}
        for content in all_contents:
            src = content['source']
            source_types[src] = source_types.get(src, 0) + 1
        
        summary.append(f"\n📚 KAYNAK DAĞILIMI:")
        for src, count in sorted(source_types.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  • {src}: {count} kaynak")
        
        # Anahtar kelime analizi
        all_text = " ".join([c['content'] for c in all_contents if c['content']])
        keywords = self.extract_keywords(all_text)
        
        summary.append(f"\n🔑 ANAHTAR KELİMELER:")
        for keyword, count in keywords[:10]:
            summary.append(f"  • {keyword}: {count} kez geçiyor")
        
        # Detaylı özet - her kaynak için
        summary.append("\n" + "="*60)
        summary.append("📑 KAYNAK BAZLI DETAYLI ANALİZ:")
        summary.append("="*60)
        
        for i, content in enumerate(all_contents, 1):
            depth_indicator = "  " * content['depth'] + ("🔗" if content['depth'] > 0 else "📄")
            summary.append(f"\n{i}. {depth_indicator} {content['title']}")
            summary.append(f"   🌐 {content['url'][:80]}")
            summary.append(f"   📊 Kaynak: {content['source']}")
            
            # İçerik özeti
            if content['content']:
                # İlk 3 cümleyi al
                sentences = content['content'].split('.')[:3]
                content_summary = '. '.join(sentences) + '.'
                summary.append(f"   💬 {content_summary[:300]}...")
            
            summary.append("")
        
        # Genel sonuç ve öneriler
        summary.append("\n" + "="*60)
        summary.append("🎯 GENEL DEĞERLENDİRME VE ÖNERİLER:")
        summary.append("="*60)
        
        summary.append(f"\n✅ Bu araştırmada {len(all_contents)} farklı kaynak incelendi.")
        summary.append(f"✅ En çok bahsedilen konular: {', '.join([k for k, _ in keywords[:5]])}")
        summary.append(f"✅ Güvenilir kaynaklar öncelikli olarak tarandı.")
        
        # En değerli kaynakları öner
        valuable_sources = [c for c in all_contents if len(c.get('content', '')) > 500][:5]
        if valuable_sources:
            summary.append(f"\n📚 ÖNERİLEN DETAYLI KAYNAKLAR:")
            for src in valuable_sources:
                summary.append(f"  • {src['title']}")
                summary.append(f"    {src['url']}")
        
        summary.append("\n" + "🎉"*30)
        summary.append("ARAŞTIRMA TAMAMLANDI!")
        summary.append("🎉"*30)
        
        return "\n".join(summary)
    
    def extract_title_from_url(self, url):
        """URL'den başlık çıkar"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            return title.get_text(strip=True) if title else urlparse(url).netloc
        except:
            return urlparse(url).netloc
    
    def extract_keywords(self, text, top_n=15):
        """Metinden anahtar kelimeleri çıkar"""
        # Basit kelime frekans analizi
        words = re.findall(r'\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}\b', text.lower())
        
        # Stop words (yaygın kelimeler)
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'will', 'been', 'were', 
            'their', 'what', 'which', 'when', 'where', 'they', 'there',
            'about', 'would', 'could', 'should', 'these', 'those',
            'için', 'olan', 'olarak', 'daha', 'iken', 'sonra', 'önce',
            'çok', 'gibi', 'kadar', 'ise', 'bile', 'ancak'
        }
        
        # Filtrele ve say
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sırala ve en çok geçenleri döndür
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    def generate_summary(self):
        """Araştırma özeti oluştur"""
        if not self.results:
            return "Henüz araştırma yapılmadı."
        
        summary = []
        summary.append("\n" + "📋"*30)
        summary.append("ARAŞTIRMA ÖZETİ")
        summary.append("📋"*30 + "\n")
        
        # Kaynak istatistikleri
        summary.append(f"📊 Toplam Kaynak: {len(self.results)}")
        
        sources_count = {}
        for result in self.results:
            source = result.get('source', 'Bilinmeyen')
            sources_count[source] = sources_count.get(source, 0) + 1
        
        summary.append("\n📚 Kaynak Dağılımı:")
        for source, count in sources_count.items():
            summary.append(f"  • {source}: {count} kaynak")
        
        # Detaylı sonuçlar
        summary.append("\n" + "="*60)
        summary.append("📑 DETAYLI SONUÇLAR:")
        summary.append("="*60 + "\n")
        
        for i, result in enumerate(self.results, 1):
            summary.append(f"\n{i}. 📄 {result['title']}")
            summary.append(f"   🔗 {result['url']}")
            summary.append(f"   📝 {result.get('snippet', '')}")
            
            if result.get('content'):
                content_preview = result['content'][:200]
                summary.append(f"\n   💬 Özet: {content_preview}...\n")
            
            if result.get('related_links'):
                summary.append(f"   🔗 İlgili Linkler ({len(result['related_links'])}):")
                for link in result['related_links'][:3]:
                    summary.append(f"      • {link[:80]}")
            
            summary.append("\n" + "-"*60)
        
        return "\n".join(summary)
    
    def save_results(self, filename=None):
        """Sonuçları dosyaya kaydet"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_results_{timestamp}.json"
        
        os.makedirs("research_data", exist_ok=True)
        filepath = os.path.join("research_data", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Sonuçlar kaydedildi: {filepath}")
        return filepath


def demo_deep_research():
    """Demo araştırma"""
    researcher = DeepWebResearcher()
    
    print("\n🔍 DERİN WEB ARAŞTIRMACI")
    print("="*60)
    print("Bu sistem web'i tarayarak kapsamlı araştırma yapar.")
    print("Erişilebilen tüm kaynakları tarar ve özet çıkarır.")
    print("="*60)
    
    topic = input("\n🔎 Araştırma konusu: ").strip()
    
    if not topic:
        topic = "artificial intelligence machine learning"
    
    max_sources = input("📊 Maksimum kaynak sayısı (varsayılan 10): ").strip()
    max_sources = int(max_sources) if max_sources.isdigit() else 10
    
    # Araştırmayı başlat
    results = researcher.deep_research(topic, max_sources=max_sources, include_links=True)
    
    # Özet göster
    summary = researcher.generate_summary()
    print(summary)
    
    # Kaydetme seçeneği
    save = input("\n💾 Sonuçları kaydetmek ister misiniz? (e/h): ").strip().lower()
    if save == 'e':
        researcher.save_results()
    
    return researcher


if __name__ == "__main__":
    if not REQUESTS_AVAILABLE:
        print("❌ requests ve beautifulsoup4 kütüphaneleri gerekli!")
        print("💡 Kurulum: pip install requests beautifulsoup4")
    else:
        demo_deep_research()
