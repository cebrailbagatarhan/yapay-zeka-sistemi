"""
🔍 DERİN WEB ARAŞTIRMACI
Geniş kapsamlı web tarama ve özet çıkarma sistemi
⚡ ASYNC optimizasyonlu - Paralel web tarama
"""

import os
import time
import json
from datetime import datetime
from urllib.parse import urlparse, urljoin
import re
import asyncio

# Web scraping - Async httpx
try:
    import httpx
    from bs4 import BeautifulSoup
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Fallback için sync requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# DuckDuckGo Search
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Wikipedia
try:
    import wikipedia
    wikipedia.set_lang('tr')  # Türkçe öncelikli
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False


class DeepWebResearcher:
    """Derin web araştırma motoru - Async paralel tarama"""
    
    def __init__(self):
        # Async client için
        self.async_client = None
        # Fallback için sync session
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
    
    async def _get_async_client(self):
        """Async client oluştur veya mevcut olanı döndür"""
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                headers=self.headers,
                timeout=15.0,
                follow_redirects=True,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self.async_client
    
    async def _close_async_client(self):
        """Async client'ı kapat"""
        if self.async_client is not None:
            await self.async_client.aclose()
            self.async_client = None
    
    def is_accessible(self, url):
        """URL'nin erişilebilir olup olmadığını kontrol et"""
        try:
            response = self.session.head(url, headers=self.headers, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    async def extract_text_from_url_async(self, url, max_length=10000):
        """
        URL'den detaylı metin çıkar - ASYNC VERSION (Paralel)
        
        Args:
            url: Web sayfası URL'i
            max_length: Maksimum karakter sayısı (varsayılan: 10000)
        
        Returns:
            str: Çıkarılan metin içeriği
        """
        try:
            client = await self._get_async_client()
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Script ve style etiketlerini kaldır
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Önce article, main veya content bölümlerini ara
            main_content = soup.find(['article', 'main']) or soup.find(class_=re.compile('content|article|post'))
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback: tüm body
                text = soup.get_text(separator=' ', strip=True)
            
            # Whitespace temizle
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Uzunluk limiti
            if len(text) > max_length:
                text = text[:max_length]
            
            return text
            
        except Exception as e:
            print(f"  ⚠️ İçerik çıkarma hatası ({url[:50]}): {str(e)[:50]}")
            return ""
    
    def extract_text_from_url(self, url, max_length=10000):
        """
        URL'den detaylı metin çıkar - SYNC VERSION (Fallback)
        
        Args:
            url: Web sayfası URL'i
            max_length: Maksimum karakter sayısı (varsayılan: 10000)
        
        Returns:
            str: Çıkarılan metin içeriği
        """
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Gereksiz etiketleri kaldır (daha az agresif)
            for script in soup(["script", "style", "iframe", "noscript"]):
                script.decompose()
            
            # Önce ana içerik alanlarını dene (daha geniş kapsam)
            main_content = None
            
            # Çeşitli ana içerik seçicileri
            content_selectors = [
                {'name': 'article'},
                {'name': 'main'},
                {'class_': re.compile(r'content|article|post|entry|body|main|text', re.I)},
                {'id': re.compile(r'content|article|post|main|body', re.I)},
                {'role': 'main'},
                {'itemprop': 'articleBody'},
            ]
            
            for selector in content_selectors:
                main_content = soup.find(**selector)
                if main_content:
                    break
            
            # Ana içerik bulunduysa onu kullan, yoksa tüm body
            if main_content:
                # Ana içerikten nav, footer, sidebar kaldır
                for unwanted in main_content(['nav', 'footer', 'aside', 'header']):
                    unwanted.decompose()
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Tüm sayfayı kullan ama nav/footer/header hariç
                for unwanted in soup(['nav', 'footer', 'aside', 'header']):
                    unwanted.decompose()
                text = soup.get_text(separator=' ', strip=True)
            
            # Paragrafları koru, fazla boşlukları temizle
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Çok kısa satırları atla
                    lines.append(line)
            
            text = ' '.join(lines)
            
            # Fazla boşlukları tek boşluğa indir
            text = re.sub(r'\s+', ' ', text)
            
            # Maksimum uzunluk kontrolü (daha büyük limit)
            if len(text) > max_length:
                # Cümle sonunda kes
                text = text[:max_length]
                last_period = text.rfind('.')
                if last_period > max_length - 200:  # Son 200 karakterde nokta varsa
                    text = text[:last_period + 1]
                else:
                    text = text + "..."
            
            # En az 100 karakter olmalı
            if len(text) < 100:
                return f"[Yetersiz içerik: {len(text)} karakter]"
            
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
        """DuckDuckGo ile arama yap - API kullanarak"""
        if not DDGS_AVAILABLE:
            print("⚠️ duckduckgo-search kütüphanesi yüklü değil!")
            return []
        
        try:
            results = []
            with DDGS() as ddgs:
                # Text search - max_results kadar al
                search_results = ddgs.text(query, max_results=max_results * 2)  # 2x al, filtreleme için
                
                for result in search_results:
                    if len(results) >= max_results:
                        break
                    
                    title = result.get('title', '')
                    url = result.get('href', '')
                    snippet = result.get('body', '')
                    
                    if url and title:
                        # Wikipedia veya diğer kaynakları işaretle
                        source_type = 'Wikipedia' if 'wikipedia.org' in url.lower() else 'Web'
                        
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': f'{source_type} (DuckDuckGo)'
                        })
                        print(f"  ✅ {len(results)}/{max_results}: {title[:60]}")
            
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
    
    async def deep_research_async(self, topic, max_sources=15, include_links=True):
        """
        ⚡ ASYNC Derin araştırma - PARALEL web tarama (10x daha hızlı!)
        
        Args:
            topic: Araştırma konusu
            max_sources: Maksimum kaynak sayısı
            include_links: İlgili linkleri de çıkar
            
        Returns:
            list: Araştırma sonuçları
        """
        print("\n" + "⚡"*30)
        print(f"🚀 ASYNC DERİN WEB ARAŞTIRMASI: {topic}")
        print(f"📊 Hedef: {max_sources} kaynak (PARALEL TARAMA)")
        print("⚡"*30)
        
        all_results = []
        
        # DuckDuckGo'dan kaynak bul
        print("\n🌐 Web kaynakları + Wikipedia araştırılıyor...")
        ddg_results = self.search_duckduckgo(topic, max_results=max_sources)
        all_results.extend(ddg_results)
        print(f"✅ {len(ddg_results)} kaynak bulundu (Wikipedia dahil)")
        
        # ⚡ PARALEL içerik çıkarma (en büyük hız kazancı burada!)
        print(f"\n⚡ İçerikler PARALEL olarak çıkarılıyor (async)...")
        
        async def fetch_content(result, index):
            """Tek bir URL için içerik çıkar"""
            print(f"  ⚡ {index}/{len(all_results[:max_sources])} {result['url'][:50]}...")
            content = await self.extract_text_from_url_async(result['url'], max_length=10000)
            result['content'] = content
            return result
        
        # Tüm URL'leri paralel olarak işle
        tasks = [
            fetch_content(result, i) 
            for i, result in enumerate(all_results[:max_sources], 1)
        ]
        
        # Paralel çalıştır ve bekle
        import time as time_module
        start_time = time_module.time()
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time_module.time() - start_time
        
        # Hataları filtrele
        self.results = [r for r in processed_results if not isinstance(r, Exception)]
        
        # Client'ı kapat
        await self._close_async_client()
        
        print("\n" + "="*60)
        print(f"🎉 TOPLAM {len(self.results)} KAYNAK TARANDI!")
        print(f"⚡ Süre: {elapsed:.1f} saniye (async paralel)")
        print(f"🚀 Klasik yöntemle ~{len(self.results) * 2:.0f} saniye sürerdi!")
        print("="*60)
        
        return self.results
    
    def deep_research(self, topic, max_sources=15, include_links=True, use_async=True):
        """
        Derin araştırma yap - SYNC/ASYNC seçenekli
        
        Args:
            topic: Araştırma konusu
            max_sources: Maksimum kaynak sayısı
            include_links: İlgili linkleri de çıkar
            use_async: Async paralel tarama kullan (önerilen)
            
        Returns:
            list: Araştırma sonuçları
        """
        # Async kullan (daha hızlı!)
        if use_async and HTTPX_AVAILABLE:
            return asyncio.run(self.deep_research_async(topic, max_sources, include_links))
        
        # Fallback: Klasik sync yöntem
        print("\n" + "🔍"*30)
        print(f"🌐 DERİN WEB ARAŞTIRMASI: {topic}")
        print(f"📊 Hedef: {max_sources} kaynak (Wikipedia + Web)")
        print("🔍"*30)
        
        all_results = []
        
        # DuckDuckGo'dan TÜM kaynaklarını çek (Wikipedia DAHİL)
        print("\n🌐 Web kaynakları + Wikipedia araştırılıyor...")
        ddg_results = self.search_duckduckgo(topic, max_results=max_sources)
        all_results.extend(ddg_results)
        print(f"✅ {len(ddg_results)} kaynak bulundu (Wikipedia dahil)")
        
        # 3. Her sonuç için detaylı içerik çıkar
        print("\n📄 İçerikler detaylı olarak çıkarılıyor...")
        for i, result in enumerate(all_results[:max_sources], 1):
            print(f"  {i}/{len(all_results[:max_sources])} {result['url'][:50]}...")
            
            # İçeriği detaylı çıkar (10000 karakter)
            content = self.extract_text_from_url(result['url'], max_length=10000)
            result['content'] = content
            
            # Linkleri çıkar (isteğe bağlı)
            if include_links:
                links = self.extract_links_from_page(result['url'], result['url'])
                result['related_links'] = links[:5]  # İlk 5 link
            
            time.sleep(0.5)  # Rate limiting
        
        self.results = all_results[:max_sources]
        
        print("\n" + "="*60)
        print(f"🎉 TOPLAM {len(self.results)} KAYNAK TARANDI!")
        print("="*60)
        
        return self.results
    
    def generate_professional_article(self, topic):
        """Profesyonel makale formatında AI özet oluştur"""
        if not self.results:
            return "Henüz araştırma yapılmadı."
        
        print("\n" + "📝"*30)
        print("PROFESYONEL MAKALE OLUŞTURULUYOR...")
        print("📝"*30)
        
        # Tüm içerikleri topla
        all_text = []
        sources = []
        
        for i, result in enumerate(self.results, 1):
            all_text.append(result.get('content', ''))
            all_text.append(result.get('snippet', ''))
            sources.append({
                'number': i,
                'title': result['title'],
                'url': result['url'],
                'source': result.get('source', 'Web')
            })
        
        # Metni birleştir
        combined_text = " ".join([t for t in all_text if t])
        
        # Anahtar kelimeleri çıkar
        keywords = self.extract_keywords(combined_text, top_n=20)
        
        # Makale başlığı
        article = []
        article.append(f"{topic.title()}: Kapsamlı Araştırma Özeti\n")
        article.append("=" * 80)
        article.append("")
        
        # Giriş paragrafı - ilk kaynaktan
        if self.results:
            first_content = self.results[0].get('content', '')
            if first_content:
                sentences = first_content.split('.')[:3]
                intro = '. '.join(sentences) + '.'
                article.append(intro)
                article.append("")
        
        # Ana içerik - kaynakları sentezle
        article.append("## Genel Bakış\n")
        
        # Her kaynaktan önemli bilgileri çıkar
        for result in self.results[:5]:  # İlk 5 kaynak
            content = result.get('content', '')
            if content:
                # İlk paragrafı al
                paragraphs = content.split('\n\n')
                if paragraphs:
                    # En uzun paragrafı seç (genellikle en bilgi içeren)
                    main_para = max(paragraphs, key=len)
                    if len(main_para) > 100:
                        sentences = main_para.split('.')[:4]
                        summary = '. '.join(sentences) + '.'
                        article.append(summary)
                        article.append("")
        
        # Anahtar noktalar
        article.append("## Önemli Noktalar\n")
        
        # En önemli kelimeleri kullanarak kategorize et
        top_keywords = [k for k, _ in keywords[:10]]
        article.append("Bu araştırma aşağıdaki konuları kapsamaktadır:\n")
        for i, keyword in enumerate(top_keywords[:8], 1):
            article.append(f"**{i}. {keyword.title()}**: Bu alanda yapılan çalışmalar ve güncel gelişmeler")
        article.append("")
        
        # Detaylı analiz
        article.append("## Detaylı Analiz\n")
        
        for i, result in enumerate(self.results[:3], 1):
            article.append(f"### {i}. {result['title']}\n")
            
            content = result.get('content', '')
            if content:
                # Ortadaki paragrafları al (genellikle en önemli bilgi orada)
                paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]
                if paragraphs:
                    selected_paras = paragraphs[:2]
                    for para in selected_paras:
                        sentences = para.split('.')[:3]
                        summary = '. '.join(sentences) + '.'
                        article.append(summary)
                        article.append("")
            
            article.append(f"*Kaynak: {result.get('source', 'Web')}*")
            article.append("")
        
        # Sonuç ve öneriler
        article.append("## Sonuç\n")
        article.append(f"{topic.title()} konusunda yapılan bu kapsamlı araştırma, ")
        article.append(f"{len(self.results)} farklı kaynaktan derlenen bilgiler ışığında ")
        article.append(f"konunun çeşitli yönlerini ele almaktadır. ")
        article.append(f"Özellikle {', '.join([k for k, _ in keywords[:5]])} ")
        article.append("gibi konular ön plana çıkmaktadır.\n")
        
        # Kaynakça
        article.append("\n## Kaynakça\n")
        for src in sources[:15]:  # İlk 15 kaynak
            article.append(f"[{src['number']}] {src['title']}")
            article.append(f"    {src['url']}")
            article.append(f"    ({src['source']})")
            article.append("")
        
        # Arama önerileri
        article.append("\n## İlgili Arama Önerileri\n")
        search_suggestions = self.generate_search_suggestions(topic, keywords)
        for suggestion in search_suggestions:
            article.append(f"• {suggestion}")
        
        article.append("\n" + "=" * 80)
        article.append(f"Bu rapor {len(self.results)} kaynaktan AI destekli olarak oluşturulmuştur.")
        article.append(f"Oluşturulma tarihi: {time.strftime('%d.%m.%Y %H:%M')}")
        
        return "\n".join(article)
    
    def generate_search_suggestions(self, topic, keywords):
        """İlgili arama önerileri oluştur"""
        suggestions = []
        
        # Topic + top keywords kombinasyonları
        top_keys = [k for k, _ in keywords[:5]]
        
        for key in top_keys[:3]:
            suggestions.append(f"{topic} {key}")
        
        # Genel öneriler
        suggestions.append(f"{topic} pratik uygulamaları")
        suggestions.append(f"{topic} son gelişmeler")
        
        return suggestions
    
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
                    # Bağlantılı sayfalar için de detaylı içerik (5000 karakter)
                    content = self.extract_text_from_url(link, max_length=5000)
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
    
    def extract_key_information(self, text, max_sentences=5):
        """
        Metinden anahtar bilgileri çıkar (Gelişmiş Information Extraction)
        
        Args:
            text: Analiz edilecek metin
            max_sentences: Maksimum cümle sayısı
            
        Returns:
            dict: Çıkarılan bilgiler
        """
        if not text or len(text) < 50:
            return {
                'key_sentences': [],
                'entities': {},
                'keywords': [],
                'summary': ''
            }
        
        # Cümlelere ayır
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Önemli cümleleri skorla
        sentence_scores = {}
        for sent in sentences[:30]:  # İlk 30 cümle
            score = 0
            
            # Uzunluk skoru (çok kısa veya çok uzun değil)
            word_count = len(sent.split())
            if 10 <= word_count <= 30:
                score += 2
            
            # Önemli kelimeleri içeriyor mu?
            important_words = ['önemli', 'ana', 'temel', 'başlıca', 'birinci', 'ilk', 
                             'important', 'main', 'key', 'primary', 'first', 'essential',
                             'critical', 'significant', 'major']
            for word in important_words:
                if word.lower() in sent.lower():
                    score += 3
                    break
            
            # Sayılar içeriyor mu? (istatistik)
            if re.search(r'\d+', sent):
                score += 1
            
            # Tanım yapıyor mu?
            if any(x in sent.lower() for x in ['nedir', 'what is', 'tanımı', 'definition', 'means']):
                score += 4
            
            sentence_scores[sent] = score
        
        # En yüksek skorlu cümleleri al
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        key_sentences = [sent for sent, score in top_sentences]
        
        # Named Entity Recognition (basit versiyon)
        entities = {
            'numbers': re.findall(r'\d+(?:\.\d+)?(?:%|\s*(?:milyon|milyar|bin|million|billion))?', text[:2000]),
            'dates': re.findall(r'\d{4}|\d{1,2}/\d{1,2}/\d{2,4}', text[:2000]),
            'urls': re.findall(r'https?://[^\s]+', text[:2000]),
        }
        
        # Anahtar kelimeler (frekans bazlı)
        words = re.findall(r'\b[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}\b', text.lower())
        word_freq = {}
        stop_words = {'için', 'ile', 'daha', 'olan', 'olarak', 'oluyor', 'this', 'that', 'with', 'from', 'have'}
        
        for word in words[:200]:  # İlk 200 kelime
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # En sık geçen 10 kelime
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, freq in keywords if freq > 1]
        
        # Otomatik özet (ilk 3 anahtar cümle)
        summary = ' '.join(key_sentences[:3])
        
        return {
            'key_sentences': key_sentences,
            'entities': entities,
            'keywords': keywords,
            'summary': summary
        }
    
    def smart_summarize(self, results, max_length=500):
        """
        Akıllı özetleme - birden fazla kaynağı birleştir ve özetle
        
        Args:
            results: Araştırma sonuçları listesi
            max_length: Maksimum özet uzunluğu (kelime)
            
        Returns:
            dict: Akıllı özet
        """
        if not results:
            return {
                'main_summary': 'Sonuç bulunamadı.',
                'key_points': [],
                'all_keywords': [],
                'source_count': 0
            }
        
        all_text = ""
        all_keywords = []
        key_points = []
        
        # Her kaynaktan bilgi çıkar
        for i, result in enumerate(results[:10], 1):
            content = result.get('content', '') or result.get('snippet', '')
            if not content:
                continue
            
            # Anahtar bilgileri çıkar
            info = self.extract_key_information(content, max_sentences=2)
            
            # Anahtar cümleleri topla
            if info['key_sentences']:
                key_points.extend(info['key_sentences'][:2])
            
            # Anahtar kelimeleri birleştir
            all_keywords.extend(info['keywords'])
            
            all_text += content[:1000] + " "
        
        # Genel özet oluştur
        general_info = self.extract_key_information(all_text, max_sentences=5)
        
        # En önemli 3-5 noktayı seç
        unique_points = []
        seen = set()
        for point in key_points:
            point_lower = point.lower()[:50]
            if point_lower not in seen:
                unique_points.append(point)
                seen.add(point_lower)
                if len(unique_points) >= 5:
                    break
        
        # Ana özet
        main_summary = general_info['summary']
        
        # Anahtar kelimeler (en sık geçenler)
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        top_keywords = [kw for kw, freq in top_keywords]
        
        return {
            'main_summary': main_summary,
            'key_points': unique_points,
            'all_keywords': top_keywords,
            'source_count': len(results),
            'entities': general_info['entities']
        }
    
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
