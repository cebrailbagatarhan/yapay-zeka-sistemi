"""
Web Araştırma ve Öz-Öğrenme Modülü
AI sisteminin internetten bilgi toplaması ve öğrenmesi için
"""

import requests
import wikipedia
import json
import time
import re
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import feedparser
import os


class WebResearcher:
    """İnternet araştırması yapan AI modülü"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.knowledge_base = {}
        self.learning_history = []
        
    def search_wikipedia(self, query, max_results=3):
        """Wikipedia'da araştırma yap"""
        try:
            print(f"🔍 Wikipedia'da araştırılıyor: {query}")
            
            # Türkçe Wikipedia'ya geç
            wikipedia.set_lang("tr")
            
            # Arama yap
            search_results = wikipedia.search(query, results=max_results)
            
            articles = []
            for title in search_results[:max_results]:
                try:
                    page = wikipedia.page(title)
                    article = {
                        'title': page.title,
                        'summary': page.summary[:500],
                        'url': page.url,
                        'content': page.content[:2000],
                        'source': 'wikipedia',
                        'timestamp': datetime.now().isoformat()
                    }
                    articles.append(article)
                    time.sleep(0.5)  # Rate limiting
                except wikipedia.exceptions.DisambiguationError as e:
                    # İlk seçeneği al
                    try:
                        page = wikipedia.page(e.options[0])
                        article = {
                            'title': page.title,
                            'summary': page.summary[:500],
                            'url': page.url,
                            'content': page.content[:2000],
                            'source': 'wikipedia',
                            'timestamp': datetime.now().isoformat()
                        }
                        articles.append(article)
                    except:
                        continue
                except:
                    continue
            
            return articles
            
        except Exception as e:
            print(f"❌ Wikipedia araştırma hatası: {e}")
            return []
    
    def search_news(self, query, max_results=5):
        """Haber sitelerinden güncel bilgi topla"""
        try:
            print(f"📰 Haber araştırması: {query}")
            
            # Daha fazla RSS feed kaynağı
            news_feeds = [
                # Türkçe haber kaynakları
                'https://www.bbc.com/turkce/index.xml',
                'https://www.ntv.com.tr/rss',
                'https://www.cnnturk.com/feed/rss/all/news',
                'https://www.hurriyet.com.tr/rss/anasayfa',
                'https://www.sabah.com.tr/rss',
                'https://www.sozcu.com.tr/feed/',
                'https://www.cumhuriyet.com.tr/rss/son_dakika.xml',
                
                # Teknoloji haberleri
                'https://www.webtekno.com/rss.xml',
                'https://www.donanimhaber.com/rss',
                'https://www.chip.com.tr/rss/genel.xml',
                
                # Uluslararası kaynaklar
                'https://feeds.reuters.com/reuters/technologyNews',
                'https://feeds.feedburner.com/TechCrunch',
                'https://www.theverge.com/rss/index.xml',
            ]
            
            news_articles = []
            
            for feed_url in news_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:max_results]:
                        if query.lower() in entry.title.lower() or query.lower() in entry.get('summary', '').lower():
                            article = {
                                'title': entry.title,
                                'summary': entry.get('summary', '')[:300],
                                'url': entry.link,
                                'published': entry.get('published', ''),
                                'source': 'news',
                                'feed_source': feed_url,
                                'timestamp': datetime.now().isoformat()
                            }
                            news_articles.append(article)
                    
                    time.sleep(0.5)
                except:
                    continue
            
            return news_articles[:max_results]
            
        except Exception as e:
            print(f"❌ Haber araştırma hatası: {e}")
            return []
    
    def search_web_content(self, query, max_results=3):
        """Genel web içeriği araştır"""
        try:
            print(f"🌐 Web araştırması: {query}")
            
            # Basit web araması (DuckDuckGo API kullanarak)
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            
            response = self.session.get(search_url)
            data = response.json()
            
            results = []
            
            # Abstract sonucu
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', 'Web Result'),
                    'summary': data['Abstract'][:500],
                    'url': data.get('AbstractURL', ''),
                    'source': 'web_search',
                    'timestamp': datetime.now().isoformat()
                })
            
            # RelatedTopics sonuçları
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'summary': topic.get('Text', '')[:500],
                        'url': topic.get('FirstURL', ''),
                        'source': 'web_search',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Web araştırma hatası: {e}")
            return []
    
    def search_academic_sources(self, query, max_results=3):
        """Akademik ve eğitim kaynaklarından araştırma"""
        try:
            print(f"🎓 Akademik kaynak araştırması: {query}")
            
            academic_results = []
            
            # Özel eğitim siteleri ve akademik kaynaklar
            academic_sites = [
                'https://scholar.google.com',
                'https://www.researchgate.net',
                'https://arxiv.org',
                'https://www.coursera.org',
                'https://www.khanacademy.org',
                'https://www.edx.org',
                'https://stackoverflow.com',
                'https://github.com',
                'https://medium.com',
                'https://towardsdatascience.com'
            ]
            
            # GitHub repository arama (public API)
            try:
                github_url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc"
                response = self.session.get(github_url)
                if response.status_code == 200:
                    github_data = response.json()
                    
                    for repo in github_data.get('items', [])[:2]:
                        academic_results.append({
                            'title': f"GitHub: {repo['name']}",
                            'summary': repo.get('description', 'GitHub repository')[:300],
                            'url': repo['html_url'],
                            'source': 'github',
                            'stars': repo.get('stargazers_count', 0),
                            'timestamp': datetime.now().isoformat()
                        })
                time.sleep(1)
            except:
                pass
            
            # Stack Overflow arama
            try:
                so_url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes&q={query}&site=stackoverflow"
                response = self.session.get(so_url)
                if response.status_code == 200:
                    so_data = response.json()
                    
                    for question in so_data.get('items', [])[:2]:
                        academic_results.append({
                            'title': f"StackOverflow: {question['title']}",
                            'summary': f"Score: {question.get('score', 0)}, Answers: {question.get('answer_count', 0)}",
                            'url': question['link'],
                            'source': 'stackoverflow',
                            'score': question.get('score', 0),
                            'timestamp': datetime.now().isoformat()
                        })
                time.sleep(1)
            except:
                pass
            
            return academic_results[:max_results]
            
        except Exception as e:
            print(f"❌ Akademik araştırma hatası: {e}")
            return []
    
    def search_specialized_sites(self, query, max_results=3):
        """Özel konulara göre site araştırması"""
        try:
            print(f"🔬 Özel site araştırması: {query}")
            
            specialized_results = []
            query_lower = query.lower()
            
            # Konuya göre özel siteler
            if any(tech in query_lower for tech in ['python', 'programming', 'kod', 'yazılım']):
                # Programlama siteleri
                programming_sites = [
                    'https://docs.python.org',
                    'https://realpython.com',
                    'https://www.w3schools.com',
                    'https://developer.mozilla.org',
                    'https://www.geeksforgeeks.org'
                ]
                
                specialized_results.append({
                    'title': 'Python Official Documentation',
                    'summary': 'Python resmi dokümantasyonu - tüm Python özelliklerinin detaylı açıklaması.',
                    'url': 'https://docs.python.org',
                    'source': 'specialized_programming',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif any(ai in query_lower for ai in ['ai', 'machine learning', 'yapay zeka', 'deep learning']):
                # AI/ML siteleri
                specialized_results.append({
                    'title': 'Papers With Code',
                    'summary': 'Machine Learning araştırma makaleleri ve kodları - en güncel AI araştırmaları.',
                    'url': 'https://paperswithcode.com',
                    'source': 'specialized_ai',
                    'timestamp': datetime.now().isoformat()
                })
                
                specialized_results.append({
                    'title': 'Towards Data Science',
                    'summary': 'Data Science ve Machine Learning konularında kaliteli yazılar ve tutorials.',
                    'url': 'https://towardsdatascience.com',
                    'source': 'specialized_ai',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif any(math in query_lower for math in ['matematik', 'mathematics', 'math']):
                # Matematik siteleri
                specialized_results.append({
                    'title': 'Wolfram MathWorld',
                    'summary': 'Kapsamlı matematik ansiklopedisi - tüm matematik konularında detaylı açıklamalar.',
                    'url': 'https://mathworld.wolfram.com',
                    'source': 'specialized_math',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif any(sci in query_lower for sci in ['bilim', 'science', 'fizik', 'kimya']):
                # Bilim siteleri
                specialized_results.append({
                    'title': 'Scientific American',
                    'summary': 'Bilimsel araştırmalar ve keşifler hakkında güncel haberler.',
                    'url': 'https://www.scientificamerican.com',
                    'source': 'specialized_science',
                    'timestamp': datetime.now().isoformat()
                })
            
            return specialized_results[:max_results]
            
        except Exception as e:
            print(f"❌ Özel site araştırma hatası: {e}")
            return []
    
    def scrape_website(self, url):
        """Belirli bir web sitesini analiz et"""
        try:
            print(f"🕷️ Web sitesi analizi: {url}")
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Metin içeriği çıkar
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'content': text[:2000],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Web sitesi analiz hatası: {e}")
            return None
    
    def comprehensive_research(self, query):
        """Kapsamlı araştırma - Tüm kaynakları birleştir"""
        print(f"\n🎯 KAPSAMLI ARAŞTIRMA: {query}")
        print("=" * 50)
        
        all_results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Wikipedia araştırması
        wiki_results = self.search_wikipedia(query)
        all_results['sources']['wikipedia'] = wiki_results
        print(f"📚 Wikipedia: {len(wiki_results)} sonuç")
        
        # Haber araştırması
        news_results = self.search_news(query)
        all_results['sources']['news'] = news_results
        print(f"📰 Haberler: {len(news_results)} sonuç")
        
        # Genel web araştırması
        web_results = self.search_web_content(query)
        all_results['sources']['web'] = web_results
        print(f"🌐 Web: {len(web_results)} sonuç")
        
        # Akademik kaynak araştırması
        academic_results = self.search_academic_sources(query)
        all_results['sources']['academic'] = academic_results
        print(f"🎓 Akademik: {len(academic_results)} sonuç")
        
        # Özel site araştırması
        specialized_results = self.search_specialized_sites(query)
        all_results['sources']['specialized'] = specialized_results
        print(f"🔬 Özel Siteler: {len(specialized_results)} sonuç")
        
        # Bilgi tabanına ekle
        self.knowledge_base[query] = all_results
        self.learning_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_sources': (len(wiki_results) + len(news_results) + len(web_results) + 
                            len(academic_results) + len(specialized_results))
        })
        
        return all_results
    
    def save_knowledge_base(self, filepath="data/knowledge_base.json"):
        """Öğrenilen bilgileri kaydet"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            print(f"💾 Bilgi tabanı kaydedildi: {filepath}")
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
    
    def load_knowledge_base(self, filepath="data/knowledge_base.json"):
        """Önceden öğrenilen bilgileri yükle"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print(f"📂 Bilgi tabanı yüklendi: {len(self.knowledge_base)} konu")
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")


class AutoLearner:
    """Otomatik öğrenme sistemi"""
    
    def __init__(self, web_researcher):
        self.researcher = web_researcher
        self.auto_topics = [
            "yapay zeka son gelişmeler",
            "machine learning yenilikler",
            "deep learning breakthrough",
            "python programlama güncel",
            "teknoloji haberleri",
            "bilim son dakika",
            "AI research papers"
        ]
        self.learning_schedule = {}
    
    def continuous_learning(self, interval_minutes=60):
        """Sürekli öğrenme modu"""
        print(f"🔄 Sürekli öğrenme başlatılıyor (her {interval_minutes} dakika)")
        
        import threading
        import time
        
        def learning_loop():
            while True:
                try:
                    # Rastgele bir konu seç
                    import random
                    topic = random.choice(self.auto_topics)
                    
                    print(f"\n🤖 Otomatik öğrenme: {topic}")
                    results = self.researcher.comprehensive_research(topic)
                    
                    # Yeni bilgileri analiz et
                    self.analyze_new_knowledge(results)
                    
                    # Bilgi tabanını kaydet
                    self.researcher.save_knowledge_base()
                    
                    # Bekleme
                    time.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    print(f"❌ Otomatik öğrenme hatası: {e}")
                    time.sleep(300)  # 5 dakika bekle
        
        # Background thread olarak başlat
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        
        return learning_thread
    
    def analyze_new_knowledge(self, research_results):
        """Yeni öğrenilen bilgileri analiz et"""
        try:
            total_sources = 0
            important_keywords = []
            
            for source_type, articles in research_results['sources'].items():
                total_sources += len(articles)
                
                for article in articles:
                    # Önemli kelimeleri çıkar
                    text = article.get('summary', '') + ' ' + article.get('content', '')
                    words = re.findall(r'\b\w+\b', text.lower())
                    
                    # Teknik terimler
                    tech_terms = ['ai', 'machine learning', 'deep learning', 'neural', 
                                 'algorithm', 'python', 'tensorflow', 'pytorch']
                    
                    for term in tech_terms:
                        if term in text.lower():
                            important_keywords.append(term)
            
            # Öğrenme özeti
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'total_sources': total_sources,
                'important_keywords': list(set(important_keywords)),
                'learning_value': total_sources * len(set(important_keywords))
            }
            
            print(f"📊 Öğrenme Analizi:")
            print(f"   📚 Toplam kaynak: {total_sources}")
            print(f"   🔑 Önemli kelimeler: {len(set(important_keywords))}")
            print(f"   💎 Öğrenme değeri: {analysis['learning_value']}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Analiz hatası: {e}")
            return {}


class SmartChatbot:
    """Araştırma yapabilen akıllı chatbot"""
    
    def __init__(self, web_researcher, auto_learner):
        self.researcher = web_researcher
        self.auto_learner = auto_learner
        self.conversation_context = []
    
    def smart_response(self, user_query):
        """Araştırma tabanlı akıllı yanıt"""
        
        # Matematik sorularını kontrol et
        math_response = self.handle_math_query(user_query)
        if math_response:
            return math_response
        
        # Önce mevcut bilgi tabanından kontrol et
        if user_query in self.researcher.knowledge_base:
            print("💡 Bilgi tabanından yanıt bulundu!")
            knowledge = self.researcher.knowledge_base[user_query]
            return self.format_knowledge_response(knowledge)
        
        # Araştırma anahtar kelimelerini belirle
        research_keywords = self.extract_research_keywords(user_query)
        
        if research_keywords:
            print(f"🔍 Araştırma yapılıyor: {research_keywords}")
            
            # Canlı araştırma yap
            research_results = self.researcher.comprehensive_research(research_keywords)
            
            # Sonuçları formatla
            response = self.format_research_response(research_results)
            
            return response
        else:
            # Normal chatbot yanıtı
            return self.generate_normal_response(user_query)
    
    def handle_math_query(self, query):
        """Matematik sorularını işle"""
        import re
        
        # Üs alma (^, **, power gibi)
        power_patterns = [
            r'(\d+)\s*[\^]\s*(\d+)',  # 3^5
            r'(\d+)\s*\*\*\s*(\d+)',  # 3**5
            r'(\d+)\s+üssü\s+(\d+)',  # 3 üssü 5
            r'(\d+)\s+power\s+(\d+)'   # 3 power 5
        ]
        
        for pattern in power_patterns:
            match = re.search(pattern, query)
            if match:
                base = int(match.group(1))
                exp = int(match.group(2))
                result = base ** exp
                
                return f"""🔢 **Üs Alma Hesabı:**
**{base}^{exp} = {result}**

💡 **Adım adım çözüm:**
1. Taban sayı: {base}
2. Üs: {exp}  
3. İşlem: {base} × {base} {"× " + str(base) if exp > 2 else ""} ({exp} kez çarpım)
4. Sonuç: **{result}**

📚 **Üs alma kuralları:**
- a^0 = 1 (sıfırıncı kuvvet)
- a^1 = a (birinci kuvvet)
- a^n = a × a × ... × a (n kez çarpım)"""
        
        # Faktöriyel (! işareti)
        factorial_pattern = r'(\d+)!'
        match = re.search(factorial_pattern, query)
        if match:
            n = int(match.group(1))
            if n <= 20:  # Güvenlik için limit
                import math
                result = math.factorial(n)
                
                calculation = " × ".join(str(i) for i in range(1, n+1))
                
                return f"""🔢 **Faktöriyel Hesabı:**
**{n}! = {result}**

💡 **Adım adım çözüm:**
1. {n}! = {calculation}
2. Sonuç: **{result}**

📚 **Faktöriyel nedir?**
n! = 1 × 2 × 3 × ... × n"""
            else:
                return f"⚠️ {n}! çok büyük bir sayı! 20'den küçük sayılar için hesaplama yapabilirim."
        
        # Basit aritmetik
        basic_patterns = [
            (r'(\d+)\s*\+\s*(\d+)', '+', lambda a, b: a + b),
            (r'(\d+)\s*-\s*(\d+)', '-', lambda a, b: a - b),
            (r'(\d+)\s*\*\s*(\d+)', '×', lambda a, b: a * b),
            (r'(\d+)\s*/\s*(\d+)', '÷', lambda a, b: a / b if b != 0 else None)
        ]
        
        for pattern, op, func in basic_patterns:
            match = re.search(pattern, query)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                
                if op == '÷' and b == 0:
                    return "⚠️ Sıfıra bölme hatası! Bir sayı sıfıra bölünemez."
                
                result = func(a, b)
                
                return f"""🔢 **Aritmetik İşlem:**
**{a} {op} {b} = {result}**

💡 Bu hesaplama web araştırması gerektirmez, doğrudan hesapladım!"""
        
        # Karekök (sqrt)
        sqrt_pattern = r'(\d+)\s*(karekök|sqrt|√)'
        match = re.search(sqrt_pattern, query)
        if match:
            n = int(match.group(1))
            import math
            result = math.sqrt(n)
            
            return f"""🔢 **Karekök Hesabı:**
**√{n} = {result:.4f}**

💡 **Açıklama:**
√{n} = {result:.4f} (yaklaşık)"""
        
        return None
    
    def extract_research_keywords(self, query):
        """Araştırma gerektiren anahtar kelimeleri çıkar"""
        research_triggers = [
            'nedir', 'nasıl', 'ne demek', 'açıkla', 'anlat',
            'son gelişmeler', 'güncel', 'yeni', 'trend',
            'araştır', 'öğren', 'bilgi ver', 'detay'
        ]
        
        for trigger in research_triggers:
            if trigger in query.lower():
                # Araştırma konusunu çıkar
                cleaned_query = query.lower()
                for trigger in research_triggers:
                    cleaned_query = cleaned_query.replace(trigger, '').strip()
                
                return cleaned_query if len(cleaned_query) > 2 else query
        
        return None
    
    def format_research_response(self, research_results):
        """Araştırma sonuçlarını formatla"""
        response = f"🔍 **'{research_results['query']}' hakkında araştırma sonuçları:**\n\n"
        
        # Wikipedia sonuçları
        wiki_results = research_results['sources'].get('wikipedia', [])
        if wiki_results:
            response += "📚 **Wikipedia'dan:**\n"
            for article in wiki_results[:2]:
                response += f"• {article['title']}: {article['summary'][:200]}...\n"
            response += "\n"
        
        # Haber sonuçları
        news_results = research_results['sources'].get('news', [])
        if news_results:
            response += "📰 **Güncel Haberler:**\n"
            for article in news_results[:2]:
                response += f"• {article['title']}: {article['summary'][:200]}...\n"
            response += "\n"
        
        # Akademik sonuçlar
        academic_results = research_results['sources'].get('academic', [])
        if academic_results:
            response += "🎓 **Akademik Kaynaklar:**\n"
            for article in academic_results[:2]:
                extra_info = ""
                if article['source'] == 'github':
                    extra_info = f" (⭐ {article.get('stars', 0)} stars)"
                elif article['source'] == 'stackoverflow':
                    extra_info = f" (👍 {article.get('score', 0)} score)"
                response += f"• {article['title']}{extra_info}: {article['summary'][:200]}...\n"
            response += "\n"
        
        # Özel site sonuçları
        specialized_results = research_results['sources'].get('specialized', [])
        if specialized_results:
            response += "🔬 **Özel Kaynaklar:**\n"
            for article in specialized_results[:2]:
                response += f"• {article['title']}: {article['summary'][:200]}...\n"
            response += "\n"
        
        # Web sonuçları
        web_results = research_results['sources'].get('web', [])
        if web_results:
            response += "🌐 **Web'den:**\n"
            for article in web_results[:2]:
                response += f"• {article['summary'][:200]}...\n"
            response += "\n"
        
        total_sources = sum(len(articles) for articles in research_results['sources'].values())
        response += f"💡 Bu bilgiler {total_sources} farklı kaynaktan canlı internet araştırmasıyla toplandı ve bilgi tabanıma eklendi!"
        
        return response
    
    def format_knowledge_response(self, knowledge):
        """Bilgi tabanından yanıt formatla"""
        response = f"💾 **Bilgi tabanımdan '{knowledge['query']}' hakkında:**\n\n"
        
        total_sources = 0
        for source_type, articles in knowledge['sources'].items():
            if articles:
                total_sources += len(articles)
                response += f"📌 **{source_type.title()}:** {articles[0]['summary'][:300]}...\n\n"
        
        response += f"ℹ️ Bu bilgiler {knowledge['timestamp'][:10]} tarihinde toplanmış ({total_sources} kaynak)"
        
        return response
    
    def generate_normal_response(self, query):
        """Normal chatbot yanıtı"""
        return f"🤖 '{query}' hakkında daha detaylı bilgi almak için araştırma yapmamı ister misiniz? 'araştır' veya 'öğren' yazabilirsiniz."
