"""
🌐 YAPAY ZEKA SİSTEMİ - WEB ARAYÜZÜ
Streamlit ile modern web arayüzü
"""

import streamlit as st
import sys
import os
from io import StringIO
import time

# Src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Model ve araştırma sistemleri
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from src.deep_web_researcher import DeepWebResearcher
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Sayfa yapılandırması
st.set_page_config(
    page_title="Yapay Zeka Sistemi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile görsel iyileştirmeler
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        padding: 20px;
        background-color: #1e3a1e;
        border-left: 5px solid #4CAF50;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        padding: 20px;
        background-color: #1e2a3a;
        border-left: 5px solid #2196F3;
        border-radius: 5px;
        margin: 10px 0;
    }
    .source-card {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Session state başlatma
if 'qwen_model' not in st.session_state:
    st.session_state.qwen_model = None
if 'qwen_tokenizer' not in st.session_state:
    st.session_state.qwen_tokenizer = None
if 'researcher' not in st.session_state:
    st.session_state.researcher = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'research_results' not in st.session_state:
    st.session_state.research_results = None

# Başlık
st.title("🤖 Yapay Zeka Sistemi")
st.markdown("### Qwen Model + Derin Web Araştırması")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Model yükleme butonu
    if st.session_state.qwen_model is None:
        if st.button("🚀 Qwen Modelini Yükle"):
            with st.spinner("Model yükleniyor..."):
                try:
                    model_path = "./qwen-model"
                    st.session_state.qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
                    st.session_state.qwen_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map=None
                    ).to('cpu')
                    st.success("✅ Model yüklendi!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Model yükleme hatası: {e}")
    else:
        st.success("✅ Qwen Modeli Hazır")
        
        # Model bilgileri
        st.info(f"""
        **Model:** Qwen2.5-1.5B-Instruct  
        **Parametreler:** 1.54B  
        **Cihaz:** CPU
        """)
    
    st.divider()
    
    # Araştırma ayarları
    st.header("🔍 Araştırma Ayarları")
    max_sources = st.slider("Maksimum Kaynak Sayısı", 5, 30, 15)
    max_tokens = st.slider("Maksimum Token (Yanıt)", 256, 2048, 700)
    
    st.divider()
    
    # İstatistikler
    st.header("📊 İstatistikler")
    st.metric("Sohbet Mesajı", len(st.session_state.chat_history))
    if st.session_state.research_results:
        st.metric("Son Araştırma Kaynağı", len(st.session_state.research_results))

# Ana içerik - Tabs
tab1, tab2, tab3 = st.tabs(["💬 Sohbet", "🔍 Derin Araştırma", "📚 Geçmiş"])

# TAB 1: Sohbet
with tab1:
    st.header("💬 Qwen ile Sohbet")
    
    if st.session_state.qwen_model is None:
        st.warning("⚠️ Lütfen önce Qwen modelini yükleyin! (Sol menüden)")
    else:
        # Sohbet geçmişi
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"**👤 Siz:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Qwen:** {msg['content']}")
                st.divider()
        
        # Soru girişi
        with st.form(key='chat_form', clear_on_submit=True):
            user_question = st.text_input("Sorunuzu yazın:", placeholder="Örn: Python nedir?")
            submit_button = st.form_submit_button("📤 Gönder")
            
            if submit_button and user_question:
                # Soruyu ekle
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Qwen'den yanıt al
                with st.spinner("Qwen düşünüyor..."):
                    try:
                        prompt = f"Soru: {user_question}\n\nYanıt:"
                        inputs = st.session_state.qwen_tokenizer(prompt, return_tensors="pt").to('cpu')
                        
                        outputs = st.session_state.qwen_model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=st.session_state.qwen_tokenizer.eos_token_id
                        )
                        
                        response = st.session_state.qwen_tokenizer.decode(
                            outputs[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True
                        )
                        
                        # Yanıtı ekle
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Hata: {e}")

# TAB 2: Derin Araştırma
with tab2:
    st.header("🔍 Derin Web Araştırması")
    
    if st.session_state.qwen_model is None:
        st.warning("⚠️ Lütfen önce Qwen modelini yükleyin! (Sol menüden)")
    else:
        # Araştırma formu
        with st.form(key='research_form'):
            research_topic = st.text_input(
                "Araştırma Konusu:", 
                placeholder="Örn: Yapay Zeka Etiği"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                research_type = st.selectbox(
                    "Araştırma Tipi:",
                    ["Standart (15 kaynak)", "Hızlı (10 kaynak)", "Kapsamlı (20 kaynak)"]
                )
            with col2:
                include_analysis = st.checkbox("Qwen Analizi Ekle", value=True)
            
            research_button = st.form_submit_button("🔍 Araştırmayı Başlat")
            
            if research_button and research_topic:
                # Kaynak sayısını belirle
                sources_map = {
                    "Hızlı (10 kaynak)": 10,
                    "Standart (15 kaynak)": 15,
                    "Kapsamlı (20 kaynak)": 20
                }
                sources = sources_map[research_type]
                
                # Researcher başlat
                if st.session_state.researcher is None:
                    st.session_state.researcher = DeepWebResearcher()
                
                # Araştırmayı yap
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🌐 Web kaynakları taranıyor...")
                progress_bar.progress(30)
                
                try:
                    results = st.session_state.researcher.deep_research(
                        research_topic, 
                        max_sources=sources,
                        include_links=False
                    )
                    
                    progress_bar.progress(60)
                    status_text.text("📄 İçerikler analiz ediliyor...")
                    
                    st.session_state.research_results = results
                    
                    # Qwen analizi
                    if include_analysis and results:
                        status_text.text("🤖 Qwen analiz yapıyor...")
                        progress_bar.progress(80)
                        
                        # Tüm içerikleri birleştir
                        combined_info = "\n\n".join([
                            f"Kaynak {i+1} ({r['source']}): {r['title']}\n{r['content'][:3000]}"
                            for i, r in enumerate(results) if r.get('content')
                        ])
                        
                        # Qwen'e sor
                        prompt = f"""Aşağıdaki web kaynaklarından toplanan bilgilere dayanarak '{research_topic}' hakkında kapsamlı bir özet yaz:

{combined_info[:8000]}

Özet:"""
                        
                        inputs = st.session_state.qwen_tokenizer(prompt, return_tensors="pt").to('cpu')
                        outputs = st.session_state.qwen_model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=st.session_state.qwen_tokenizer.eos_token_id
                        )
                        
                        analysis = st.session_state.qwen_tokenizer.decode(
                            outputs[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True
                        )
                        
                        st.session_state.qwen_analysis = analysis
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Araştırma tamamlandı!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Araştırma hatası: {e}")
        
        # Sonuçları göster
        if st.session_state.research_results:
            st.divider()
            st.subheader(f"📊 Sonuçlar ({len(st.session_state.research_results)} kaynak)")
            
            # Qwen analizi varsa göster
            if hasattr(st.session_state, 'qwen_analysis'):
                st.markdown("### 🤖 Qwen Analizi")
                st.markdown(f'<div class="success-box">{st.session_state.qwen_analysis}</div>', 
                           unsafe_allow_html=True)
                st.divider()
            
            # Kaynakları göster
            st.markdown("### 📚 Bulunan Kaynaklar")
            for i, result in enumerate(st.session_state.research_results, 1):
                with st.expander(f"📄 {i}. {result['title'][:100]}"):
                    st.markdown(f"**Kaynak:** {result['source']}")
                    st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                    st.markdown(f"**Özet:** {result['snippet'][:300]}...")
                    if result.get('content'):
                        st.markdown("---")
                        st.markdown("**İçerik:**")
                        st.text_area(
                            label="",
                            value=result['content'][:2000] + "...",
                            height=200,
                            key=f"content_{i}"
                        )

# TAB 3: Geçmiş
with tab3:
    st.header("📚 Sohbet Geçmişi")
    
    if not st.session_state.chat_history:
        st.info("Henüz sohbet geçmişi yok.")
    else:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg['role'] == 'user':
                st.markdown(f"**👤 {i+1}. Mesaj (Siz):**")
                st.text(msg['content'])
            else:
                st.markdown(f"**🤖 {i+1}. Mesaj (Qwen):**")
                st.text(msg['content'])
            st.divider()
        
        # Geçmişi temizle butonu
        if st.button("🗑️ Geçmişi Temizle"):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    🤖 Yapay Zeka Sistemi v1.0 | Qwen2.5-1.5B + Derin Web Araştırması
</div>
""", unsafe_allow_html=True)
