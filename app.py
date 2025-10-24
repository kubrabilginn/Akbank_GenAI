import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List

# Google SDK (Sadece LLM için)
from google import genai
# Sentence Transformers (Embedding için)
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# API Key kontrolü (Sadece LLM için gerekli)
# ------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ Google API Anahtarı bulunamadı (LLM için gerekli). Streamlit Secrets bölümüne GEMINI_API_KEY ekleyin.")
    st.stop()

# Google AI'ı sadece LLM için yapılandır
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Google AI SDK yapılandırılırken hata (LLM için): {e}")
    st.stop()

# Model isimleri
# 🛑 DÜZELTME: Embedding modeli artık Sentence Transformer
embedding_model_name = 'all-MiniLM-L6-v2' # Popüler ve hızlı bir model
llm_model_name = "gemini-1.5-flash-latest"

# ------------------------------------------------
# Sentence Transformer Modelini Yükleme (Cache ile)
# ------------------------------------------------
@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embedding_model():
    """Hugging Face'den Sentence Transformer modelini yükler."""
    return SentenceTransformer(embedding_model_name)

# ------------------------------------------------
# Tarifleri yükleme (Değişiklik yok)
# ------------------------------------------------
from datasets import load_dataset

@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    # ... (Fonksiyon içeriği aynı kalır) ...
    ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:50]") # 🛑 Veri miktarını azalttık (Limitleri zorlamamak için)
    recipes = []
    for item in ds:
        title = item.get("Title", "")
        ingredients = item.get("Ingredients", [])
        if isinstance(ingredients, list):
            ingredients = ", ".join(ingredients)
        instructions = item.get("Instructions", "")
        full_recipe = f"TARİF ADI: {title}\nMAL ингредиенттер: {ingredients}\nADIMLAR: {instructions}" # Malzemeler etiketini düzelttim
        recipes.append(full_recipe)
    return recipes

# ------------------------------------------------
# Veri ve Embedding Cache (Sentence Transformer ile)
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler hazırlanıyor...")
def load_data_and_embeddings(_embedding_model): # Model artık argüman olarak geliyor
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]

    try:
        # 🛑 DÜZELTME: Sentence Transformer ile embedding oluşturma
        embeds = _embedding_model.encode(recipe_docs, show_progress_bar=False).tolist() # NumPy array'i listeye çevir
    except Exception as e:
        st.error(f"Embedding oluşturulurken hata: {str(e)}")
        st.stop()

    return recipe_docs, doc_ids, np.array(embeds) # Embeddings'i NumPy array olarak döndür


# ✅ Kosinüs benzerliği (Değişiklik yok)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (HF Embedding + Gemini LLM)")
st.divider()

# Embedding modelini yükle
embedding_model = load_embedding_model()
# Veri ve embeddingleri yükle
docs, ids, embeddings = load_data_and_embeddings(embedding_model)

st.caption(f"Veri tabanımızda {len(docs)} tarif bulunmaktadır. (LLM: {llm_model_name} | Embedding: {embedding_model_name})")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ne pişirmek istersin? (örn: Ispanaklı bir şey)")

if query:
    st.session_state.history.append({"role": "user", "content": query})

    with st.spinner("Tarif aranıyor ve yanıt oluşturuluyor..."):
        try:
            # 1. Sorgu embed (Sentence Transformer ile)
            q_embed = np.array(embedding_model.encode(query))

            # 2. Cosine similarity hesapla (Değişiklik yok)
            sims = [(i, cosine_similarity(q_embed, emb)) for i, emb in enumerate(embeddings)]
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]

            # 3. En iyi dokümanları al (Değişiklik yok)
            top_docs_content = [docs[i] for i, _ in sims]
            source_names = [doc.split('\n')[0].replace('TARİF ADI: ', '') for doc in top_docs_content]
            context = "\n---\n".join(top_docs_content)

            # 4. Prompt oluşturma (Değişiklik yok)
            prompt = f"""Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver.
Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla.

BAĞLAM:
{context}

SORU: {query}
YANIT:"""

            # 5. LLM'ye gönderme (Gemini kullanmaya devam ediyoruz)
            llm = genai.GenerativeModel(model_name=llm_model_name)
            safety_settings = [ # Güvenlik ayarları
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = llm.generate_content(prompt, safety_settings=safety_settings)

            try:
                llm_response = response.text
            except ValueError: # Yanıt engellenirse
                llm_response = "Modelden yanıt alınamadı veya yanıt güvenlik nedeniyle engellendi."
                # print(response.prompt_feedback) # Geri bildirimi görmek için

            # Geçmişe ekle
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        except Exception as e:
            st.error(f"RAG/API Hatası: {str(e)}")
            st.session_state.history.append({"role": "assistant", "content": f"Üzgünüm, bir hata oluştu: {e}", "sources": []})


# Geçmişi gösterme (Değişiklik yok)
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            for name in set(msg.get("sources", [])):
                st.markdown(f"**-** *{name}*")
