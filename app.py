import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List

from google import genai
from google.genai import types

# ------------------------------------------------
# API Key kontrolü
# ------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ API Anahtarı bulunamadı. Streamlit Secrets bölümüne GEMINI_API_KEY ekleyin.")
    st.stop()

client = genai.Client(api_key=API_KEY)
embedding_model = "models/text-embedding-004"
llm_model = "gemini-2.5-flash"

# ------------------------------------------------
# Tarifleri yükleme
# ------------------------------------------------
@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> List[str]:
    """
    Tarif verilerini yükler ve liste olarak döner.
    Bu örnekte CSV’den yükleme yapıyoruz, dilersen JSON veya başka bir kaynaktan da olabilir.
    """
    data_path = "recipes.csv"  # CSV dosyanın yolu
    if not os.path.exists(data_path):
        st.error(f"❌ Tarif dosyası bulunamadı: {data_path}")
        st.stop()
    
    df = pd.read_csv(data_path)
    
    # Tarifi içeren sütun adı 'recipe_text' olduğunu varsayıyoruz
    if 'recipe_text' not in df.columns:
        st.error("❌ CSV içinde 'recipe_text' sütunu bulunamadı.")
        st.stop()
    
    return df['recipe_text'].tolist()

# ------------------------------------------------
# Veri ve Embedding Cache
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler yükleniyor...")
def load_data_and_embeddings():
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]

    embed_request = {
        "model": embedding_model,
        "texts": recipe_docs
    }

    try:
        res = client.models.embed_content(**embed_request)
        embeds = res.embeddings
    except Exception as e:
        st.error(f"Embedding oluşturulurken hata: {str(e)}")
        raise e

    return recipe_docs, doc_ids, embeds

# ✅ Kosinüs benzerliği
def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Yemek Tarifleri Chatbotu (Chroma’sız tam uyumlu)")
st.divider()

docs, ids, embeddings = load_data_and_embeddings()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ne pişirmek istersin? (örn: Ispanaklı bir şey)")

if query:
    st.session_state.history.append({"role": "user", "content": query})

    with st.spinner("Tarif aranıyor..."):
        q_embed = client.models.embed_content(
            model=embedding_model, input=query
        ).embedding.values

        sims = [(i, cosine_similarity(q_embed, emb)) for i, emb in enumerate(embeddings)]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]

        top_docs = [docs[i] for i, _ in sims]

        if not top_docs:
            answer = "Üzgünüm, uygun tarif bulamadım."
        else:
            prompt = f"""
Aşağıda yemek tarifleri var. Kullanıcının sorusuna yardımcı ol:

BAĞLAM:
{"\n---\n".join(top_docs)}

SORU: {query}
YANIT:
"""
            answer = client.models.generate_content(
                model=llm_model,
                contents=prompt
            ).text

        st.session_state.history.append({"role": "assistant", "content": answer})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
