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
from datasets import load_dataset

@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    """
    Hugging Face datasets üzerinden yemek tariflerini yükler.
    İlk 200 tarifi alır ve Title + Ingredients + Instructions şeklinde birleştirir.
    """
    ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
    recipes = []

    for item in ds:
        title = item.get("Title", "")
        ingredients = item.get("Ingredients", "")
        instructions = item.get("Instructions", "")
        full_recipe = f"{title}\nMalzemeler: {ingredients}\nYapılışı: {instructions}"
        recipes.append(full_recipe)
    
    return recipes

# ------------------------------------------------
# Veri ve Embedding Cache
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler yükleniyor...")
def load_data_and_embeddings():
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]
    embeds = []

    try:
        # Her bir metin için embed oluştur
        for doc in recipe_docs:
            res = client.models.embed_content(
                model=embedding_model,
                text=doc  # artık 'text' parametresi kullanılıyor
            )
            embeds.append(res.embedding.values)
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
    # Sorgu için embedding
    q_res = client.models.embed_content(
        model=embedding_model,
        text=query   # 'text' parametresi tekil metin için kullanılıyor
    )
    q_embed = q_res.embedding.values


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
