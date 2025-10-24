import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List

# Google SDK ve diğer yardımcılar
from google import genai
from google.genai import types

# ------------------------------------------------
# API Key kontrolü (Secrets'tan okunur)
# ------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ API Anahtarı bulunamadı. Streamlit Secrets bölümüne GEMINI_API_KEY ekleyin.")
    st.stop()

# 🛑 genai.configure() ÇAĞRISINI TAMAMEN KALDIRDIK. 🛑
# Kütüphanenin API_KEY'i ortam değişkeninden otomatik almasını bekliyoruz.

embedding_model_name = "models/embedding-001"
llm_model_name = "gemini-2.5-flash"

# ------------------------------------------------
# Tarifleri yükleme (Değişiklik yok)
# ------------------------------------------------
from datasets import load_dataset

@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
    recipes = []
    for item in ds:
        title = item.get("Title", "")
        ingredients = item.get("Ingredients", [])
        if isinstance(ingredients, list):
            ingredients = ", ".join(ingredients)
        instructions = item.get("Instructions", "")
        full_recipe = f"TARİF ADI: {title}\nMALZEMELER: {ingredients}\nADIMLAR: {instructions}"
        recipes.append(full_recipe)
    return recipes

# ------------------------------------------------
# Veri ve Embedding Cache (configure olmadan)
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler hazırlanıyor...")
def load_data_and_embeddings():
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]

    try:
        # API anahtarı ortam değişkeninden otomatik alınmalı
        result = genai.embed_content(
            model=embedding_model_name,
            content=recipe_docs,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeds = result['embedding']
    except Exception as e:
        # Hata mesajına API anahtarı kontrolü ekleyelim
        if "API_KEY" in str(e).upper():
             st.error(f"Embedding oluşturulurken API Anahtarı hatası: {str(e)}. Lütfen Secrets'ı kontrol edin.")
        else:
             st.error(f"Embedding oluşturulurken hata: {str(e)}")
        st.stop()

    return recipe_docs, doc_ids, np.array(embeds)


# ✅ Kosinüs benzerliği (Değişiklik yok)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (Doğrudan SDK RAG)")
st.divider()

# Veri ve embeddingleri yükle
docs, ids, embeddings = load_data_and_embeddings()

st.caption(f"Veri tabanımızda {len(docs)} tarif bulunmaktadır. (Gemini 2.5 Flash ile güçlendirilmiştir)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ne pişirmek istersin? (örn: Ispanaklı bir şey)")

if query:
    st.session_state.history.append({"role": "user", "content": query})

    with st.spinner("Tarif aranıyor ve yanıt oluşturuluyor..."):
        try:
            # 1. Sorgu embed (configure olmadan)
            q_res = genai.embed_content(
                model=embedding_model_name,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            q_embed = np.array(q_res['embedding'])

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

            # 5. LLM'ye gönderme (configure olmadan)
            llm = genai.GenerativeModel(model_name=llm_model_name)
            response = llm.generate_content(prompt)

            llm_response = response.text

            # Geçmişe ekle
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        except Exception as e:
             # Hata mesajına API anahtarı kontrolü ekleyelim
            if "API_KEY" in str(e).upper() or "permission denied" in str(e).lower():
                 st.error(f"RAG/API Hatası: {str(e)}. Lütfen Secrets bölümündeki GEMINI_API_KEY'i kontrol edin.")
            else:
                 st.error(f"RAG/API Hatası: {str(e)}")
            st.session_state.history.append({"role": "assistant", "content": f"Üzgünüm, bir hata oluştu: {e}", "sources": []})


# Geçmişi gösterme (Değişiklik yok)
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**
