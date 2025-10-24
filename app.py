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

# Google AI Client'ı başlat
try:
    genai.configure(api_key=API_KEY)
    client = genai.GenerativeModel # Client yerine doğrudan modeli kullanacağız
except Exception as e:
    st.error(f"Google AI Client başlatılırken hata: {e}")
    st.stop()

embedding_model = "models/embedding-001" # Daha stabil olan embedding modeli
llm_model = "gemini-2.5-flash"

# ------------------------------------------------
# Tarifleri yükleme
# ------------------------------------------------
from datasets import load_dataset

@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
    recipes = []
    for item in ds:
        title = item.get("Title", "")
        # Malzemeler listesini string'e çevir (eğer liste ise)
        ingredients = item.get("Ingredients", [])
        if isinstance(ingredients, list):
            ingredients = ", ".join(ingredients)
        instructions = item.get("Instructions", "")
        full_recipe = f"TARİF ADI: {title}\nMALZEMELER: {ingredients}\nADIMLAR: {instructions}"
        recipes.append(full_recipe)
    return recipes

# ------------------------------------------------
# Veri ve Embedding Cache
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler hazırlanıyor...")
def load_data_and_embeddings():
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]
    
    try:
        # Toplu embedding isteği
        result = genai.embed_content(
            model=embedding_model,
            content=recipe_docs, # DÜZELTME: 'text' yerine 'content' ve toplu istek
            task_type="RETRIEVAL_DOCUMENT" # RAG için doğru task_type
        )
        embeds = result['embedding']
    except Exception as e:
        st.error(f"Embedding oluşturulurken hata: {str(e)}")
        st.stop() # Hata varsa uygulamayı durdur

    return recipe_docs, doc_ids, np.array(embeds) # Embeddings'i NumPy array olarak döndür


# ✅ Kosinüs benzerliği
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # NumPy arrayleri üzerinden daha hızlı hesaplama
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
            # 1. Sorgu embed
            q_res = genai.embed_content(
                model=embedding_model,
                content=query, # DÜZELTME: 'text' yerine 'content'
                task_type="RETRIEVAL_QUERY" # Sorgu için doğru task_type
            )
            q_embed = np.array(q_res['embedding'])

            # 2. Cosine similarity hesapla ve en iyi 3'ü bul
            sims = [(i, cosine_similarity(q_embed, emb)) for i, emb in enumerate(embeddings)]
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]

            # 3. En iyi dokümanları (context) al
            top_docs_content = [docs[i] for i, _ in sims]
            source_names = [doc.split('\n')[0].replace('TARİF ADI: ', '') for doc in top_docs_content]
            context = "\n---\n".join(top_docs_content)

            # 4. Prompt oluşturma
            prompt = f"""Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver. 
Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla.

BAĞLAM:
{context}

SORU: {query}
YANIT:"""

            # 5. LLM'ye gönderme
            llm = client(model_name=llm_model) # Modeli başlat
            response = llm.generate_content(prompt)
            
            llm_response = response.text
            
            # Geçmişe ekle
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        except Exception as e:
            error_msg = f"RAG/API Hatası: {str(e)}"
            st.session_state.history.append({"role": "assistant", "content": error_msg, "sources": []})

# Geçmişi gösterme
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            for name in set(msg.get("sources", [])):
                st.markdown(f"**-** *{name}*")
