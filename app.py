import streamlit as st
import os
import pandas as pd
import chromadb
from datasets import load_dataset
from typing import List

from google import genai
from google.genai import types

from chromadb.api.types import EmbeddingFunction

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ API Key bulunamadı. Streamlit Secrets kısmına GEMINI_API_KEY ekleyin.")
    st.stop()


@st.cache_resource
def get_ai_client():
    return genai.Client(api_key=API_KEY)


@st.cache_data
def load_and_prepare_data():
    dataset = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
    df = dataset.to_pandas()
    df['full_recipe'] = df.apply(
        lambda row: f"TARİF ADI: {row['Title']}\nMALZEMELER: {', '.join(row['Ingredients'])}\nADIMLAR: {row['Instructions']}",
        axis=1
    )
    df['id'] = [f"doc_{i}" for i in range(len(df))]
    return df['full_recipe'].tolist(), df['id'].tolist()


# ----------------------------------------------------------------------
# ✅ Chroma İçin Uyumlu Embedding Sınıfı
# ----------------------------------------------------------------------
class ChromaGeminiEmbedFunction(EmbeddingFunction):
    def __init__(self, client):
        self.client = client
        self.model = "models/text-embedding-004"
        self._ef_name = "gemini_chroma_wrapper"

    def name(self):
        return self._ef_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            res = self.client.models.embed_content(
                model=self.model,
                input=t
            )
            vectors.append(res.embedding.values)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        res = self.client.models.embed_content(
            model=self.model,
            input=text
        )
        return res.embedding.values

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


# ----------------------------------------------------------------------
# ✅ ChromaDB Bağlantısı - Bellek içi
# ----------------------------------------------------------------------
@st.cache_resource
def get_chroma_db(recipe_docs, doc_ids):
    client = get_ai_client()
    embed_fn = ChromaGeminiEmbedFunction(client)

    chroma_client = chromadb.Client()

    collection_name = "yemek_tarifleri_rag"

    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embed_fn
        )
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embed_fn
        )

    if collection.count() == 0:
        collection.add(
            documents=recipe_docs,
            ids=doc_ids
        )

    return collection


# ----------------------------------------------------------------------
# ✅ Uygulama Arayüzü
# ----------------------------------------------------------------------
recipe_docs, doc_ids = load_and_prepare_data()
db_collection = get_chroma_db(recipe_docs, doc_ids)
ai_client = get_ai_client()
llm_model = "gemini-2.5-flash"

st.set_page_config(page_title="Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Yemek Tarifleri Chatbotu")
st.caption(f"Veri tabanında {len(recipe_docs)} tarif mevcut.")
st.divider()

if 'history' not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Tarif sorunuzu girin...")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})

    with st.spinner("Tarif aranıyor..."):
        try:
            results = db_collection.query(
                query_texts=[user_query],
                n_results=3,
                include=['documents']
            )

            docs = results['documents'][0] if results['documents'] else []
            context = "\n---\n".join(docs)

            if not docs:
                llm_response = "Üzgünüm, uygun tarif bulamadım."
            else:
                PROMPT = f"""
Aşağıdaki içerik yemek tarifleridir. Kullanıcının sorusuna yardımcı olacak şekilde yanıt üret.

BAĞLAM:
{context}

SORU: {user_query}
YANIT:
"""

                response = ai_client.models.generate_content(
                    model=llm_model,
                    contents=PROMPT
                )
                llm_response = response.text

            st.session_state.history.append({"role": "assistant", "content": llm_response})

        except Exception as e:
            st.session_state.history.append({"role": "assistant", "content": f"Hata: {e}"})


for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
