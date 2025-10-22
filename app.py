import streamlit as st
import os
import pandas as pd
import chromadb
from datasets import load_dataset
from typing import List

# Google SDK ve diğer yardımcılar
from google import genai
from google.genai import types

# ----------------------------------------------------------------------
# 1. API Anahtarının Güvenli Kontrolü
# ----------------------------------------------------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ API Anahtarı bulunamadı. Lütfen Streamlit Secrets'ta 'GEMINI_API_KEY' Secret'ını ayarlayın.")
    st.stop()

# ----------------------------------------------------------------------
# 2. RAG Bileşenleri Tanımları (Cache ile Hızlandırma)
# ----------------------------------------------------------------------

# Google GenAI İstemcisini oluşturma
@st.cache_resource
def get_ai_client():
    return genai.Client(api_key=API_KEY)

# Veri Seti Yükleme ve Hazırlama
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


# --- ChromaDB Uyumlu Embedding Wrapper Sınıfı ---
class ChromaGeminiEmbedFunction:
    """ChromaDB'nin beklediği arayüzü sağlayan özel sınıf."""
    
    def __init__(self, client):
        self.client = client
        self._name = "gemini_custom_embedder_v5" # Sabit isim
        self.model = "embedding-001"

    def name(self):
        return self._name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.batch_embed_content(
            model=self.model,
            contents=texts
        )
        return [r.values for r in response.embeddings]
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
    
# --- Wrapper Sınıfı Sonu ---


@st.cache_resource
def get_chroma_db(recipe_docs, doc_ids):
    client = get_ai_client()
    gemini_embed_function = ChromaGeminiEmbedFunction(client) 
    collection_name = "yemek_tarifleri_rag"
    
    # ChromaDB'yi bellek içi modda başlatma
    chroma_client = chromadb.Client()
    
    try:
        # Önce koleksiyonu ALMAYI dener
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=gemini_embed_function 
        )
    except:
        # Koleksiyon yoksa OLUŞTURUR.
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=gemini_embed_function
        )
        
    # Belgeleri ekleme (Sadece ilk kez eklenecektir)
    if collection.count() == 0:
        collection.add(
            documents=recipe_docs,
            ids=doc_ids
        )
    
    return collection

# ----------------------------------------------------------------------
# 3. Streamlit Uygulama Arayüzü (Ana İşlem)
# ----------------------------------------------------------------------

# RAG bileşenlerini yükle
recipe_docs, doc_ids = load_and_prepare_data()
db_collection = get_chroma_db(recipe_docs, doc_ids)
ai_client = get_ai_client()
llm_model = "gemini-2.5-flash"


st.set_page_config(page_title="Akbank GenAI Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (Doğrudan SDK RAG)")
st.caption(f"Veri tabanımızda {len(recipe_docs)} tarif bulunmaktadır. (Gemini 2.5 Flash ile güçlendirilmiştir)")
st.divider()

if 'history' not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Tarif sorunuzu girin (Örn: Ispanak ve peynirle ne yapabilirim?)")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    
    with st.spinner(f"'{user_query}' için tarif aranıyor..."):
        try:
            # 1. Geri Getirme (Retrieval) - ChromaDB'den kaynak bulma
            results = db_collection.query(
                query_texts=[user_query],
                n_results=3,
                include=['documents']
            )
            
            context = "\n---\n".join(results['documents'][0])
            source_names = [doc.split('\n')[0].replace('TARİF ADI: ', '') for doc in results['documents'][0]]

            # 2. Üretim (Generation) - Prompt oluşturma
            PROMPT = f"""Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver. 
Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla.

BAĞLAM:
{context}

SORU: {user_query}
YANIT:"""
            
            # 3. LLM'ye gönderme
            response = ai_client.models.generate_content(
                model=llm_model,
                contents=PROMPT
            )
            
            llm_response = response.text
            
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        except Exception as e:
            error_msg = f"RAG/API Hatası: Lütfen API anahtarınızın doğru olduğundan emin olun. Detay: {e}"
            st.session_state.history.append({"role": "assistant", "content": error_msg, "sources": []})

# Geçmişi gösterme
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            for name in set(message["sources"]):
                st.markdown(f"**-** *{name}*")
