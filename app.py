import streamlit as st
import os
import pandas as pd
from datasets import load_dataset
import chromadb # ChromaDB'yi manuel de dahil ediyoruz

# LangChain Çekirdek ve Bağlayıcıları (requirements.txt'den geliyor)
# app.py dosyasının başındaki importları bulun (Yaklaşık 5. satır):

# Eski Hatalı Import Zinciri:
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA

# YENİ VE DÜZELTİLMİŞ IMPORT ZİNCİRİ:
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA

# KRİTİK EKSİK IMPORT: Bu satır, hatayı çözer
from langchain_core.documents import Document
# ----------------------------------------------------------------------
# 1. API Anahtarının Güvenli Kontrolü
# ----------------------------------------------------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ API Anahtarı bulunamadı. Lütfen Streamlit Secrets'ta 'GEMINI_API_KEY' Secret'ını ayarlayın.")
    st.stop()

# Ortam değişkenini (LangSmith'i engellemek için) tekrar ayarlıyoruz
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ----------------------------------------------------------------------
# 2. RAG Bileşenleri Tanımları (Cache ile Hızlandırma)
# ----------------------------------------------------------------------

# Veri Seti Yükleme ve Hazırlama
@st.cache_data
def load_and_prepare_data():
    # Load ve hazırlık kısmı aynı kalır
    dataset = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
    df = dataset.to_pandas()
    df['full_recipe'] = df.apply(
        lambda row: f"TARİF ADI: {row['Title']}\nMALZEMELER: {', '.join(row['Ingredients'])}\nADIMLAR: {row['Instructions']}",
        axis=1
    )
    # LangChain'in Document objesi için gerekli olan formatlama (metadata için)
    from langchain_core.documents import Document
    docs = [Document(page_content=recipe) for recipe in df['full_recipe']]
    return docs, df['full_recipe'].tolist() # İhtiyaç duyulursa eski liste de döner

# Embedding Modelini Tanımlama
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="embedding-001", 
        google_api_key=API_KEY 
    )

# ChromaDB ve Retriever'ı Kurma (LANGCHAIN ÜZERİNDEN)
@st.cache_resource
def get_retriever(docs):
    embedding_model = get_embedding_model()
    
    # LangChain, Chroma'yı başlatırken gerekli tüm kontrolleri (izin, name) kendisi yapar
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embedding_model, 
        collection_name="yemek_tarifleri_rag",
        # Kalıcılığı kapatmak için None kullanıyoruz, bellek içi çalışır
        persist_directory=None 
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM ve RAG Zincirini Kurma
@st.cache_resource
def get_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2, 
        google_api_key=API_KEY 
    )

    PROMPT_TEMPLATE = """Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver. 
    Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla.

    BAĞLAM:
    {context}

    SORU: {question}
    YANIT:"""
    
    from langchain.prompts import PromptTemplate # Geri getirdiğimiz paketlerden import et
    custom_rag_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_rag_prompt},
        return_source_documents=True
    )
    return qa_chain

# ----------------------------------------------------------------------
# 3. Streamlit Uygulama Arayüzü (Ana İşlem)
# ----------------------------------------------------------------------

# RAG bileşenlerini yükle
docs, _ = load_and_prepare_data()
retriever = get_retriever(docs)
qa_chain = get_qa_chain(retriever)

# ... (Kullanıcı arayüz kodu aynı kalır) ...

st.set_page_config(page_title="Akbank GenAI Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (LangChain RAG)")
st.caption(f"Veri tabanımızda {len(docs)} tarif bulunmaktadır. (Gemini 2.5 Flash ile güçlendirilmiştir)")
st.divider()

if 'history' not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Tarif sorunuzu girin (Örn: Ispanak ve peynirle ne yapabilirim?)")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    
    with st.spinner(f"'{user_query}' için tarif aranıyor..."):
        try:
            # LangChain zincirini çalıştırma
            response = qa_chain.invoke({"query": user_query})
            
            llm_response = response['result']
            source_docs = response['source_documents']
            
            # Kaynak metinleri işleme
            source_names = [doc.page_content.split('\n')[0].replace('TARİF ADI: ', '') for doc in source_docs]

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
