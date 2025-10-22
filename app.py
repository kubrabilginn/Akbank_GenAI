
import streamlit as st
import os
import pandas as pd
from datasets import load_dataset
# LangChain bileşenleri
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ----------------------------------------------------------------------
# 1. API Anahtarının Güvenli Kontrolü
# ----------------------------------------------------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ API Anahtarı bulunamadı. Lütfen Streamlit Cloud'da 'GEMINI_API_KEY' Secret'ını ayarlayın.")
    st.stop()
# Hata çözümü için langsmith takibini uygulama seviyesinde devre dışı bırakma
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_SESSION"] = "false"# ----------------------------------------------------------------------
# 2. RAG Bileşenleri Tanımları (FONKSİYONLAR BURADA BAŞLAR)
# ----------------------------------------------------------------------

# LLM ve Embedding Modelini Tanımlama (Doğrudan API Anahtarı ile)
@st.cache_resource
def get_llm_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2, 
        google_api_key=API_KEY 
    )

@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        google_api_key=API_KEY 
    )

# Veri Seti Yükleme ve Hazırlama
@st.cache_data
def load_and_prepare_data():
    dataset = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
    df = dataset.to_pandas()
    df['full_recipe'] = df.apply(
        lambda row: f"TARİF ADI: {row['Title']}\nMALZEMELER: {', '.join(row['Ingredients'])}\nADIMLAR: {row['Instructions']}",
        axis=1
    )
    return df['full_recipe'].tolist()

# Vektör Veritabanı ve Retriever'ı Yükleme/Oluşturma
@st.cache_resource
def get_retriever(recipe_docs):
    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_texts(
        texts=recipe_docs, 
        embedding=embedding_model, 
        collection_name="yemek_tarifleri_rag"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG Zincirini Kurma (Cache dekoratörü kalıcı olarak kaldırıldı)
def get_qa_chain(retriever):
    llm = get_llm_model()
    
    PROMPT_TEMPLATE = """Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver. 
    Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla ve dışarıdan bilgi ekleme.

    BAĞLAM:
    {context}

    SORU: {question}
    YANIT:"""
    custom_rag_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_rag_prompt},
        return_source_documents=True
    )

# ----------------------------------------------------------------------
# 3. Streamlit Uygulama Arayüzü (Ana İşlem) - HER ŞEY BURADA BAŞLAR
# ----------------------------------------------------------------------

# 3.1 RAG Bileşenlerini Yükleme/Kurma
# Bu çağırmalar artık fonksiyon tanımlarının altında yapıldığı için NameError çözüldü.
recipe_docs = load_and_prepare_data()
retriever = get_retriever(recipe_docs)
qa_chain = get_qa_chain(retriever)


# 3.2 Arayüz Başlıkları
st.set_page_config(page_title="Akbank GenAI Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (RAG)")
st.caption(f"Veri tabanımızda {len(recipe_docs)} tarif bulunmaktadır. (Gemini 2.5 Flash ile güçlendirilmiştir)")
st.divider()

if 'history' not in st.session_state:
    st.session_state.history = []

# Kullanıcı Girişi
user_query = st.chat_input("Tarif sorunuzu girin (Örn: Ispanak ve peynirle ne yapabilirim?)")

if user_query:
    # Kullanıcı sorgusunu kaydet
    st.session_state.history.append({"role": "user", "content": user_query})
    
    with st.spinner(f"'{user_query}' için tarif aranıyor..."):
        try:
            # RAG Zincirini Çalıştırma
            response = qa_chain.invoke({"query": user_query})
            llm_response = response['result']
            source_docs = response['source_documents']

            # Yanıtı ve kaynakları geçmişe ekleme
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_docs})

        except Exception as e:
            error_msg = f"API Hatası: Lütfen API anahtarınızın Streamlit Secrets'ta doğru ayarlandığından emin olun. Hata: {e}"
            st.session_state.history.append({"role": "assistant", "content": error_msg, "sources": []})

# Geçmişi gösterme
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Yanıtta kullanılan kaynakları göster
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            source_names = [doc.page_content.split('\n')[0].replace('TARİF ADI: ', '') for doc in message["sources"]]
            for name in set(source_names):
                st.markdown(f"**-** *{name}*")
