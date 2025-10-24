import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List

# Hugging Face Kütüphaneleri
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ------------------------------------------------
# Hugging Face API Token (Opsiyonel ama Önerilir)
# ------------------------------------------------
# HF_TOKEN = os.environ.get("HF_TOKEN") # Streamlit Secrets'a HF_TOKEN ekleyebilirsiniz (daha yüksek limitler için)
# if not HF_TOKEN:
#     st.warning("Hugging Face API Token bulunamadı. Ücretsiz limitlerle devam ediliyor.")

# Model isimleri
embedding_model_name = 'all-MiniLM-L6-v2'
# LLM: Hugging Face Inference API üzerinden popüler bir model
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Veya daha küçük bir model: "HuggingFaceH4/zephyr-7b-beta"

# Hugging Face Inference Client'ı başlatma
try:
    # HF_TOKEN varsa kullanılır, yoksa ücretsiz limitlerle çalışır
    hf_client = InferenceClient(model=llm_model_name) #, token=HF_TOKEN) 
except Exception as e:
    st.error(f"Hugging Face Inference Client başlatılırken hata: {e}")
    st.stop()


# ------------------------------------------------
# Sentence Transformer Modelini Yükleme (Cache ile)
# ------------------------------------------------
@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embedding_model():
    return SentenceTransformer(embedding_model_name)

# ------------------------------------------------
# Tarifleri yükleme (Değişiklik yok)
# ------------------------------------------------
# app.py dosyasındaki load_recipes fonksiyonunu bulun:

@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    # 🛑 DEĞİŞİKLİK BURADA: 50 yerine 200 tarif yükle
    ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]") 
    # ... (fonksiyonun geri kalanı aynı) ...
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
# Veri ve Embedding Cache (Sentence Transformer ile)
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler hazırlanıyor...")
def load_data_and_embeddings(_embedding_model):
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]
    try:
        embeds = _embedding_model.encode(recipe_docs, show_progress_bar=False).tolist()
    except Exception as e:
        st.error(f"Embedding oluşturulurken hata: {str(e)}")
        st.stop()
    return recipe_docs, doc_ids, np.array(embeds)


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
st.title("🍽️ GenAI Yemek Tarifleri Chatbotu 🥗")
st.divider()

# Embedding modelini yükle
embedding_model = load_embedding_model()
# Veri ve embeddingleri yükle
docs, ids, embeddings = load_data_and_embeddings(embedding_model)

st.caption(f"Veri tabanımızda {len(docs)} tarif bulunmaktadır. (LLM: {llm_model_name} | Embedding: {embedding_model_name})")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ne pişirmek istersin? (örn: Ispanaklı bir şey)")

# app.py dosyasındaki if query: bloğunun içindeki try...except bölümü

if query:
    st.session_state.history.append({"role": "user", "content": query})

    with st.spinner("Tarif aranıyor ve yanıt oluşturuluyor..."):
        try:
            # ... (Embedding ve Arama kodları aynı kalır) ...
            
            # 4. Prompt oluşturma (Değişiklik yok)
            prompt = f"""<s>[INST] Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver. 
Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla.

BAĞLAM:
{context}

SORU: {query} [/INST]
YANIT:"""

            # 5. Hugging Face Inference API'sine gönderme
            response = hf_client.text_generation(
                prompt,
                max_new_tokens=250, 
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            llm_response = response.strip() 

            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        # 🛑 HATA YAKALAMAYI DETAYLANDIRDIK 🛑
        except Exception as e:
            import traceback # Traceback'i yazdırmak için import et
            tb_str = traceback.format_exc() # Hatanın tam traceback'ini al
            
            # Hem arayüze hem de loglara detaylı hata yazdır
            error_msg = f"RAG/API Hatası Oluştu!\nDetaylar:\n{str(e)}\n\nTraceback:\n{tb_str}"
            st.error(error_msg) # Arayüzde göster
            print(error_msg)    # Streamlit loglarına yazdır
            
            # Geçmişe basit hata mesajı ekle
            st.session_state.history.append({"role": "assistant", "content": f"Üzgünüm, bir hata oluştu: {str(e)}", "sources": []})

# ... (Geçmişi gösterme kodu aynı kalır) ...

# Geçmişi gösterme (Değişiklik yok)
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            for name in set(msg.get("sources", [])):
                st.markdown(f"**-** *{name}*")
