import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List
import traceback # Hata ayıklama için

# Hugging Face Kütüphaneleri
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ------------------------------------------------
# Hugging Face API Token (Opsiyonel ama Önerilir)
# ------------------------------------------------
# Streamlit Secrets'a HF_TOKEN ekleyebilirsiniz (daha yüksek limitler için)
HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     st.warning("Hugging Face API Token bulunamadı. Ücretsiz limitlerle devam ediliyor.")

# Model isimleri
embedding_model_name = 'all-MiniLM-L6-v2'
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Veya "HuggingFaceH4/zephyr-7b-beta"

# Hugging Face Inference Client'ı başlatma
try:
    # HF_TOKEN varsa kullanılır, yoksa ücretsiz limitlerle çalışır
    hf_client = InferenceClient(model=llm_model_name, token=HF_TOKEN)
except Exception as e:
    st.error(f"Hugging Face Inference Client başlatılırken hata: {e}")
    st.stop()


# ------------------------------------------------
# Sentence Transformer Modelini Yükleme (Cache ile)
# ------------------------------------------------
@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embedding_model():
    """Hugging Face'den Sentence Transformer modelini yükler."""
    try:
        model = SentenceTransformer(embedding_model_name)
        return model
    except Exception as e:
        st.error(f"Sentence Transformer modeli yüklenirken hata: {e}")
        st.stop()

# ------------------------------------------------
# Tarifleri yükleme
# ------------------------------------------------
@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    """
    Hugging Face datasets üzerinden yemek tariflerini yükler.
    İlk 200 tarifi alır ve Title + Ingredients + Instructions şeklinde birleştirir.
    """
    try:
        # Tarif sayısını 200'e geri çıkardık
        ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
        recipes = []
        for item in ds:
            title = item.get("Title", "")
            ingredients = item.get("Ingredients", [])
            if isinstance(ingredients, list):
                ingredients = ", ".join(ingredients) # Listeyi string'e çevir
            else:
                ingredients = str(ingredients) # Güvenlik için string'e çevir
            instructions = item.get("Instructions", "")
            full_recipe = f"TARİF ADI: {title}\nMALZEMELER: {ingredients}\nADIMLAR: {instructions}"
            recipes.append(full_recipe)
        if not recipes:
             st.error("Tarifler yüklenemedi veya veri seti boş.")
             st.stop()
        return recipes
    except Exception as e:
        st.error(f"Veri seti yüklenirken hata: {e}")
        st.stop()


# ------------------------------------------------
# Veri ve Embedding Cache (Sentence Transformer ile)
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler hazırlanıyor...")
def load_data_and_embeddings(_embedding_model): # Model artık argüman olarak geliyor
    """Tarifleri yükler ve Sentence Transformer ile embed eder."""
    recipe_docs = load_recipes()
    if not recipe_docs: # Eğer tarif yüklenemediyse dur
        st.error("Tarif dokümanları boş, embedding oluşturulamıyor.")
        st.stop()

    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]
    try:
        # Sentence Transformer ile embedding oluşturma
        embeds = _embedding_model.encode(recipe_docs, show_progress_bar=False)
        if embeds is None or len(embeds) == 0:
             st.error("Embedding oluşturma başarısız oldu, boş sonuç döndü.")
             st.stop()
        # Embeddings listesini NumPy array'e çevir
        embeds_np = np.array(embeds)
        return recipe_docs, doc_ids, embeds_np
    except Exception as e:
        st.error(f"Embedding oluşturulurken hata: {str(e)}")
        st.stop()


# ✅ Kosinüs benzerliği
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki NumPy vektörü arasındaki kosinüs benzerliğini hesaplar."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Sıfır vektörler için benzerlik 0'dır
    # Hesaplamayı float32 ile yaparak hassasiyeti artırabiliriz
    dot_product = np.dot(a.astype(np.float32), b.astype(np.float32))
    return dot_product / (norm_a * norm_b)

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (HF Embedding + HF LLM)")
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

            # 2. Cosine similarity hesapla
            # Hata ayıklama: Embedding boyutlarını kontrol et
            if embeddings.shape[1] != q_embed.shape[0]:
                 st.error(f"Embedding boyutları uyuşmuyor! Döküman: {embeddings.shape[1]}, Sorgu: {q_embed.shape[0]}")
                 st.stop()

            sims = [(i, cosine_similarity(q_embed, emb)) for i, emb in enumerate(embeddings)]
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3] # En iyi 3 sonucu al

            # 3. En iyi dokümanları (context) al
            top_docs_content = [docs[i] for i, score in sims if score > 0.5] # Benzerlik eşiği ekleyelim (opsiyonel)
            if not top_docs_content:
                 llm_response = "Üzgünüm, bu isteğe yeterince benzer bir tarif bulamadım."
                 source_names = []
            else:
                source_names = [doc.split('\n')[0].replace('TARİF ADI: ', '') for doc in top_docs_content]
                context = "\n---\n".join(top_docs_content)

                # 4. Prompt oluşturma (Mistral/Zephyr formatı)
                prompt = f"""<s>[INST] Aşağıdaki bağlamda sana verilen yemek tariflerini kullanarak, kullanıcının sorusuna detaylı ve yardımcı bir şekilde yanıt ver.
Eğer bağlamda uygun tarif bulamazsan, kibarca sadece "Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım." diye yanıtla. Yanıtını sadece Türkçe ver.

BAĞLAM:
{context}

SORU: {query} [/INST]
YANIT:"""

                # 5. Hugging Face Inference API'sine gönderme
                response = hf_client.text_generation(
                    prompt,
                    max_new_tokens=300, # Yanıt uzunluğunu biraz artırdık
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True # Daha yaratıcı yanıtlar için
                )

                llm_response = response.strip() # Gelen yanıtı temizle

            # Geçmişe ekle
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        except Exception as e:
            # Hata yakalamayı detaylandırdık
            tb_str = traceback.format_exc()
            error_msg = f"RAG/API Hatası Oluştu!\nDetaylar:\n{str(e)}\n\nTraceback:\n{tb_str}"
            st.error(error_msg)
            print(error_msg) # Loglara da yazdır
            st.session_state.history.append({"role": "assistant", "content": f"Üzgünüm, bir hata oluştu: {str(e)}", "sources": []})


# Geçmişi gösterme
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            # Kaynak listesi boş değilse göster
            sources = msg.get("sources", [])
            if sources:
                 for name in set(sources):
                     st.markdown(f"**-** *{name}*")
            else:
                 st.markdown("*Bu yanıt için özel bir tarif kullanılmadı.*")
