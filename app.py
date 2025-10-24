import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List
import traceback # Hata ayıklama için

# Gerekli Kütüphaneler
from sentence_transformers import SentenceTransformer # Embedding için
from groq import Groq                             # LLM için
from datasets import load_dataset

# ------------------------------------------------
# API Key kontrolü (Artık Groq için)
# ------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Groq API Anahtarı bulunamadı. Streamlit Secrets bölümüne GROQ_API_KEY ekleyin.")
    st.stop()

# Groq İstemcisini başlatma
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Groq istemcisi başlatılırken hata: {e}")
    st.stop()

# Model isimleri
embedding_model_name = 'all-MiniLM-L6-v2' # HF Embedding
llm_model_name = "llama-3.1-8b-instant" # Güncel ve aktif Groq modeli
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
# Tarifleri yükleme (Değişiklik yok)
# ------------------------------------------------
@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    """Hugging Face datasets üzerinden yemek tariflerini yükler (200 adet)."""
    try:
        ds = load_dataset("Hieu-Pham/kaggle_food_recipes", split="train[:200]")
        recipes = []
        for item in ds:
            title = item.get("Title", "")
            ingredients = item.get("Ingredients", [])
            if isinstance(ingredients, list):
                ingredients = ", ".join(ingredients)
            else:
                ingredients = str(ingredients)
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
def load_data_and_embeddings(_embedding_model):
    """Tarifleri yükler ve Sentence Transformer ile embed eder."""
    recipe_docs = load_recipes()
    if not recipe_docs:
        st.error("Tarif dokümanları boş, embedding oluşturulamıyor.")
        st.stop()

    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]
    try:
        embeds = _embedding_model.encode(recipe_docs, show_progress_bar=False)
        if embeds is None or len(embeds) == 0:
             st.error("Embedding oluşturma başarısız oldu, boş sonuç döndü.")
             st.stop()
        embeds_np = np.array(embeds)
        return recipe_docs, doc_ids, embeds_np
    except Exception as e:
        st.error(f"Embedding oluşturulurken hata: {str(e)}")
        st.stop()


# ✅ Kosinüs benzerliği (Değişiklik yok)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot_product = np.dot(a.astype(np.float32), b.astype(np.float32))
    return dot_product / (norm_a * norm_b)

# ------------------------------------------------
# UI
# ------------------------------------------------
st.set_page_config(page_title="Yemek Tarifleri Chatbotu", layout="wide")
st.title("🍽️ Akbank GenAI Yemek Tarifleri Chatbotu (HF Embedding + Groq LLM)")
st.divider()

# Embedding modelini yükle
embedding_model = load_embedding_model()
# Veri ve embeddingleri yükle
docs, ids, embeddings = load_data_and_embeddings(embedding_model)

st.caption(f"Veri tabanımızda {len(docs)} tarif bulunmaktadır. (LLM: {llm_model_name} @ Groq | Embedding: {embedding_model_name})")

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
            if embeddings.shape[1] != q_embed.shape[0]:
                 st.error(f"Embedding boyutları uyuşmuyor! Döküman: {embeddings.shape[1]}, Sorgu: {q_embed.shape[0]}")
                 st.stop()

            sims = [(i, cosine_similarity(q_embed, emb)) for i, emb in enumerate(embeddings)]
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]

            # 3. En iyi dokümanları (context) al
            top_docs_content = [docs[i] for i, score in sims]
            if not top_docs_content:
                 llm_response = "Üzgünüm, bu isteğe uygun bir tarif bulamadım."
                 source_names = []
            else:
                source_names = [doc.split('\n')[0].replace('TARİF ADI: ', '') for doc in top_docs_content]
                context = "\n---\n".join(top_docs_content)

                # 4. Prompt oluşturma (Groq için mesaj formatı)
                # Llama3 ve Mixtral genellikle bu formatı anlar
                system_prompt = """Sen yardımsever bir yemek tarifi asistanısın. Sana verilen bağlamdaki tarifleri kullanarak kullanıcının sorusuna cevap ver. Eğer bağlamda uygun tarif yoksa, sadece 'Üzgünüm, veri tabanımda bu isteğe uygun bir tarif bulamadım.' de. Yanıtını sadece Türkçe ver."""
                user_prompt = f"""BAĞLAM:
{context}

SORU: {query}"""

                # 5. Groq API'sine gönderme
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                    model=llm_model_name,
                    temperature=0.7,
                    max_tokens=300,
                    top_p=0.9,
                )

                llm_response = chat_completion.choices[0].message.content.strip()

            # Geçmişe ekle
            st.session_state.history.append({"role": "assistant", "content": llm_response, "sources": source_names})

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"RAG/API Hatası Oluştu!\nDetaylar:\n{str(e)}\n\nTraceback:\n{tb_str}"
            st.error(error_msg)
            print(error_msg) # Loglara da yazdır
            st.session_state.history.append({"role": "assistant", "content": f"Üzgünüm, bir hata oluştu: {str(e)}", "sources": []})


# Geçmişi gösterme (Değişiklik yok)
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**Kullanılan Kaynak Tarifler:**")
            sources = msg.get("sources", [])
            if sources:
                 for name in set(sources):
                     st.markdown(f"**-** *{name}*")
            else:
                 st.markdown("*Bu yanıt için özel bir tarif kullanılmadı veya bulunamadı.*")
