import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List

# Google SDK ve diğer yardımcılar
from google import genai
from google.genai import types

# ------------------------------------------------
# API Key kontrolü ve SDK Yapılandırması
# ------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ API Anahtarı bulunamadı. Streamlit Secrets bölümüne GEMINI_API_KEY ekleyin.")
    st.stop()

# 🛑 KESİN ÇÖZÜM: genai.configure() çağrısını geri getiriyoruz.
# Bu, SDK'nın embed_content gibi fonksiyonları bulmasını sağlar.
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Google AI SDK yapılandırılırken hata: {e}")
    st.stop()

embedding_model_name = "models/embedding-001"
llm_model_name = "gemini-2.5-flash"

# ------------------------------------------------
# Tarifleri yükleme (Değişiklik yok)
# ------------------------------------------------
from datasets import load_dataset

@st.cache_data(show_spinner="Tarifler yükleniyor...")
def load_recipes() -> list[str]:
    # ... (Fonksiyon içeriği aynı kalır) ...
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
# Veri ve Embedding Cache (configure ile çalışmalı)
# ------------------------------------------------
@st.cache_data(show_spinner="Veriler ve embeddingler hazırlanıyor...")
def load_data_and_embeddings():
    recipe_docs = load_recipes()
    doc_ids = [f"doc_{i}" for i in range(len(recipe_docs))]

    try:
        # configure çağrıldığı için API anahtarını tekrar
