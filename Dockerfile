FROM python:3.10-slim

# ... [Diğer kurulum komutları] ...

# Dosyaları Kopyala (Sırayı koruyun)
COPY requirements.txt .
COPY app.py .
COPY run.sh . 

# Python bağımlılıklarını kur
RUN pip install --no-cache-dir -r requirements.txt

# ÇÖZÜM: 'Permission denied' hatasını çözmek için kullanıcı oluşturma ve HOME dizini ayarlama
RUN useradd -m appuser
USER appuser
ENV HOME=/home/appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Bu, Hugging Face/Datasets'in cache dizinini kullanıcının HOME dizinine yönlendirir.
ENV HF_HOME /home/appuser/.cache/huggingface
ENV STREAMLIT_SERVER_PORT=7860

# Streamlit Portunu dışarıya aç
EXPOSE 7860

# Uygulamayı başlatmak için run.sh'i kullan
ENTRYPOINT ["/bin/bash", "run.sh"]