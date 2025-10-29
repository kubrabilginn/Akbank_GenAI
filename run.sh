#!/bin/bash
# run.sh: Uygulamayı başlatan ana script

# 2. CACHE Dizinlerini Kullanıcının İzinli Alanına Yönlendirme
# Bu, 'Permission denied' hatasını çözer.
export XDG_CACHE_HOME=/home/appuser/.cache
export STREAMLIT_SERVER_PORT=7860

# 3. Streamlit uygulamasını başlatma komutu
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
