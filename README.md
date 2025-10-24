
🍽️ Akbank GenAI Bootcamp Projesi: RAG Tabanlı Yemek Tarifleri Chatbotu (HF Embedding + Groq LLM)🥐🍳
1. Giriş ve Proje Amacı
Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, Retrieval Augmented Generation (RAG) mimarisi kullanan interaktif bir yemek tarifi chatbotudur. Amacı, harici bir bilgi tabanına (yemek tarifleri) dayalı olarak kullanıcı sorularına doğru, bağlamsal ve güvenilir yanıtlar üretmektir. Proje, yaşanan çeşitli API ve uyumluluk sorunlarının ardından, stabil ve hızlı çalışması için açık kaynaklı Hugging Face embedding modelleri ve Groq API LLM'leri üzerine kurulmuştur.

2. Çözüm Mimarisi ve Kullanılan Teknolojiler
Projenin teknik omurgası, LLM'nin yanıtını harici kaynaklardan gelen verilerle desteklediği modern bir RAG yaklaşımına dayanmaktadır. Karşılaşılan zorluklar nedeniyle, başlangıçta planlanan bazı kütüphaneler (örn: LangChain, ChromaDB) çıkarılmış ve daha temel, stabil bileşenler tercih edilmiştir.

2.1. Dağıtım ve Çalışma Ortamı
Dağıtım Platformu: Streamlit Cloud (Kalıcı ve erişilebilir URL sağlar).

Arayüz: Streamlit (Kullanıcı etkileşimi için).

2.2. RAG Bileşenleri
LLM (Generation): Groq API üzerinden sunulan llama-3.1-8b-instant modeli kullanılmıştır. Kullanıcı sorgusuna göre, sağlanan bağlamı kullanarak akıcı ve ilgili yanıtlar üretir.

Embedding Modeli: Hugging Face Sentence Transformers kütüphanesinden all-MiniLM-L6-v2 modeli kullanılmıştır. Tarif metinlerini ve kullanıcı sorgularını anlamsal vektörlere dönüştürür.

Vektör Araması: Temel NumPy kütüphanesi ve Cosine Similarity (Kosinüs Benzerliği) algoritması kullanılarak, kullanıcı sorgusuna en alakalı kaynak tarifler (ilk 3) bulunur.

Orkestrasyon & Arayüz: Streamlit kullanılarak, RAG sürecinin adımları (embedding oluşturma, benzerlik hesaplama, LLM'ye prompt gönderme) yönetilmiş ve kullanıcı arayüzü oluşturulmuştur.

3. Veri Seti Hakkında Bilgi
Veri Kaynağı: Hugging Face Datasets platformundan elde edilen Hieu-Pham/kaggle_food_recipes veri seti.

Hazırlık: Prototipleme, hızlı başlangıç ve API limitlerini aşmamak amacıyla veri setinin ilk 200 adet tarifi kullanılmıştır. Tariflerin Title, Ingredients ve Instructions sütunları birleştirilerek RAG için işlenmeye hazır hale getirilmiştir.

4. Elde Edilen Sonuçlar
Stabil Çalışma: Yaşanan birçok bağımlılık, API limiti ve uyumsuzluk sorunu aşılarak, uygulama Streamlit Cloud üzerinde kalıcı bir linkle başarıyla çalışır hale getirilmiştir.

Performans: Embedding işlemi için yerel Sentence Transformers ve LLM yanıtları için hızlı Groq API kullanılarak tatmin edici bir yanıt süresi elde edilmiştir.

Doğruluk ve Kaynak Gösterimi: Chatbot, sadece kendi veri tabanından (ilk 200 tarif) çektiği bilgilere dayanarak yanıtlar üretir. Yanıtla birlikte kullanılan kaynak tariflerin adları da kullanıcıya gösterilir.

5. Kodun Çalışma Kılavuzu (Local Çalıştırma)
Projenin yerel makinede çalıştırılabilmesi için gereken adımlar aşağıdadır:

Gerekli Dosyalar: app.py, requirements.txt dosyaları repoda mevcuttur.


6. 🔗 Web Arayüzü Linki (Product Kılavuzu)
Projenin Streamlit Cloud'da çalışan kalıcı web linki aşağıdadır. Uygulamayı test etmek için bu linki kullanın.

Kalıcı Web URL'si:
https://akbankgenai-diqgapppp2uk8gjg92cu6jw7.streamlit.app/

Test Kılavuzu: Uygulamaya girdikten sonra, "salçalı bir tarif", "mac and cheese nasıl yapılır" veya "falafel tarifi" gibi sorular sorarak chatbot'u test edebilirsiniz. Yanıtla birlikte kullanılan kaynak tarifler de gösterilecektir.
