🍽️ Akbank GenAI Bootcamp Projesi: RAG Tabanlı Yemek Tarifleri Chatbotu (Hugging Face Ekosistemi)
1. Giriş ve Proje Amacı
Bu proje, Retrieval Augmented Generation (RAG) mimarisi kullanılarak geliştirilmiştir. Amacı, kullanıcıların spesifik yemek tarifleri hakkında soru sorabileceği, harici bir bilgi tabanına (yemek tarifleri) dayalı, doğru ve güvenilir yanıtlar üreten interaktif bir web chatbotu sunmaktır. Proje, tamamen açık kaynaklı Hugging Face modelleri ve kütüphaneleri kullanılarak oluşturulmuştur.

2. Çözüm Mimarisi ve Kullanılan Teknolojiler
Projenin teknik omurgası, LLM'nin yanıtını harici kaynaklardan gelen verilerle desteklediği modern bir RAG zincirine dayanmaktadır. Tüm yapay zeka bileşenleri Hugging Face ekosisteminden sağlanmıştır.

2.1. Dağıtım ve Çalışma Ortamı
Dağıtım Platformu: Streamlit Cloud (Kalıcı URL sağlar).

Arayüz: Streamlit.

2.2. RAG Bileşenleri
LLM (Generation): Hugging Face Inference API üzerinden sunulan bir model (Örn: Mistral) kullanılmıştır. Kullanıcı sorgusuna göre bağlamı sentezleyerek akıcı yanıtlar üretir.

Embedding Modeli: Hugging Face Sentence Transformers kütüphanesinden all-MiniLM-L6-v2 modeli kullanılmıştır. Tarif metinlerini sayısal vektörlere dönüştürerek anlamsal aramayı mümkün kılar.

Vektör Araması: Temel NumPy kütüphanesi ve Cosine Similarity (Kosinüs Benzerliği) algoritması kullanılarak, kullanıcı sorgusuna en alakalı kaynaklar (tarifler) hızlıca bulunur.

Orkestrasyon & Arayüz: Streamlit kullanılarak hem RAG sürecinin adımları (embedding, arama, LLM'ye gönderme) yönetilmiş hem de kullanıcı etkileşimini sağlayan arayüz oluşturulmuştur.

(Not: Önceki denemelerde yaşanan uyumluluk sorunları nedeniyle ChromaDB çıkarılmış, yerine temel NumPy vektör araması entegre edilmiştir.)

3. Veri Seti Hakkında Bilgi
Veri Kaynağı: Hugging Face Datasets platformundan elde edilen Hieu-Pham/kaggle_food_recipes veri seti.

Hazırlık: Prototipleme ve ücretsiz API limitleri dahilinde kalmak amacıyla veri setinin ilk 200 adet tarifi kullanılmıştır. Tariflerin Title, Ingredients ve Instructions sütunları birleştirilerek RAG için işlenmeye hazır hale getirilmiştir.

4. Elde Edilen Sonuçlar
Başarılı Dağıtım: Tüm bağımlılık ve API sorunları çözülerek uygulama, Streamlit Cloud üzerinde kalıcı bir linkle başarıyla çalışır hale getirilmiştir.

Tamamen Açık Kaynak: Proje, Google API'lerine bağımlı kalmadan, Hugging Face'in ücretsiz modelleri ve kütüphaneleri ile çalışmaktadır.

Doğruluk ve İzlenebilirlik: Chatbot, sadece kendi veri tabanından çektiği tariflere dayalı yanıtlar üretir ve kullanılan kaynak tariflerin adlarını kullanıcıya gösterir.

5. Kodun Çalışma Kılavuzu (Local Çalıştırma)
Projenin yerel makinede çalıştırılabilmesi için gereken adımlar aşağıdadır:

Gerekli Dosyalar: app.py, requirements.txt ve geliştirme sürecini içeren .ipynb dosyaları repoda mevcuttur.

Bağımlılıkların Yüklenmesi:

Bash

pip install -r requirements.txt
(Opsiyonel) Hugging Face Token: Daha yüksek API limitleri için Hugging Face hesabınızdan bir API Token oluşturup ortam değişkeni olarak ayarlayabilirsiniz:

Bash

export HF_TOKEN="SİZİN_HF_TOKENINIZ"
Uygulamanın Başlatılması:

Bash

streamlit run app.py
6. 🔗 Web Arayüzü Linki (Product Kılavuzu)
Projenin Streamlit Cloud'da çalışan kalıcı web linki aşağıdadır. Uygulamayı test etmek için bu linki kullanın.

Kalıcı Web URL'si:
[BURAYA STREAMLIT CLOUD'DAN ALDIĞINIZ ÇALIŞAN LİNKİ YAPIŞTIRIN]

Test Kılavuzu: Uygulamaya girdikten sonra, "peynir ve domatesle yapılan kolay bir tarif" veya "bir kokteylin adımları nelerdir" gibi sorular sorarak chatbot'u test edebilirsiniz. Yanıtla birlikte kullanılan kaynak tarifler de gösterilecektir.
