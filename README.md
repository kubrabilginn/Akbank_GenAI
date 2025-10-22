🍽️ Akbank GenAI 101 Bootcamp Projesi: RAG Tabanlı Yemek Tarifleri Chatbotu
1. Giriş ve Proje Amacı
Bu proje, RAG (Retrieval Augmented Generation) mimarisi kullanılarak geliştirilmiş, kullanıcıların spesifik yemek tarifleri hakkında soru sorabileceği interaktif bir chatbot sunmayı amaçlamaktadır. Proje, Büyük Dil Modellerinin (LLM) bilgi sınırlaması sorununa çözüm getirerek, harici bir bilgi tabanına (yemek tarifleri) güvenli bir şekilde erişmesini sağlamaktadır.

2. Çözüm Mimarisi ve Kullanılan Yöntemler
Projenin teknik omurgası, LLM'nin yanıtını harici kaynaklardan gelen verilerle desteklediği modern bir RAG zincirine dayanmaktadır.

2.1. Mimarinin Ana Bileşenleri:

Orkestrasyon Çatısı: LangChain (RAG zincirini yönetmek için kullanıldı).

Üretim Modeli (LLM): Google Gemini 2.5 Flash (Kullanıcı sorgusuna akıcı yanıt üretir).

Gömme Modeli (Embedding): Google GenerativeAIEmbeddings (text-embedding-004) (Metinleri vektörlere dönüştürür).

Vektör Veritabanı (Vector DB): ChromaDB (Vektör temsillerini depolar ve aranmasını sağlar).

Web Arayüzü: Streamlit (Kullanıcı etkileşimini sağlar).

2.2. RAG İş Akışı:
Geri Getirme (Retrieval): Kullanıcı sorusu, gömme modeli ile vektöre dönüştürülür ve ChromaDB'de en alakalı kaynak belgeler (tarifler) geri getirilir.

Destekleme (Augmentation): Geri getirilen tarifler, özel bir Prompt Template içine yerleştirilerek kullanıcının sorusuyla birlikte LLM'ye sunulur.

Üretim (Generation): Gemini 2.5 Flash, sadece kendisine sunulan bağlamı (tarifleri) kullanarak nihai yanıtı üretir ve halüsinasyonları engeller.

3. Veri Seti Hakkında Bilgi

Veri Kaynağı: Projede, Hugging Face Datasets platformundan elde edilen Hieu-Pham/kaggle_food_recipes veri seti kullanılmıştır.

Veri Seti Hazırlığı: Prototipleme ve hız için veri setinin tamamı yerine, ilk 200 adet tarifi içeren küçük bir alt küme kullanılmıştır. Tariflerin Title, Ingredients ve Instructions sütunları birleştirilerek RAG için işlenmeye hazır hale getirilmiştir.

4. Elde Edilen Sonuçlar
Chatbot, malzeme, yemek türü veya pişirme adımları bazlı sorgulara, kendi veri tabanından çektiği doğru ve ilgili tarif metinlerini kullanarak başarılı yanıtlar vermektedir.

Uygulama, sürekli dağıtım (Streamlit Cloud) sayesinde kalıcı bir web linki üzerinden erişilebilirdir.

Kullanıcılar, yanıtla birlikte kullanılan kaynak tariflerin adlarını görerek bilginin güvenilirliğini teyit edebilir.

5. Kodun Çalışma Kılavuzu (Local Çalıştırma)
Projenin yerel makinede çalıştırılabilmesi için gereken adımlar aşağıdadır:

Gerekli Dosyalar: app.py, requirements.txt ve geliştirme sürecini içeren .ipynb dosyaları repoda mevcuttur.

Bağımlılıkların Yüklenmesi:

Bash

```pip install -r requirements.txt```

API Anahtarının Tanımlanması: Gemini API Anahtarınızı ortam değişkeni olarak ayarlayın:

Bash

```export GEMINI_API_KEY="SİZİN_ANAHTARINIZ"```
Uygulamanın Başlatılması:

Bash

```streamlit run app.py```

6. 🔗 Web Arayüzü Linki (Product Kılavuzu)
Projenin Streamlit Cloud'da çalışan kalıcı web linki aşağıdadır. Uygulamayı test etmek için bu linki kullanın.

Kalıcı Web URL'si:
[BURAYA STREAMLIT CLOUD'DAN ALDIĞINIZ ÇALIŞAN LİNKİ YAPIŞTIRIN] 


NOT: Web arayüzüne girdikten sonra sorgu alanına malzeme (örneğin: peynir ve domatesle yapılan kolay bir tarif) veya tarif adı yazarak chatbot'u test edebilirsiniz.
