import nltk
#nltk.download('punkt_tab')

text = "Natural Language Processing is a branch of artificial intelligence."

from nltk.tokenize import word_tokenize #tokenize kelimelere ya da cümlelere ya da anlamlı küçük parçalara ayırır 

tokens = word_tokenize(text) #kelimelere ayırır
print(tokens)
from nltk.corpus import stopwords
#nltk.download('stopwords') #Stopword, bir dilde çok sık geçen ama çoğu zaman analizde anlamsal katkısı az olan kelimelerdir Burada önemli kelimeler: yazı, paylaşmak
#Stopword olanlar: bu, seninle, istedim

stop_words = set(stopwords.words('english')) #dosyadaki kelimeleri okur #set->Bir liste verirsen içindeki tekrar edenleri kaldırıp küme (set) yapısı oluşturur.
filtered_tokens = [word for word in tokens if word not in stop_words]
""" The following code creates a  filtered list of tokens by removing stop words. parantez içindekinin uzun hali
filtered = []
 for word in tokens:
    if word not in stop_words:
         filtered.append(word)"""

print(filtered_tokens)

#Lemmatization: kelimelerin temel formunu bulur. kök haline getirir.
#running -> run
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running" , pos="v")) #part of speech v: verb, n: noun, a: adjective, r: adverb o an lemmetize ettiğimiz kelimenin türünü belirler 

#metin sınıflandırma işlemi için arama motorunda çokça kullanılır
#nltk.download('averaged_perceptron_tagger_eng') # average per.tagger->Bu, cümledeki her kelimenin isim mi, fiil mi, sıfat mı olduğunu tahmin eden bir makine öğrenmesi algoritmasıdır.
from nltk import pos_tag 
pos_tags= pos_tag(filtered_tokens)
print(pos_tags)
#ı am running derken [('I', 'PRP'), ('am', 'VBP'), ('running', 'VBG')]  şeklinde sınıflar kelimeleri average perception tagger

#nltk.download('maxent_ne_chunker_tab')
#Bu, NLTK kütüphanesinin içinde bulunan ve özel adları (kişiler, yerler, kuruluşlar gibi) metin içinde tanımaya yarayan makine öğrenmesi tabanlı bir modeldir.
#Açılımı:
#maxent = Maximum Entropy (bir istatistiksel model türü)
#ne = Named Entity (özel ad)
#chunker = Cümle parçalama/etiketleme aracı
#tab = Yeni versiyon, etiketleri bir tablo modeli ile verir (eski chunker'a göre daha modern)


#📦 NLP’de MaxEnt Ne İşe Yarar?
#Doğal dil işleme problemlerinde şunu yapar:
#Bir kelimenin ne tür bir varlık (entity) olduğunu tahmin eder:
#PERSON, LOCATION, ORGANIZATION, vs.
#Model, cümle içinde kelimenin kendisine ve bağlamına bakar ve bir olasılık dağılımını tahmin eder.


nltk.download('words')

from nltk import ne_chunk #named entity recognition
tree = ne_chunk(pos_tags)
print(tree)

#Metin temizleme ve ön işleme 
#Lowercading: büyük harfleri küçük harfe çevirir.

text = "Natural Language Processing is a branch of artificial intelligence. 100%"
text = text.lower()
print(text)
 #noktalama işaretlerini kaldırır
import re
text = re.sub(r'[^\w\s]', '', text) #\w: kelime karakteri, \s: boşluk karakteri, ^: başlangıç, $: bitiş re->regular expression
print(text)
#sayıları kaldırır.
text = re.sub(r'\d+', '', text) #\d: sayı karakteri, +: bir veya daha fazla sayı karakteri
print(text)

#vectorization: metinleri sayısal değerlere çevirir.
corpus = ["I love programming", "I love natural language processing", "I love artificial intelligence", "I love python"]
#bag of words modeli vektörize modelidir
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer, kelimelerin kaç kez geçtiğini sayan bir vektör üreticidir.
#Metindeki her kelimeyi bir özellik (feature) olarak kabul eder.
#Örneğin, "I love programming" metninde "I", "love", "programming" kelimeleri özellikler olarak kabul edilir.
#Bu özelliklerin sayısal değerleri, metinlerin sayısal temsillerini oluşturur.

vectorizer = CountVectorizer()
#vectorizer adında bir nesne oluşturuyorsun.
#Henüz metne uygulamadın, sadece aracı tanımladın.

X = vectorizer.fit_transform(corpus)
#fit_transform şunu yapar:
#fit: Tüm kelimeleri öğrenir (sözlük oluşturur).
#transform: Her cümleyi bir kelime sayısı vektörüne çevirir.
print(vectorizer.get_feature_names_out()) #kelimeleri görüntüler
print(X.toarray())
# X bir sparse matrix (seyrek matris) nesnesidir.
# toarray() onu tam görünümlü bir 2D NumPy dizisine çevirir.
# Her satır = bir cümle
# Her sütun = bir kelime
# Her hücre = o kelimenin o cümlede kaç kez geçtiği
print("--------------------------------------------")
#tf-idf modeli vektörize modelidir term frequency-inverse document frequency modelidir
#tf: bir kelimenin bir cümlede kaç kez geçtiği
#idf: bir kelimenin kaç cümlede geçtiği
#tf-idf: bir kelimenin bir cümlede kaç kez geçtiği ve kaç cümlede geçtiği, nadirliği
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out()) #kelimeleri görüntüler
print(X2.toarray())
#değeri bie yakınsa o kelime o cümlede çok önemli demektir.
#değeri 0 ise o kelime o cümlede hiç geçmemiş demektir.


#N-gram modeli: bir kelimenin bir cümlede kaç kez geçtiği
"""| N-Gram Türü | Örnek             | Amaç                        |
| ----------- | ----------------- | --------------------------- |
| Unigram     | `"love"`          | Kelime frekansına bakar     |
| Bigram      | `"love Python"`   | Bağlam bilgisi ekler        |
| Trigram     | `"I love Python"` | Daha detaylı bağlam bilgisi |"""


# Otomatik tamamlama, spam tespiti, yazım önerisi.
# Nerede kullanılır? => Dilin anlamını anlamaz. Sadece istatistiksel olarak kullanılır.

# Apple is a fruit.
# Apple is a company.

corpus = [
    "NLP çok eğlenceli alan",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz"
]

from sklearn.feature_extraction.text import TfidfVectorizer
# Unigram ve Bigram birlikte kullanılır (1,2) derken
vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True) 
#tfid kullandık çünkü vectörize ettik unigram ve bigramı makine sayıları anlar sadece ondan
X = vectorizer.fit_transform(corpus)

print(f"Feature Names: {vectorizer.get_feature_names_out()}")
print(f"X: {X.toarray()}")



# Word Embedding
# Her kelimeye sayısal bir vektör ata. Bu vektörler sayesinde:
# Kelimeler arasındaki anlamsal yakınlık öğreniliyor.
# Aynı bağlam geçen kelimeler, uzayda da birbirine yakın olur.
# Araba -> [0.21, -0.43, 0.92, ........, 0.01] 100 veya 300+ boyutlu.

# Güzel ek özellik => Vektör cebiri bile yapılabilir.
# vec("king") - vec("man") + vec("woman") = vec("queen")

# Nerede kullanılır? 

# Derin öğrenme.
# Chatbot, anlamsal arama

corpus = [
    "NLP çok eğlenceli alan",
    "Doğal dil işleme çok önemli",
    "Eğlenceli projeler yapıyoruz"
]
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

#her cümleyi tokenize et. kelime listesi oluştur.
# kelimeleri parçala, liste haline getir.
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]
print("******")
print(tokenized_sentences)

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)
"""vector_size=100: Her kelime 100 boyutlu bir vektörle temsil edilir.

window=5: Model, her kelimenin çevresindeki 5 kelimeyi dikkate alarak bağlam öğrenir.

min_count=1: En az 1 kez geçen kelimeler modele dahil edilir.

workers=2: 2 CPU çekirdeği kullanılarak model paralel olarak eğitilir (hızlı eğitim).

📌 Bu satır modelin kelime vektörlerini öğrenmesini sağlar."""
print("*******")
print(model.wv['nlp'])
print("*******")
print(model.wv.most_similar('nlp'))
""" | Terim                     | Anlamı                                                  |
| ------------------------- | ------------------------------------------------------- |
| `model`                   -> Word2Vec eğitimi yapılan modelin tamamı                 |
| `model.wv`                -> Sadece kelime vektörlerine erişim için kullanılan parça |
| `model.wv['kelime']`      -> O kelimenin sayısal temsili (embedding)                 |
| `model.wv.most_similar()` -> En benzer kelimeleri bulur                              |"""
#Word2Vec, kelimeleri sayılara (vektörlere) dönüştüren bir yapay zeka modelidir. Ama rastgele değil — bu sayılar, kelimelerin anlamlarını yansıtır!



# Sentence Embedding


corpus = [
    "NLP çok eğlenceli alan",
    "NLP çok önemli",
    "Eğlenceli projeler yapıyoruz"
]
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in corpus]

model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)

# Ortalama Vektör Alma
import numpy as np

def sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    return np.zeros(100)

vec1 = sentence_vector(corpus[0]) #corpustaki ilk cümlesinin 100 boyutlu ortalama vektörü.
vec2 = sentence_vector(corpus[1]) #corpustaki ikinci cümlesinin 100 boyutlu ortalama vektörü.
"""******************************************************
Çıktılar -0.1 ile +0.1 civarında ondalıklı 100 sayılık diziler gibi görünür.
Bu vektörler artık:
İki cümle arasındaki benzerliği (ör. kosinüs benzerliği) hesaplamak,
Kümeleme / sınıflandırma yapmak,
Görselleştirme (t-SNE, PCA) oluşturmak
gibi işlemlerde kullanılabilir.
***************************************************"""
print(vec1)
print(vec2)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# cosinesimilarty(a,b) = a.b / |a| * |b| => -1,1 arasında değer döner. nor
#Bu, vec1 ile vec2 arasındaki açısal benzerliği hesaplar. cosü verir iki vektörün aralarındaki açı yani
#np.dot(vec1, vec2)
#İki vektörün iç çarpımı (dot product) alınır.
"""
İki vektör paralel ve aynı yöndeyse → 1 anlamsal benzerlik var
Ters yöndeyse → -1 anlam tam tersi
Birbirine dikse → 0 anlamsal benzerlik yok
değerini döndürür.  cümle benzerliğini çok kullanırız
"""

# Average Word Embedding

print(cosine_similarity(vec1, vec2))

# Sentence-BERT (SBERT) 
# Bu dosya üzerinde uygulayalım.
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
#all-MiniLM-L6-v2: Hafif, hızlı ve güçlü bir SBERT modeli
cümleler = [
    "NLP çok eğlenceli alan",
    "NLP çok önemli",
    "Eğlenceli projeler yapıyoruz"
]

embeddings = model.encode(cümleler, convert_to_tensor=False)

print(embeddings)
"""from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # Hafif, hızlı ve güçlü bir SBERT modeli

# Cümleleri tanımla
sentence1 = "NLP çok eğlenceli alan"
sentence2 = "NLP çok önemli"

# Vektörlere dönüştür
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

 Kosinüs benzerliği hesapla
similarity = util.pytorch_cos_sim(embedding1, embedding2)
print("Benzerlik:", similarity.item())"""
