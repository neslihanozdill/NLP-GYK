import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK verilerini indir 
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

class NLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def preprocess_text(self, text):
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        
        # Önce stop words'leri filtrele
        filtered_tokens = [token for token in tokens if token not in self.stop_words and token.isalpha()]
        
        # NLTK lemmatizer kullan
        nltk_lemmatized = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Basit lemmatization da uygula
        basic_lemmatized = self.basic_lemmatize(filtered_tokens)
        
        # Her iki yöntemi birleştir (NLTK + basit kurallar)
        final_tokens = []
        for i, token in enumerate(filtered_tokens):
            # NLTK lemmatizer sonucu
            nltk_result = nltk_lemmatized[i]
            # Basit lemmatization sonucu
            basic_result = basic_lemmatized[i]
            
            # Daha kısa olanı tercih et (genellikle daha iyi lemmatization)
            if len(basic_result) < len(nltk_result):
                final_tokens.append(basic_result)
            else:
                final_tokens.append(nltk_result)
        
        return " ".join(final_tokens)

    def basic_lemmatize(self, tokens):
        
        lemmatized = []
        for token in tokens:
            if token.endswith('ing'):
                token = token[:-3]
           
            elif token.endswith('ed'):
                token = token[:-2]
            
            elif token.endswith('s') and len(token) > 3:
                token = token[:-1]
            lemmatized.append(token)
        return lemmatized

    def fit_transform(self, text_list):
        processed_texts = [self.preprocess_text(text) for text in text_list]
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        return tfidf_matrix

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def show_results(self, tfidf_matrix):
        print("\n🔍 TF-IDF Feature Names:")
        feature_names = self.get_feature_names()
        for i, name in enumerate(feature_names):
            print(f"  {i+1}. {name}")
        
        print(f"\n📊 TF-IDF Matrix Shape: {tfidf_matrix.shape}")
        print("\n📈 TF-IDF Array:")
        print(tfidf_matrix.toarray())
        
        # İstatistikler
        print(f"\n📋 Toplam özellik sayısı: {len(feature_names)}")
        print(f"📄 İşlenen metin sayısı: {tfidf_matrix.shape[0]}")

# 📄 Metinleri .txt dosyasından oku
def read_texts_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
    return lines

# 🚀 Ana çalışma
if __name__ == "__main__":
    # Örnek metinler (dosya yoksa bunları kullan)
    sample_texts = [
        "This is the first sample text for NLP preprocessing.",
        "Natural language processing is a fascinating field of study.",
        "Machine learning and artificial intelligence are transforming the world.",
        "Text analysis and sentiment analysis are important applications.",
        "Python is a powerful programming language for data science."
    ]
    
    try:
        import os
        # Dosyanın tam yolunu al
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "texts.txt")
        texts = read_texts_from_file(file_path)
        print("📄 Metinler texts.txt dosyasından okundu.")
    except FileNotFoundError:
        texts = sample_texts
        print("📄 texts.txt dosyası bulunamadı, örnek metinler kullanılıyor.")

    pipeline = NLPPreprocessor()
    tfidf_result = pipeline.fit_transform(texts)
    pipeline.show_results(tfidf_result)
