
from tensorflow.keras.models import load_model
import pickle

model = load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)



import pandas as pd

csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
drop_cols = [
    'id', 'author', 'subreddit', 'link_id', 'parent_id',
    'created_utc', 'rater_id', 'example_very_unclear'
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

import re
import string
from ftfy import fix_text

def clean_text(text):
    text = fix_text(text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    return text.translate(str.maketrans("", "", string.punctuation))

df["processed"] = df["text"].apply(clean_text)
df.drop(columns=["text"], inplace=True)

from sklearn.model_selection import train_test_split

X = df["processed"]
y = df.drop(columns=["processed"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from tensorflow.keras.preprocessing.sequence import pad_sequences

X_test_seq = tokenizer.texts_to_sequences(X_test)
max_len = max(len(seq) for seq in tokenizer.texts_to_sequences(X_train))
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")



y_prob = model.predict(X_test_pad)
labels = y_test.columns.tolist()

from sklearn.metrics import f1_score, classification_report
import numpy as np

def find_optimal_thresholds(y_true, y_scores):
    best_thresholds = []
    for i in range(y_true.shape[1]):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            preds = (y_scores[:, i] >= t).astype(int)
            score = f1_score(y_true.iloc[:, i], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        best_thresholds.append(best_t)
    return np.array(best_thresholds)

thresholds = find_optimal_thresholds(y_test, y_prob)
y_pred_bin = (y_prob >= thresholds).astype(int)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_bin, target_names=labels))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10,5))
plt.bar(range(len(thresholds)), thresholds)
plt.xticks(range(len(thresholds)), labels, rotation=90)
plt.title("Etiket Başına Optimal Eşik Değerleri")
plt.ylabel("Eşik")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,10))
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("1 - Özgüllük")
plt.ylabel("Hassasiyet Değeri")
plt.title("Her Duygu İçin ROC Eğrileri")
plt.legend(loc="lower right", fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import re
import string
from ftfy import fix_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_emotions(text, model, tokenizer, thresholds, max_len, labels):
    def clean(text):
        text = fix_text(text)
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        return text.translate(str.maketrans('', '', string.punctuation)).strip()

    cleaned_text = clean(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    proba = model.predict(padded_seq)[0]
    predicted_labels = [label for label, p, t in zip(labels, proba, thresholds) if p >= t]

    print("📝 Input Text:", text)
    print("🎯 Detected Emotions:", predicted_labels)
    return {
        "predictions": predicted_labels,
        "probabilities": dict(zip(labels, proba))
    }


example = "This was the best experience I've ever had. It truly brought me so much joy!"
result = predict_emotions(example, model, tokenizer, thresholds, max_len, labels)
