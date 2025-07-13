import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split


csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)


drop_cols = [
    'id', 'author', 'subreddit', 'link_id', 'parent_id',
    'created_utc', 'rater_id', 'example_very_unclear'
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


df['clean_text'] = df['text'].apply(clean_text)
df.drop(columns=['text'], inplace=True)


X = df['clean_text']
y = df.drop(columns=['clean_text'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Tokenization ve Padding

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max([len(seq) for seq in X_train_seq])

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Embedding

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

embedding_dim = 100  # İstersen 50, 128, 300 yapabilirsin

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(28, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Eğitim

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    callbacks=[early_stop]
)

 # Değerlendirme
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred_binary, target_names=y.columns.tolist()))
y_pred = model.predict(X_test_pad)
y_pred_binary = (y_pred > 0.3).astype(int)

print("Classification Report:")

import pickle

model.save("model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
