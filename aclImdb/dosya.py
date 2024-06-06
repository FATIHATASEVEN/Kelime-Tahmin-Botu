import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Veri setini yükleme ve hazırlama
directory = "C:\\Users\\tr\\Desktop\\aclImdb\\test\\pos"
files = os.listdir(directory)
texts = []
drama_texts = []
action_texts = []
science_texts = []
love_texts = []
fear_texts = []
other_texts = []

for file_name in files:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        texts.append(content)
        if 'drama' in content.lower():
            drama_texts.append(content)
        elif 'action' in content.lower():
            action_texts.append(content)
        elif 'science' in content.lower():
            science_texts.append(content)
        elif 'love' in content.lower():
            love_texts.append(content)
        elif 'fear' in content.lower():
            fear_texts.append(content)
        else:
            other_texts.append(content)

labels = ["positive"] * len(texts)  # Tüm dosyaların etiketi "positive"

# Toplam dosya sayısını yazdırma
total_files = len(files)
print("Toplam dosya sayısı:", total_files)

# Kategorilere göre dosya sayısını yazdırma
num_drama = len(drama_texts)
num_action = len(action_texts)
num_science = len(science_texts)
num_love = len(love_texts)
num_fear = len(fear_texts)
num_other = len(other_texts)
print("Drama filmi sayısı:", num_drama)
print("Aksiyon filmi sayısı:", num_action)
print("Bilim kurgu filmi sayısı:", num_science)
print("Aşk filmi sayısı:", num_love)
print("Korku filmi sayısı:", num_fear)
print("Diğer filmler sayısı:", num_other)

# Toplam kelime sayısı ve 'and' kelimesinin sıklığı
total_word_count = sum(len(text.split()) for text in texts)
print("Veri setinde toplam kelime sayısı:", total_word_count)

total_and_count = sum(text.count(' and ') for text in texts)
print("Veri setinde 'and' kelimesinin kullanım sıklığı:", total_and_count)

# En çok ve en az kullanılan kelimeleri bulma fonksiyonu
def most_and_least_common_words(texts):
    all_text = ' '.join(texts)
    words = all_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    least_common_words = word_counts.most_common()[:-11:-1]
    return most_common_words, least_common_words

# En çok ve en az kullanılan kelimeleri bulma
most_common, least_common = most_and_least_common_words(texts)
print("En çok kullanılan ilk 10 kelime:", most_common)
print("En az kullanılan ilk 10 kelime:", least_common)

# En çok kullanılan harfi bulma fonksiyonu
def most_common_letter(texts):
    all_text = ''.join(texts)
    letters = [char for char in all_text if char.isalpha()]
    letter_counts = Counter(letters)
    most_common_letter, _ = letter_counts.most_common(1)[0]
    return most_common_letter

# En çok kullanılan harfi bulma
most_common_letter = most_common_letter(texts)
print("Veri setinde en çok kullanılan harf:", most_common_letter)

# DataFrame oluşturma
data = pd.DataFrame({'text': texts, 'label': labels})

# Metin ve etiket sütunlarını seçme
texts = data['text'].values
labels = data['label'].values

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Metin veri setini eğitim ve test setlerine ayırma
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

# Metin verilerini vektörlere dönüştürmek için Tokenizer kullanın
max_words = 10000  # Kelime sayısı sınırı artırıldı
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

# Metin verilerini vektörlere dönüştürün
X_train = tokenizer.texts_to_sequences(train_texts)
max_length = 200  # Belge uzunluğu sınırı artırıldı
X_train = pad_sequences(X_train, maxlen=max_length)

# Modeli oluşturun
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(max_words, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin ve eğitim geçmişini alın
history = model.fit(X_train, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Eğitim ve doğruluk kayıplarını alın
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Grafiği çizin
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test verilerini vektörlere dönüştürün
X_test = tokenizer.texts_to_sequences(test_texts)
X_test = pad_sequences(X_test, maxlen=max_length)

# Modelin doğruluğunu test edin
accuracy = model.evaluate(X_test, test_labels)[1]
print("Model Accuracy:", accuracy)

# Metin üretim fonksiyonu
def generate_text(seed_text, next_words, max_sequence_len, temperature=1.0):
    result = seed_text
    all_words = []  # Tüm kelimeleri saklamak için
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Olasılıkları sıcaklığa göre ölçekleme
        predicted_probs = np.asarray(predicted_probs).astype('float64')
        predicted_probs = np.log(predicted_probs + 1e-7) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Rastgele bir kelime seçimi
        predicted_word_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        
        result += " " + output_word
        all_words.append(output_word)
    return result, all_words

# 7-8 kelimelik yorum üretme
seed_text = "The movie was"
next_words = 7
generated_text, all_words = generate_text(seed_text, next_words, max_length, temperature=0.7)
print("Generated Text:", generated_text)

# Tüm kelimelerin göründüğü kümeyi ve sayımını ekrana yazdırma
word_counts = Counter(all_words)
print("Generated Words and Their Counts:", word_counts)

# Yeni yorum dosyasını oluşturma
new_file_path = os.path.join(directory, "generated_review.txt")
with open(new_file_path, 'w', encoding='utf-8') as file:
    file.write(generated_text)
print(f"Generated text saved to {new_file_path}")

# Yeni yorumun kategorisine göre ekleme
if 'drama' in generated_text.lower():
    drama_texts.append(generated_text)
elif 'action' in generated_text.lower():
    action_texts.append(generated_text)
elif 'science' in generated_text.lower():
    science_texts.append(generated_text)
elif 'love' in generated_text.lower():
    love_texts.append(generated_text)
elif 'fear' in generated_text.lower():
    fear_texts.append(generated_text)
else:
    other_texts.append(generated_text)

# Yeni kategorilere göre dosya sayısını yazdırma
num_drama = len(drama_texts)
num_action = len(action_texts)
num_science = len(science_texts)
num_love = len(love_texts)
num_fear = len(fear_texts)
num_other = len(other_texts)
print("Drama filmi sayısı (güncellenmiş):", num_drama)
print("Aksiyon filmi sayısı (güncellenmiş):", num_action)
print("Bilim kurgu filmi sayısı (güncellenmiş):", num_science)
print("Aşk filmi sayısı (güncellenmiş):", num_love)
print("Korku filmi sayısı (güncellenmiş):", num_fear)
print("Diğer filmler sayısı (güncellenmiş):", num_other)
