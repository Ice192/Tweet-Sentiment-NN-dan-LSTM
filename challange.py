import io
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flasgger import Swagger
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
swagger = Swagger(app, template_file='swagger.yml')

# Membaca file singkatan menggunakan pandas tanpa header dengan encoding latin1
singkatan_df = pd.read_csv('data/new_kamusalay.csv', header=None, encoding='latin1')
singkatan_df.columns = ['singkatan', 'kepanjangan']

# Membuat dictionary singkatan
singkatan_dict = dict(zip(singkatan_df['singkatan'], singkatan_df['kepanjangan']))

# Initialize the stemmer and stopword remover outside the function for efficiency
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()


# Fungsi untuk membersihkan teks dari karakter yang di-encode, escape sequences, dan karakter Unicode
def clean_text(text):
    
    # Menghapus karakter yang di-encode dengan format \xHH
    pattern_encoded = re.compile(r'\\x[0-9A-Fa-f]{2}')
    text = pattern_encoded.sub('', text)
    
    # Menghapus escape sequences seperti \n, \t, dll.
    pattern_escape = re.compile(r'\\n|\\t')
    text = pattern_escape.sub(' ', text)
    
    # Membersihkan karakter Unicode
    text_cleaned = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')

     # Konversi teks menjadi lowercase
    text_cleaned = text_cleaned.lower()
    
    return text_cleaned.strip()

def stop_word(text):
    # Remove stopwords using Sastrawi
    text = stopword_remover.remove(text)
    
    # Stem the words using Sastrawi
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    
    return text


# Fungsi untuk mengganti singkatan
def replace_singkatan(text, singkatan_dict):
    words = text.split()
    for i in range(len(words)):
        word_lower = words[i].lower()
        if word_lower in singkatan_dict:
            words[i] = singkatan_dict[word_lower]
    return ' '.join(words)

# Load the pre-trained model and vectorizer
model = pickle.load(open('model_nn.pickle', 'rb'))
vectorizer = pickle.load(open('feature_nn.pickle', 'rb'))

# Definisikan parameter pada Feature Extraction dan Tokenizer
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=" ", lower=True)

# Definisikan Label untuk sentiment
sentiment = ['negative','neutral','positive']

# Load Hasil Feature Extraction
file = open("x_pad_sequences_Lstm.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

# Load Model Lstm
model_file_from_lstm = load_model('model_Lstm.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tweet_text_nn', methods=['GET', 'POST'])
def tweet_form_nn():
    if request.method == 'POST':
        tweet = request.form.get('tweet')
        
        # Simpan original tweet
        original_tweet = tweet
        
        # Bersihkan teks dan konversi ke lowercase
        tweet = clean_text(tweet)
        
        # Mengganti singkatan
        tweet = replace_singkatan(tweet, singkatan_dict)
    
        # Menghapus kata-kata seperti 'dan' 'yang' dan lain-lain
        tweet = stop_word(tweet)

         # Vectorize tweet yang sudah diproses
        tweet_vectorized = vectorizer.transform([tweet])
        
        # Prediksi sentimen menggunakan model yang telah dilatih
        prediction = model.predict(tweet_vectorized)

         # Map prediksi ke label
        predicted_sentiment = sentiment[prediction[0]]
        
        
        return render_template('tweet_form_nn.html', original_tweet=original_tweet, predicted_sentiment=predicted_sentiment)
    
    return render_template('tweet_form_nn.html')

@app.route('/tweet_text_lstm', methods=['GET', 'POST'])
def tweet_form_lstm():
    if request.method == 'POST':
        tweet = request.form.get('tweet')
        
        # Simpan original tweet
        original_tweet = tweet
        
        # Bersihkan teks dan konversi ke lowercase
        tweet = clean_text(tweet)
        
        # Mengganti singkatan
        tweet = replace_singkatan(tweet, singkatan_dict)
    
        # Menghapus kata-kata seperti 'dan' 'yang' dan lain-lain
        tweet = stop_word(tweet)

        # Masukkan Feature
        feature = tokenizer.texts_to_sequences(tweet)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

        # Prediksi
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        
        return render_template('tweet_form_lstm.html', original_tweet=original_tweet, get_sentiment=get_sentiment)
    
    return render_template('tweet_form_lstm.html')

@app.route('/upload_process_nn', methods=['GET', 'POST'])
def upload_process_nn():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Membaca file CSV yang diunggah
            tweets_df = pd.read_csv(file, encoding='latin1')
            
            # Inisialisasi list untuk hasil proses
            original_tweets = []
            cleaned_tweets = []
            predicted_sentiments = []
            
            # Proses setiap tweet dalam file
            for tweet in tweets_df['Tweet']:
                original_tweet = tweet

                # Menghilangkan text non-ASCII (seperti emoji)
                tweet = clean_text(tweet)

                # Mengganti singkatan
                tweet_cleaned = replace_singkatan(tweet, singkatan_dict)

                # Stopword
                tweet_vectorize = stop_word(tweet_cleaned)

                tweet_vectorize = vectorizer.transform([tweet_vectorize])
        
                # Prediksi sentimen menggunakan model yang telah dilatih
                prediction = model.predict(tweet_vectorize)

                # Map prediksi ke label
                predicted_sentiments = sentiment[prediction[0]]
                     
                # Tambahkan hasil ke list
                original_tweets.append(original_tweet)
                cleaned_tweets.append(tweet_cleaned)
                
            # Buat DataFrame baru untuk hasilnya
            result_df = pd.DataFrame({
                'original_tweet': original_tweets,
                'cleaned_tweet': cleaned_tweets,
                'predicted_sentiments': predicted_sentiments
            })
            
            # Simpan DataFrame ke CSV dalam memory
            output = io.BytesIO()
            result_df.to_csv(output, index=False)
            output.seek(0)
            
            # Kembalikan file CSV untuk didownload
            return send_file(output, mimetype='text/csv', download_name='predicted_tweets.csv', as_attachment=True)
        
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    
    return render_template('upload_file_nn.html')

@app.route('/upload_process_lstm', methods=['GET', 'POST'])
def upload_process_lstm():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Membaca file CSV yang diunggah
            tweets_df = pd.read_csv(file, encoding='latin1')
            
            # Inisialisasi list untuk hasil proses
            original_tweets = []
            cleaned_tweets = []
            predicted_sentiments = []
            
            # Proses setiap tweet dalam file
            for tweet in tweets_df['Tweet']:
                original_tweet = tweet

                # Menghilangkan text non-ASCII (seperti emoji)
                tweet = clean_text(tweet)

                # Mengganti singkatan
                tweet_cleaned = replace_singkatan(tweet, singkatan_dict)

                # Stopword
                tweet_feature = stop_word(tweet_cleaned)

                # Masukkan Feature
                feature = tokenizer.texts_to_sequences(tweet_feature)
                feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

                # Prediksi
                prediction = model_file_from_lstm.predict(feature)
                get_sentiment = sentiment[np.argmax(prediction[0])]
                                
                # Tambahkan hasil ke list
                original_tweets.append(original_tweet)
                cleaned_tweets.append(tweet_cleaned)
                predicted_sentiments.append(get_sentiment)
            
            # Buat DataFrame baru untuk hasilnya
            result_df = pd.DataFrame({
                'original_tweet': original_tweets,
                'cleaned_tweet': cleaned_tweets,
                'sentiment': predicted_sentiments
            })
            
            # Simpan DataFrame ke CSV dalam memory
            output = io.BytesIO()
            result_df.to_csv(output, index=False)
            output.seek(0)
            
            # Kembalikan file CSV untuk didownload
            return send_file(output, mimetype='text/csv', download_name='Sentiment_tweets_LSTM.csv', as_attachment=True)
        
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    
    return render_template('upload_file_lstm.html')

if __name__ == '__main__':
    app.run(debug=True)
