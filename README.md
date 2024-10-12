# Tweet Sentiment Analysis with Neural Networks and LSTM

This project is a Flask web application that performs sentiment analysis on tweets using two different models: a simple neural network and an LSTM (Long Short-Term Memory) model. The application provides functionality for both text input analysis and batch file processing via CSV upload.

## Features

- **Text Cleaning**: The app includes functions to clean tweets, such as removing encoded characters, escape sequences, and Unicode characters.
- **Abbreviation Replacement**: Abbreviations are replaced using a custom dictionary of abbreviations mapped to their full form.
- **Stopword Removal and Stemming**: Utilizes the Sastrawi library for removing stopwords and stemming Indonesian words.
- **Sentiment Prediction**: Supports sentiment prediction using:
  - A pre-trained neural network (NN) model
  - A pre-trained LSTM model

## Endpoints

1. `/`: Home page, renders the index.
2. `/tweet_text_nn`: Input a tweet and predict the sentiment using the neural network model.
3. `/tweet_text_lstm`: Input a tweet and predict the sentiment using the LSTM model.
4. `/upload_process_nn`: Upload a CSV file containing tweets, clean them, and predict their sentiment using the neural network model.
5. `/upload_process_lstm`: Upload a CSV file containing tweets, clean them, and predict their sentiment using the LSTM model.

## Libraries Used

- **Flask**: For creating the web application.
- **Pandas**: For reading and processing CSV files.
- **Sastrawi**: For stopword removal and stemming (specifically for Indonesian text).
- **Flasgger**: For Swagger API documentation.
- **TensorFlow/Keras**: For loading and using the neural network and LSTM models.
- **Numpy**: For handling numerical operations.
- **Pickle**: For loading pre-trained models and feature vectors.
