#RNN model - LSTM (long short term memory)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding

# Read the dataset
# df2 = pd.read_csv(r"D:\SLIIT\Year 4\Semester 1\Research\ChatBot\Codes\SentimentAnalysis_Training\Data\test.csv")
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the CSV file
relative_path = "Data/test.csv"
csv_path = os.path.join(script_directory, relative_path)

# Read the CSV file using the relative path
df2 = pd.read_csv(csv_path)

# Filter out neutral category
df_all = df2[df2['category'] != 'neutral']

print(df_all.shape)
df_all.head(5)

# Convert category labels to numerical values
sentiment_label = df_all.category.factorize()

df_all['clean_text'] = df_all['clean_text'].astype(str)  # Ensure all values are strings
df_all['clean_text'] = df_all['clean_text'].fillna('')  # Handle missing values by replacing them with empty string

# Preprocess text data
text = df_all.clean_text.values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text)
encoded_docs = tokenizer.texts_to_sequences(text)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequence, sentiment_label[0], test_size=0.2, random_state=42)

# Define the model architecture
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

def fitmodel():
    # Compile and train the model
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=16)

    # Evaluate the model on the test set
    _, test_acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_acc)

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]
