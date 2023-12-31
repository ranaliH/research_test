# import torch
# import librosa
# import numpy as np
# import soundfile as sf
# from scipy.io import wavfile
# import IPython
# from IPython.display import Audio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# file_name = 'my-audio.wav'
# Audio(file_name)

# data = wavfile.read(file_name)
# framerate = data[0]
# sounddata = data[1]
# time = np.arange(0,len(sounddata))/framerate
# print('Sampling rate:',framerate,'Hz')
# input_audio, _ = librosa.load(file_name, sr=16000)

# input_values = tokenizer(input_audio, return_tensors="pt").input_values
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = tokenizer.batch_decode(predicted_ids)[0]
# print(transcription)


import tensorflow as tf
from keras import layers
import os
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import csv
import keras as ke
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from pydub import AudioSegment
from jiwer import wer
import tensorflow_io as tfio


warnings.filterwarnings("ignore")
audio_path1 = '../AudioData/clips/'
csvpath = "../AudioData/All_Voice2.csv"

df_data = pd.read_csv(csvpath)
df_data = pd.DataFrame(df_data)


split = int(len(df_data) *0.70)
df_train = df_data[:split]
df_val = df_data[split:]

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!-. "]

char_to_num = ke.layers.StringLookup(vocabulary=characters, oov_token="")

num_to_char = ke.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),oov_token="",invert=True
)

frame_length = 256

frame_step = 160

fit_length = 384

def encode_signal_sample(wav_file,label):
    file = tf.io.read_file(audio_path1 + wav_file)#changed

    audio, _= tf.audio.decode_wav(file)
    audio = tf.squeeze(audio,axis=-1)
    audio = tf.cast(audio, tf.float32)

    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fit_length
    )


    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram,0.5)

    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)

    return spectrogram, label

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["path"]),list(df_train["sentence"]))
)

train_dataset = (
    train_dataset.map(encode_signal_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["path"]),list(df_val["sentence"]))
)

validation_dataset = (
    validation_dataset.map(encode_signal_sample,num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

def CTCLoss(y_true,y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len,1),dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len,1),dtype="int64")

    loss = K.ctc_batch_cost(y_true,y_pred,input_length,label_length)
    return loss

def build_model(input_dim,output_dim,rnn_layers=5,rnn_units=128):
    input_spetrogram = layers.Input((None,input_dim),name="input")
    x = layers.Reshape((-1,input_dim,1), name = "expand_dim")(input_spetrogram)
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11,41],
        strides=[2,2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)

    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=[11,21],
        strides=[1,2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)

    x = layers.BatchNormalization(name="conv_2_bin")(x)
    x = layers.ReLU(name="conv_2_relu")(x)

    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    for i in range(1, rnn_layers +1):
        recurrent = layers.GRU(
            units = rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}"
        )
        if i<rnn_layers:
            x = layers.Dropout(rate=0.5)(x)

    x = layers.Dense(units=rnn_units *2,name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)

    output = layers.Dense(units=output_dim + 1, activation = "softmax")(x)

    model = ke.Model(input_spetrogram, output, name="DeepSpeech_2")

    opt = ke.optimizers.Adam(learning_rate=0.4)

    model.compile(optimizer=opt,loss=CTCLoss)

    return model

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

class CallbackEval(ke.callbacks.Callback):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        # wer_score = wer(targets, predictions)
        # print("-" * 100)
        # print(f"Word Error Rate: {wer_score: 4f}")
        # print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Targets     : {targets[i]}")
            print(f"Predictions : {predictions[i]}")
            print("-" * 100)

model =build_model(
    input_dim=193,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=512,
)

model.summary(line_length=110)

epochs =90
validation_callback = CallbackEval(validation_dataset)

history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = epochs,
    callbacks=[validation_callback]
)

predictions = []
targets = []

for batch in validation_dataset:
    X,y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)


for i in np.random.randint(0,len(predictions),5):
    print(f"Target  : {targets[i]}")
    print(f"Predictions  : {predictions[i]}")
    print("-" * 100)
