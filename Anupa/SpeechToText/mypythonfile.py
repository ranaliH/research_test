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

file_names = os.listdir(audio_path1)

# Create a CSV file to store the file names
csv_file = 'file_names.csv'  # Replace with the desired CSV file name

# Write the file names to the CSV file
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['File Name'])  # Write header row
    writer.writerows([[file_name] for file_name in file_names])

print('File names saved to', csv_file)