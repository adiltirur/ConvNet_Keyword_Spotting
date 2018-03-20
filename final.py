import tensorflow as tf

from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from tqdm import tqdm

#For MFCC
import librosa

import os
import argparse
import sys

from sklearn.model_selection import train_test_split

import numpy as np

#For the audo recordings
import pyaudio
import wave

import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'# to not log tf cpu error

print ("Neural Network Project")
print ("Convelutional Neural Network For Keyword Spotting")
print ("Adil Chakkala paramba")
print ("1817501")

for i in range (500):#Running the model just more # times
    model = load_model('/home/adil/NN_Project/keras/my_model.h5')
    DATA_PATH = "/home/adil/NN_Project/keras/data/"

    FORMAT = pyaudio.paInt16
    CHANNELS = 1#Mono
    RATE = 16000#Information bits per second
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "/home/adil/NN_Project/keras/record.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print "Speak"
    print time.sleep(2)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)#Recording the audio
    #print "finished recording"


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    def get_model():#Building the network with Downsampling
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
        model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
        model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def predict(filepath, model):#The prediction function
        sample = wav2mfcc(filepath)
        sample_reshaped = sample.reshape(1, 20, 11, 1)
        return get_labels()[0][
                np.argmax(model.predict(sample_reshaped))
        ]

    def wav2mfcc(file_path, max_len=11):#The MFCC convert function
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if (max_len > mfcc.shape[1]):
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

    def get_labels(path=DATA_PATH):
        labels = os.listdir(path)
        label_indices = np.arange(0, len(labels))
        return labels, label_indices, to_categorical(label_indices)

    predict_dir = "/home/adil/NN_Project/keras/record.wav"
    print("prediction = "+predict(predict_dir, model=model))
    print("\n \n ================================================= \n \n ================================================= \n \n")
# python -W ignore final.py 2>/dev/null
