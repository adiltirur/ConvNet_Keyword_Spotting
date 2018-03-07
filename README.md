# Implementation of Convelutional Neural Network For Keyword Spotting in Keras

## Required libraries
* Tensorflow
* Keras
* librosa
* sklearn
* numoy
* tqdm
* h5py
* pyaudio
* time
* wave

## Getting the dataset
https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz

Once you have downloaded the dataset copy it into the data folder in the repository root directory or you can keep it anywhere but remember to change the data directory in the program.

## Building it From Scratch

So so to build the model from scratch, change the DATA_PATH variable in the nn_final_full.py to the Main folder of your dataset and on the last line in the predicit fuction point change the /home/adil/NeuralNetwork/validation_data/Shrilakshmi/bed.ogg to any of the audio file in the validation_data folder or you can use one of your recordings.

## Using the Pre-trained Model

Note in the pre trained model I have only trained 11 speeches, they are :
* bed
* zero
* one
* two
* three
* four
* five
* six
* seven
* eight
* nine

So to use this You can Run the program final.py and change model = load_model('/home/adil/NN_Project/keras/my_model.h5') to model = load_model(<to my_mdel.h5 in the model directory of the repository\>) and also WAVE_OUTPUT_FILENAME to <any directory you want\record.wav> also copy paste the same directory path to the third last line predict_dir
