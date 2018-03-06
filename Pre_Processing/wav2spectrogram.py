#import the pyplot and wavfile modules

import matplotlib.pyplot as plot

from scipy.io import wavfile
import os
from PIL import Image

"""DIR = '/home/adil/Desktop/'
size = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
# Read the wav file (mono)
for i in range(500):"""
samplingFrequency, signalData = wavfile.read('/home/adil/Downloads/validation_data/bhushan/bed.wav')

plot.plot(241)
plot.axis('off')
plot.specgram(signalData,Fs=samplingFrequency)

plot.savefig('/home/adil/Downloads/validation_data/bhushan/bed.png')    

    #img = Image.open('/home/adil/NN_Project/dataset/spectrogram/nine/'+str(i)+'.png').convert('L')
    #img.save('/home/adil/NN_Project/dataset/spectrogram/bw/nine/'+str(i)+'.png')
