from full_preprocessing import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from keras.utils import plot_model
import h5py

save_data_to_array(max_len=11)

X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, 20, 11, 1)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]


"""print(model.output.op.name)
saver = tf.train.Saver()
saver.save(K.get_session(), '/adil/NeuralNetwork/models/keras_model.ckpt')
model = get_model()"""
#saver.save(K.get_session(), '/tmp/keras_model.ckpt')
model = get_model()
model.fit(X_train, y_train_hot, batch_size=100, epochs=10000, verbose =1, validation_data=(X_test, y_test_hot))
model.save('new_model.h5')
plot_model(model, to_file='/home/adil/model.png')
print(predict('/home/adil/Downloads/validation_data/Anu/bed.ogg', model=model))
