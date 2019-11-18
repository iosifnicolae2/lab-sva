from __future__ import print_function

import cv2
import keras
import numpy
from keras.backend import argmax
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 3
epochs = 40

# input image dimensions
img_rows, img_cols = 4, 4


x_train = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
])

y_train = numpy.array([
    1,
    1,
    2,
    2,
    1,
    1,
    2,
    2,
])


x_test = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

y_test = numpy.array([
    1
])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2 , 2), # masca care se aplica pe imaginea de intrare ca sa formeze cele 32 de filtre
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(12, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Extrage maximul
model.add(Dense(128, activation='relu'))  # fiecare la fiecare
model.add(Dropout(0.25))  #
model.add(Flatten()) # il face array din vector
model.add(Dense(128, activation='relu'))  # fiecare la fiecare
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) # specific problemelor de categorisire

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.summary()
model.save("model-mnsit.h5")

saved_model = keras.models.load_model('model-mnsit.h5')

score = saved_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


input = numpy.array([
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
]) # ar trebui sa dea 2

input = input.reshape(input.shape[0], img_rows, img_cols, 1)
input = input.astype('float32')
input /= 255

output = saved_model.predict_proba(input)
output_argmax = argmax(output)
print("Evaluating:\n  input:\n{}\n  output:\n{}\n  output_argmax:\n{}".format(input, output, output_argmax))

if __name__ == '__main__':
    print("end")