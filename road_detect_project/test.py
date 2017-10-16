import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, MaxPool2D, Conv2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(64, kernel_size=(13, 13), strides=(4, 4), activation='relu',
                 padding="same", input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(112, kernel_size=(4, 4), strides=(1, 1), activation='relu', padding='same'))
model.add(Dropout(0.9))
model.add(Conv2D(80, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation="sigmoid"))

model.compile(optimizer="sgd",
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.output)

imggen = ImageDataGenerator(rescale=1. / 255)
imggenerator = imggen.flow_from_directory(directory=r'./dataset', target_size=(64, 64), color_mode='rgb')
img = next(imggenerator)
res = model.predict(img[0])
print(res)
# print(img[0].shape)
