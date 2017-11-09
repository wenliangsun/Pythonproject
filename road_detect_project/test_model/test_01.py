import keras
from keras.models import Sequential
from keras.layers import Dropout, MaxPool2D, Conv2D
from keras.layers import Dense, Flatten, Reshape
from keras.optimizers import SGD

from road_detect_project.model.params import optimization_params, dataset_params
from road_detect_project.model.data import AerialDataset

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
model.add(Reshape(input_shape=(256,), target_shape=(16, 16)))

opt_params = optimization_params
sgd = SGD(lr=0.0014, decay=0.95, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

dataset = AerialDataset()
path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
params = dataset_params
dataset.load(path, params=params)
train_gen = dataset.gen_data("train", epoch=100, batch_size=16)
valid_gen = dataset.gen_data("valid", epoch=1, batch_size=16)
test_gen = dataset.gen_data("test_PyQt5", epoch=1, batch_size=1)

model.fit_generator(train_gen, steps_per_epoch=100, epochs=8,validation_data=valid_gen,
                    validation_steps=5)
res = model.predict_generator(test_gen,steps=1)
print(res)