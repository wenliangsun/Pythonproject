from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
(x_train,train_label),(x_test,test_label) = mnist.load_data()
x_train = x_train.reshape((60000,784))
train_label = to_categorical(train_label,num_classes=10)
print(x_train.shape)
model = Sequential()
model.add(Dense(100,activation='relu',input_dim=784))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='sgd',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train,train_label,batch_size=32,epochs=10)