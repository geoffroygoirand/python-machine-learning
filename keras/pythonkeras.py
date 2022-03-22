import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

model.summary()

model.fit(X,Y, nb_epoch=100, verbose=False)

model.compile(loss='caterogical_crossentropy', optimizer='sgd', metrics=['accuracy'])
