import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow import keras
from keras.layers import Dense, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split

# pipeline
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2)

# model
model = keras.Sequential([Flatten(input_shape=(28, 28, 1)),
                          Dense(20, activation='relu'),
                          BatchNormalization(),
                          Dense(10, activation='softmax')])
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_split, y_train_split, batch_size=32, epochs=20, validation_data=(x_val_split, y_val_split))

# prediction
n = 0
res = model.predict(np.expand_dims(x_test[n], axis=0))
print(np.argmax(res))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred[:20])
print(y_test[:20])
