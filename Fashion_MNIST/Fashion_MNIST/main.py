import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.datasets.fashion_mnist as mnist
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# show one

plt.figure()
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()


# show first 25 with labels

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[y_train[i]])
#
# plt.show()

model = keras.Sequential([Flatten(input_shape=(28, 28)),
                          Dense(128, activation='relu'),
                          Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # SGD optimizer maybe

print(model.summary())

model.fit(x_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Accuracy is: ', test_acc)

pred = model.predict(x_train)
pred = np.argmax(pred, axis=1)

print('Prediction: ', pred[:20])
print('Label: ', y_train[:20])
