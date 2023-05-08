import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import torch

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras_preprocessing.image import load_img, img_to_array

# download dataset
train, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
size = 224


def resize_image(img_for_resize):

    img_for_resize = tf.cast(img_for_resize, tf.float32)
    img_for_resize = tf.image.resize(img_for_resize, (size, size))
    img_for_resize = img_for_resize / 255.0
    return img_for_resize


def model_train(size):
    train_resized = train[0].map(resize_image)
    train_batches = train_resized.shuffle(1000).batch(16)

    base_layers = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False)
    base_layers.trainable = False

    model = tf.keras.Sequential([
        base_layers,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(train_batches, epochs=1)

    img = load_img(f'haski.jpeg')
    img_array = img_to_array(img)
    img_resized, _ = resize_image(img_array)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'CAT' if prediction < 0.5 else 'DOG'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{pred_label} {prediction}')
    plt.show()


def save(model):
    torch.save(model.state_dict(), '/model_MobileNetV2')
    torch.save(model.optimizer.state_dict(), 'B:\My dear python projects\CATS_DOGS\optimizer')
    torch.save(model.train_losses, 'B:\My dear python projects\CATS_DOGS\losses')


# model_MobileNetV2.save('B:\My dear python projects\CATS_DOGS\model_MobileNetV2')


def evaluate():
    model = tf.keras.models.load_model('model_MobileNetV2')
    img = load_img(f'koshka.jpg')
    img_array = img_to_array(img)
    img_resized = resize_image(img_array)
    img_expended = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expended)[0][0]
    pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'
    plt.figure()
    plt.imshow(img)
    plt.title(f'{pred_label} {prediction}')
    plt.show()


evaluate()
