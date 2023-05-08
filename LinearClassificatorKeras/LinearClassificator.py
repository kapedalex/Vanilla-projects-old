import keras
from keras.layers import Dense


def create_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(1,)))
    model.add(keras.layers.Dense(units=32, activation="linear"))
    model.add(keras.layers.Dense(units=1, ))
    model.compile(loss="mse", optimizer="sgd")
    return model


def predict():
    input_data = ([0.3, 0.7, 0.9])
    output_data = ([0.5, 0.9, 1.0])
    model = create_model()
    model.fit(x=input_data, y=output_data, epochs=500)
    model.summary()
    predicted = model.predict([0.3])

    print("prediction: " + str(predicted))
    print()


predict()
