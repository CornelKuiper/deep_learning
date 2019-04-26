import numpy as np
import tensorflow as tf
import pickle
import load_cifar
from tensorflow import keras

def cifar_model(shape):
    inputs = tf.keras.Input(shape=shape)
    x = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu')(x)
    predictions = keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs,outputs=predictions)
    return model    

def cnn_model(shape):
    inputs = tf.keras.Input(shape=shape)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    predictions = keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs,outputs=predictions)
    return model

def main():
    # ((train_data, train_labels),(test_data, test_labels)) = keras.datasets.fashion_mnist.load_data()
    # train_data = train_data.reshape((60000, 28, 28, 1))
    # test_data = test_data.reshape((10000, 28, 28, 1))
    # train_data, test_data = train_data / 255.0, test_data / 255.0

    train_data, train_labels = load_cifar.distorted_inputs(100)
    test_data, test_labels = load_cifar.inputs('test', 100)
    shape = [32,32,3]

    model = cifar_model(shape)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
      # Interrupt training if `val_loss` stops improving for over 2 epochs
      tf.keras.callbacks.EarlyStopping(patience=2, monitor='acc'),
      # Write TensorBoard logs to `./logs` directory
      tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    model.fit(train_data, train_labels, epochs=50, steps_per_epoch=500, callbacks=callbacks,validation_data=(test_data, test_labels),validation_steps=10)
    model.evaluate(test_data, test_labels, steps=100)

if __name__ == "__main__":
    main()













# def cnn_model():
#     model = keras.models.Sequential()
#     model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#     model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dense(10, activation='softmax'))
#     return model