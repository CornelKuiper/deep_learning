import tensorflow as tf

TRAINING = True


def model(name):
    # depth of the convlayers
    depth = [64, 128, 256]
    x = inputs = tf.keras.layers.Input([32, 32, 3])
    tf.summary.image('x', x, 5)
    # make 2 convlayers with a depth as defined above
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   strides=1, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   strides=1, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(x)

    # flatten and make a dense layer of size 256
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # dense layer of size 10, followed by softmax (so we will output the most likely class as 1)
    x = tf.keras.layers.Dense(units=10)(x)
    x = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs, x, name=name)
