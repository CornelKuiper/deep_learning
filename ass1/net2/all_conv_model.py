import tensorflow as tf

TRAINING = True


def model(name):
    # size of the layers
    depth = [128, 256]
    x = inputs = tf.keras.layers.Input([32, 32, 3])
    # make 3 convlayers with the size as defined in depth
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   strides=1, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   strides=1, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   strides=1, padding='SAME')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)


    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3,
                               strides=1, padding='SAME')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1,
                               strides=1, padding='SAME')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # the last convlayer results in 10 filters, corresponding to the amount of classes
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=1,
                               strides=1, padding='SAME')(x)
    # global average pooling, before the softmax is performed
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs, x, name=name)


