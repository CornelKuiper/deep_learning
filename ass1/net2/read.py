import tensorflow as tf

size = 128


def post_process(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=16. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return image


def _parse_function(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=16. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return (image, label)
