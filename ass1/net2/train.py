import tensorflow as tf
from all_conv_model import model
from dataset import get_dataset

DIR = 'cifar10_alt'
lr = 0.0005
decay = 0.95
optimizer = 'rmsprop'
name = 'opt{}_lr{}_dc{}_allconv2'.format(optimizer, lr, decay)


train_dataset, test_dataset = get_dataset(batch_size=32, dataset_name='cifar10')

model_ = model('model')


class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, decay_rate, decay_steps):
        super(LearningRateScheduler, self).__init__()
        # tf.keras.backend.set_value(self.model.optimizer.lr, learning_rate)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.file_writer = tf.summary.FileWriter('./logs/{}/{}/custom'.format(DIR, name))

    def on_batch_begin(self, epoch, logs=None):

        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        learning_rate *= self.decay_rate ** (1.0 / self.decay_steps)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, learning_rate)

    def on_epoch_begin(self, epoch, logs=None):
        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        summary = tf.Summary(value=[tf.Summary.Value(tag='learning rate',
                                                     simple_value=learning_rate)])
        # summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        self.file_writer.add_summary(summary, epoch)
        self.file_writer.flush()


model_.compile(optimizer=tf.keras.optimizers.RMSprop(lr),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs/{}/{}'.format(DIR, name), write_graph=False),
    LearningRateScheduler(decay_rate=decay, decay_steps=1000)
]

model_.fit(train_dataset, epochs=50, steps_per_epoch=1000, verbose=1,
           validation_data=test_dataset, validation_steps=500, callbacks=callbacks)
