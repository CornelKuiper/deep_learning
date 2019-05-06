import tensorflow as tf
from all_conv_model import model # import model for model 1 and all_conv_model for model 2
from dataset import get_dataset

DIR = 'cifar10_alt'     # directory for the logs
lr = 0.0005             # learning rate
decay = 0.95            # decay rate
optimizer = 'rmsprop'
name = 'opt{}_lr{}_dc{}_allconv2'.format(optimizer, lr, decay) # filename


train_dataset, test_dataset = get_dataset(batch_size=32, dataset_name='cifar10')

model_ = model('model')

# learning rate schedular for the learning rate decay. 
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


# compilation of the model, here the optimizer can be changed
model_.compile(optimizer=tf.keras.optimizers.RMSprop(lr),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# the callbacks are made, here the learning rate scheduler is also used. The decay steps can be changed here. 
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs/{}/{}'.format(DIR, name), write_graph=False),
    LearningRateScheduler(decay_rate=decay, decay_steps=1000)
]

# fit the model, and perform validation after each epoch.
model_.fit(train_dataset, epochs=50, steps_per_epoch=1000, verbose=1,
           validation_data=test_dataset, validation_steps=500, callbacks=callbacks)
