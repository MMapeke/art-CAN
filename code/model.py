import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout
import numpy as np

# TODO: right noise generators, losses, etc. + initialize stuff like weights
# TODO: Confirm data pipeline works
# TODO: Confirm gpu compatible
# TODO: Integrate CAN Features

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.backend.binary_crossentropy

def make_generator_model():
    model = tf.keras.Sequential()
    
    model.add(Dense(4*4*512, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Reshape((4, 4, 512)))

    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.Activation(tf.nn.tanh))

    assert model.output_shape == (None, 64, 64, 3)  # Note: None is the batch size

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(BatchNormalization()) # reference doesn't have this
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    assert model.output_shape == (None, 4, 4, 512)  # Note: None is the batch size

    return model

class Generator(tf.keras.Model):
    def __init__(self, learning_rate):
        super(Generator, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.main = make_generator_model()

    def call(self, input):
        return self.main(input)

    def generator_loss(self, fake_output):
        return tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.main = make_discriminator_model()

        # May switch the discriminate and classification heads to dense layers
        self.discriminate = tf.keras.Sequential(
            [
                Conv2D(1, (4, 4), strides=(1, 1), padding='valid'),
                Reshape((1, )),
                tf.keras.layers.Activation(tf.nn.sigmoid)
            ]
        ) 
        
        # TODO: Classification Network
        self.classify = None

    def call(self, input):
        out = self.main(input)
        d_out = self.discriminate(out)

        # TODO: Classification Output
        c_out = None 
        return d_out, c_out

    def discriminator_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return tf.math.reduce_mean(total_loss)