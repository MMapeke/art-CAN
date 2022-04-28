import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, BatchNormalization

# TODO: Get Basic DCGAN Style Architecture Working (Use Lab Code)
# TODO: use correct architecture based on paper + right noise generators, losses, etc.
# TODO: Confirm data pipeline works
# TODO: Integrate CAN Features

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())

    return model

class Generator(tf.keras.Model):
    def __init__(self, learning_rate):
        super(Generator, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.main = make_generator_model()

    def call(self, input):
        return self.main(input)

    def generator_loss(self, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate):
        super(Discriminator, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.main = make_discriminator_model()

        self.discriminate = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
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
        return total_loss