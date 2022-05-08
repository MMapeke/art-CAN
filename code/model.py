import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Softmax
import numpy as np

# TODO: Confirm data pipeline works for all datasets
# TODO: Model Saving + Loading, Visualizations, Plots

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
category_cross_entropy = tf.keras.losses.CategoricalCrossentropy()

def make_generator_model():
    model = tf.keras.Sequential()
    
    model.add(Dense(4*4*512, use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Reshape((4, 4, 512)))

    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(tf.keras.layers.Activation(tf.nn.tanh))

    assert model.output_shape == (None, 64, 64, 3)  # Note: None is the batch size

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     input_shape=[64, 64, 3]))
    model.add(BatchNormalization()) # reference doesn't have this
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    assert model.output_shape == (None, 4, 4, 512)  # Note: None is the batch size

    return model

class Generator(tf.keras.Model):
    def __init__(self, learning_rate, beta, num_classes):
        super(Generator, self).__init__()
        self.K = num_classes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta)
        self.main = make_generator_model()

    def call(self, input):
        return self.main(input)

    def generator_loss(self, fake_output):
        return tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

    # TODO: Unsure about this implementation
    def generator_loss_CAN(self, fake_output, fake_predicted_classes):
        gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        # Implementing by doing categorical cross entropy with uniform distribution
        y = tf.fill(fake_predicted_classes.shape, float(1) / self.K)
        style_amb_loss = category_cross_entropy(y, fake_predicted_classes)
    
        total_loss = gan_loss + style_amb_loss

        return total_loss

class Discriminator(tf.keras.Model):
    def __init__(self, learning_rate, beta, num_classes):
        super().__init__()
        self.K = num_classes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta)
        self.main = make_discriminator_model()

        # May switch the discriminate and classification heads to dense layers
        self.discriminate = tf.keras.Sequential(
            [
                Conv2D(1, (4, 4), strides=(1, 1), padding='valid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)),
                Reshape((1, )),
                # tf.keras.layers.Activation(tf.nn.sigmoid)
            ],
            name="discrimination_head"
        ) 
        
        # TODO: This has insane amount of parameters, maybe tune for 64x64?
        self.classify = tf.keras.Sequential(
            [
                Flatten(),
                Dense(1024),
                LeakyReLU(0.2),
                Dense(512),
                LeakyReLU(0.2),
                Dense(self.K),
                Softmax()
            ],
            name = "classification_head"
        )

    def call(self, input):
        out = self.main(input)
        d_out = self.discriminate(out)

        # style classification output
        c_out = self.classify(out)
        return d_out, c_out

    def discriminator_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return tf.math.reduce_mean(total_loss)

    def discriminator_loss_CAN(self, real_output, fake_output, y, real_predicted_classes):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        can_classification_loss = category_cross_entropy(y, real_predicted_classes)
 
        total_loss = real_loss + fake_loss + can_classification_loss
        return tf.math.reduce_mean(total_loss)