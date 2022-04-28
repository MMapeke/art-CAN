import argparse
import tensorflow as tf 
from model import Discriminator, Generator
from preprocessing import load_wikiart, convert_to_tensor_dataset_2

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    args = parser.parse_args()
    return args

@tf.function
def train_step(generator, discriminator, batch):
    noise = tf.random.normal([args.batch_size, args.latent_size])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        images, classes = batch

        generated_images = generator(noise)

        print(images.shape)
        real_output = discriminator(images)
        print(real_output.shape)
        exit()
        fake_output = discriminator(images)

        gen_loss = generator.generator_loss(fake_output)
        disc_loss = discriminator.discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator.optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


def train(generator, discriminator, dataset):
    for epoch in range(args.num_epochs):
        print("Epoch - ", epoch)
        for _, batch, in enumerate(dataset):
            train_step(generator, discriminator, batch)
        
        # Logic for saving intermediate models would go here

def main(args):
    
    data, label_true, label_index, num_of_images = load_wikiart('wikiart_ultra_slim')
    print("Number of images - ", num_of_images)


    """
        Preprocessing note: 

        in the other assignments, all of the images get preprocessed into a list, but I'm
        generally unsure if that's possible here (since there will be 25gb of images, storing in that in RAM is 
        impossible on local machine and prob expensive on GCP), so for now there's two 'convert_to_tensor_dataset' 
        functions, one where the 'input' is a flattened image, and one where the 'input' is a image path, with the 
        expectation that the image will get read later in batches
    """

    train_dataset = convert_to_tensor_dataset_2(data, label_index, args.batch_size)
    
    generator = Generator(args.gen_lr)
    discriminator = Discriminator(args.disc_lr)
    train(generator, discriminator, train_dataset)

if __name__ == "__main__":
    args = parseArguments()
    main(args)