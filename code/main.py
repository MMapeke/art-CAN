import argparse
import tensorflow as tf 
from model import Discriminator, Generator
from preprocessing import *
import numpy as np
from PIL import Image
import platform
import matplotlib.pyplot as plt

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("--gen_lr", type=float, default=1e-3)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--data", type=int, default=0)
    args = parser.parse_args()
    return args

def train_step(generator, discriminator, batch):
    noise = tf.random.normal([args.batch_size, args.latent_size])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        images, _ = batch

        generated_images = generator(noise)
        real_output, _ = discriminator(images)
        fake_output, _ = discriminator(generated_images)

        gen_loss = generator.generator_loss(fake_output)
        disc_loss = discriminator.discriminator_loss(real_output, fake_output)
        print("Gen Loss: ", gen_loss.numpy(), "Disc Loss: ", disc_loss.numpy())

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator.optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(generator, discriminator, dataset):
    gen_losses = []
    dis_losses = []


    noise = tf.random.normal([9, args.latent_size])
    for epoch in range(args.num_epochs):
        print("Epoch - ", epoch)
        epoch_gen_loss = 0
        epoch_dis_loss = 0
        num_batches = 0
        for _, batch, in enumerate(dataset):
            batch_gen_loss, batch_dis_loss = train_step(generator, discriminator, batch)
            epoch_gen_loss = epoch_gen_loss + batch_gen_loss
            epoch_dis_loss = epoch_dis_loss + batch_dis_loss
            num_batches = num_batches + 1

        gen_losses.append(epoch_gen_loss / num_batches)
        dis_losses.append(epoch_dis_loss / num_batches)
        
        # Sanity Check: Saving Generates Images after each epoch to check if something being learned
        output = generator(noise)
        img0 = output[0]
        img1 = output[1]
        img2 = output[2]
        img3 = output[3]
        img4 = output[4]
        img5 = output[5]
        img6 = output[6]
        img7 = output[7]
        img8 = output[8]

        generated_img = tf.concat(
            (
                tf.concat((img0, img1, img2), axis = 0), 
                tf.concat((img3, img4, img5), axis = 0),
                tf.concat((img6, img7, img8), axis = 0)),
            axis = 1)
        
        # Normalize from [-1, 1] -> [0, 255]
        generated_img = (generated_img + 1) * 0.5
        generated_img = generated_img * 255

        # Switch here for custom directory names
        inter_dir = "intermediate-gen_lr_" + str(args.gen_lr) + "-disc_lr_" + str(args.disc_lr) + "-beta_" + str(args.beta) + "-dataset_" + str(args.data)
        
        if(platform.system() == "Darwin" or platform.system() == "Linux"): # MacOS / Linux and tf doesn't work well with relative filepaths
            inter_dir = os.path.dirname(os.path.abspath(__file__)) +  "/../results/" + inter_dir
            if (not os.path.exists(inter_dir)):
                os.makedirs(inter_dir)
            tf.keras.preprocessing.image.save_img(inter_dir + "/epoch-" + str(epoch) + ".png", generated_img)        
        else:
            inter_dir = "../results/" + inter_dir
            if (not os.path.exists(inter_dir)):
                os.makedirs(inter_dir)
            tf.keras.preprocessing.image.save_img(inter_dir + "/epoch-" + str(epoch) + ".png", generated_img)
        

        # Logic for saving intermediate models would go here

    return gen_losses, dis_losses, inter_dir

def main(args):

    if (args.data > 2):
        print("--data must be 0, 1, 2")
        exit()
    
    dataset_name = None
    if (args.data == 0):
        dataset_name = "wikiart_ultra_slim"
    elif (args.data == 1):
        dataset_name = "wikiart_slim"
    elif (args.data == 2):
        dataset_name = "wikiart"

    print(tf.test.is_gpu_available())
    if (not tf.test.is_gpu_available()):
        pass
        # exit()

    # Version 1: Loading as list, then passing to tf.dataset
    data, label_true, label_index, num_of_images = load_wikiart(dataset_name)
    print("Number of images - ", num_of_images)
    train_dataset = convert_to_tensor_dataset_2(data, label_index, args.batch_size, args.image_size)

    # Version 2: Using Image Folder
    # Not sure if this is correct logic for number of images
    # train_dataset = load_wikiart_as_image_folder_dataset('wikiart_ultra_slim', args.batch_size)
    # i = 0 
    # for _ in enumerate(train_dataset):
        # i = i + 1
    # print("Number of images: ", i * args.batch_size)

    """
        Preprocessing note: 

        in the other assignments, all of the images get preprocessed into a list, but I'm
        generally unsure if that's possible here (since there will be 25gb of images, storing in that in RAM is 
        impossible on local machine and prob expensive on GCP), so for now there's two 'convert_to_tensor_dataset' 
        functions, one where the 'input' is a flattened image, and one where the 'input' is a image path, with the 
        expectation that the image will get read later in batches
    """

    print("Learning rates: ", str(args.gen_lr), str(args.disc_lr))
    generator = Generator(args.gen_lr, args.beta)
    discriminator = Discriminator(args.disc_lr, args.beta)
    
    generator.build(input_shape=(None, 100))
    generator.summary()
    discriminator.build(input_shape=(None, 64, 64, 3))
    discriminator.summary()

    gen_losses, dis_losses, directory = train(generator, discriminator, train_dataset)
    epochs = range(len(gen_losses))
    plt.plot(epochs, gen_losses, 'b', label='Generator Loss')
    plt.plot(epochs, dis_losses, 'b', label='Discriminator Loss')
    plt.title('Generator and Discriminator loss')
    plt.legend()
    plt.savefig(directory + '/loss_graph.png')
    

if __name__ == "__main__":
    args = parseArguments()
    main(args)
