import argparse
import tensorflow as tf 
from model import Generator

# if not working, check if defaults are consistent w main.py
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--data", type=int, default=0)
    parser.add_argument("--weights_path", type=str)
    # out path must be a .png
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    return args

def main(args):
    if (args.data > 2):
        print("--data must be 0, 1, 2")
        exit()
    
    dataset_name = None
    if (args.data == 0):
        dataset_name = "wikiart_ultra_slim"
        num_classes = 3
    elif (args.data == 1):
        dataset_name = "wikiart_slim"
        num_classes = 5
    elif (args.data == 2):
        dataset_name = "wikiart"
        num_classes = 27
        print("SANITY CHECK NUMBER OF CLASSES IN WIKIART on GCP, then remove this")
        exit()
    
    generator = Generator(args.gen_lr, args.beta, num_classes)
    
    generator.build(input_shape=(None, 100))
    generator.load_weights(args.weights_path)
    generator.compile()

    noise = tf.random.normal([9, args.latent_size])
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

    generated_img = tf.concat((
        tf.concat((img0, img1, img2), axis = 0), 
        tf.concat((img3, img4, img5), axis = 0),
        tf.concat((img6, img7, img8), axis = 0)),
        axis = 1)
        
    # Normalize from [-1, 1] -> [0, 255]
    generated_img = (generated_img + 1) * 0.5
    generated_img = generated_img * 255

    tf.keras.preprocessing.image.save_img(args.out_path, generated_img)

if __name__ == "__main__":
    args = parseArguments()
    main(args)