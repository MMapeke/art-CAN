from PIL import Image
import numpy as np
import os
from glob import glob
import tensorflow as tf

def load_mnist(batch_size, buffer_size=1024):
    """
    Load and preprocess MNIST dataset from tf.keras.datasets.mnist.

    Inputs:
    - batch_size: An integer value of batch size.
    - buffer_size: Buffer size for random sampling in tf.data.Dataset.shuffle().

    Returns:
    - train_dataset: A tf.data.Dataset instance of MNIST dataset. Batching and shuffling are already supported.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0
    x_train = np.expand_dims(x_train, axis=1)  # [batch_sz, channel_sz, height, width]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
    return train_dataset

def load_wikiart():
	"""
    Load wikiart from ./data folder

    Inputs:
    - None

    Returns:
    - data: A list of strings, where each element is a filepath to an individual image
	- label_true: A list of strings, where each element is the art style of the image in the corresponding index in 'data'
	- label_index: The same as the 'label_true', except each element is an int representing a number unique to each art style
    - num_of_images: An int representing the number of images in the data list
	"""

	data = glob(os.path.join("./cs1470-final/data/wikiart_slim/**/", '*.jpg')) 
	num_of_images = len(data)
	label_true = [''] * num_of_images
	label_index = [0] * num_of_images

	prefix_length = len('./cs1470-final/data/wikiart_slim/') # 33
	folder_path_list = glob('./cs1470-final/data/wikiart_slim/**/', recursive=True)[1:]

	label_to_folder_index = {} # will be filled as { art_style (string): index (int): }
	for index, folder_name in enumerate(folder_path_list):
		label = folder_name[prefix_length:-1] # prefix_length cuts out './cs1470-final/data/wikiart/' from string, leaving 'art_style' as the label for the images (excluding the '/' at the end)
		label_to_folder_index[label] = index

	for index, image_path in enumerate(data):
		label = image_path[prefix_length:][: image_path[prefix_length:].find('/')]
		label_true[index] = label
		label_index[index] = label_to_folder_index[label]

	return data, label_true, label_index, num_of_images

def convert_to_tensor_dataset_1(data, labels, batch_size, buffer_size=1024):
	"""
	VERSION THAT USES IMAGE PATHS AS THE DATA

    Takes in a list of image paths and a list of labels and puts it into a tf.data.Dataset

    Inputs:
    - data: A list of strings, where each element is a filepath to an individual image
	- label: A list where each element is the art style of the image in the corresponding index in 'data'
    - batch_size: An integer value of batch size.
    - buffer_size: Buffer size for random sampling in tf.data.Dataset.shuffle().

    Returns:
    - train_dataset: A tf.data.Dataset instance of MNIST dataset. Batching and shuffling are already supported.
    """
	train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
	train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
	return train_dataset

def convert_to_tensor_dataset_2(data, labels, batch_size, buffer_size=1024):
	"""
	VERSION THAT USES FLATTENED AND CROPPED IMAGES AS THE DATA

    Takes in a list of image paths and a list of labels and puts it into a tf.data.Dataset

    Inputs:
	- data: A list of strings, where each element is a filepath to an individual image
	- labels: A list where each element is the art style of the image in the corresponding index in 'data'
    - batch_size: An integer value of batch size.
    - buffer_size: Buffer size for random sampling in tf.data.Dataset.shuffle().

    Returns:
    - train_dataset: A tf.data.Dataset instance of MNIST dataset. Batching and shuffling are already supported.
    """
	input = get_images(data[0:512]) # [num_of_images, channel_sizes, height, width] 
	input = input / 255.0
	
	# VAE assignment uses [batch_sz, channel_sz, height, width] instead, may need to reshape here?

	train_dataset = tf.data.Dataset.from_tensor_slices((input, labels[0:512]))
	train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
	return train_dataset

def get_images(image_paths, resize_height=64, resize_width=64):
	images = []
	for image_path in image_paths:
		images.append(get_image(image_path, resize_height, resize_width))
	return np.array(images)

def get_image(image_path, resize_height=64, resize_width=64):
	image = Image.open(image_path)
	resized_image = image.resize((resize_height, resize_width))
	final_image = np.asarray(resized_image)
	return final_image