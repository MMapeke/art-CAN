from PIL import Image
import numpy as np
import os
from glob import glob
import tensorflow as tf
import tensorflow_datasets as tfds

def load_wikiart(root_folder_name='wikiart'):
	"""
    Load wikiart from ./data folder

    Inputs:
    - root_folder_name: str of the root folder of images

    Returns:
    - data: A list of strings, where each element is a filepath to an individual image
	- label_true: A list of strings, where each element is the art style of the image in the corresponding index in 'data'
	- label_index: The same as the 'label_true', except each element is an int representing a number unique to each art style
    - num_of_images: An int representing the number of images in the data list
	"""

	data = glob(os.path.join(f'../../../data/{root_folder_name}/**/', '*.jpg')) 
	
	num_of_images = len(data)
	label_true = [''] * num_of_images
	label_index = [0] * num_of_images

	prefix_length = len(f'../../../data/{root_folder_name}/')
	folder_path_list = glob(f'../../../data/{root_folder_name}/**/', recursive=True)[1:]

	label_to_folder_index = {} # will be filled as { art_style (string): index (int): }
	for index, folder_name in enumerate(folder_path_list):
		label = folder_name[prefix_length:-1] # prefix_length cuts out './cs1470-final/data/wikiart/' from string, leaving 'art_style' as the label for the images (excluding the '/' at the end)
		label_to_folder_index[label] = index

	for index, image_path in enumerate(data):
		label = image_path[prefix_length:][: image_path[prefix_length:].find('/')]
		if (label.find('\\') != -1): # Fix for windows filesystem
			label = label[:label.find('\\')]
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

def convert_to_tensor_dataset_2(data, labels, batch_size, image_size, buffer_size=1024):
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
	input = get_images(data, image_size, image_size) # [num_of_images, channel_sizes, height, width] 
	input = input / 255.0
	# [0, 1] -> [-1, 1]
	input = (input * 2) - 1.0
	
	# VAE assignment uses [batch_sz, channel_sz, height, width] instead, may need to reshape here?

	train_dataset = tf.data.Dataset.from_tensor_slices((input, labels))
	train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
	return train_dataset

def load_wikiart_as_image_folder_dataset(root_folder_name='wikiart', batch_size=None):
	"""
    Load wikiart from ./data folder

    Inputs:
    - root_folder_name: str of the root folder of images

    Returns:
	- tf.data.Dataset, or if split=None, dict<key: tfds.Split, value: tfds.data.Dataset>.
	"""

	builder = tfds.ImageFolder(os.path.dirname(os.path.abspath(__file__)) + f'/../../../data/{root_folder_name}/')
	ds = builder.as_dataset(batch_size=batch_size, shuffle_files=True)
	# print(builder.info)  # num examples, labels... are automatically calculated
	# tfds.show_examples(ds, builder.info)

	return ds

def get_images(image_paths, resize_height=64, resize_width=64):
	images = []
	for image_path in image_paths:
		images.append(get_image(image_path, resize_height, resize_width))
	return np.array(images)

def get_image(image_path, resize_height=64, resize_width=64):
	if (tf.is_tensor(image_path)):
		image_path = bytes.decode(image_path.numpy())
	image = Image.open(image_path)
	resized_image = image.resize((resize_height, resize_width))
	final_image = np.array(resized_image)
	return final_image
