#%%
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np

import os
import zipfile
import random
import urllib.request
from IPython import display
from typing import List,Tuple
# set Seed
SEED : int = 5
np.random.seed(SEED)

BATCH_SIZE : int = 2000
LATENT_DIM : int = 512
IMAGE_SIZE : int = 64	
## For Typing
Tensor = tf.Tensor
Layer = tf.keras.layers.Layer
Dataset = tf.data.Dataset
Model = tf.keras.models.Model
#%%
try:
	os.mkdir("anime/")
except OSError :
	pass

#data_url : str = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Resources/anime-faces.zip"

data_file_name : str = "animefaces.zip"
download_dir : str = "anime/"

with zipfile.ZipFile(data_file_name, "r") as zip_ref:
	zip_ref.extractall(download_dir)

"""
	Utilities Functions
"""
#%%
def get_dataset_slice_paths(image_dir : str) :
	image_file_list :List[str] = os.listdir(image_dir)
	image_paths : List[str] = [os.path.join(image_dir,file_name) for file_name in image_file_list]

	return image_paths

def map_image(image_filename : str) -> Tensor:

	raw_image : Tensor = tf.io.read_file(image_filename)
	image : Tensor = tf.image.decode_jpeg(raw_image)
	image =  tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
	image = tf.reshape(image, shape=(IMAGE_SIZE, IMAGE_SIZE, 3)) 

	return image


paths : str = get_dataset_slice_paths("/tmp/anime/images")

random.shuffle(paths)

paths_len : int = len(paths)
train_paths_len : int = int(paths_len * 0.8)

train_paths_list : List[str] = paths[:train_paths_len]
val_paths_list : List[str] = paths[train_paths_len:]

# Prepare the Training datasets
training_dataset : Dataset = tf.data.Dataset.from_tensor_slices(train_paths_list)
training_dataset = training_dataset.map(map_image).shuffle(1000).batch(BATCH_SIZE)

# preprate the Validation Datasets
validation_dataset : Dataset = tf.data.Dataset.from_tensor_slices(val_paths_list)
validation_dataset = validation_dataset.map(map_image).batch(BATCH_SIZE)

print(f"Number of Batches in the Training Set : {len(training_dataset)}")
print(f"Number of Batches in the Training Set : {len(validation_dataset)}")

def display_faces(dataset : Dataset , size : int =10):
	dataset = dataset.unbatch().take(size)
	ncols : int = 3
	nrows : int = size // ncols + 1
	i : int = 0
	for image in dataset:
		i += 1
		disp_image = np.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
		plt.subplot(nrows, ncols, i)
		plt.yticks([])
		plt.xticks([])
		plt.imshow(disp_image)

def display_one_row(disp_image : List[Tensor], offset : int, shape=(64,64)):
	for idx, image in enumerate(disp_image):
		plt.subplot(3, 10 , offset + idx+1)
		plt.yticks([])
		plt.xticks([])
		plt.imshow(image)


def display_result(input_list, predicted_list)-> None:
	#TODO
	#todo Make your life easier dude 
	display_one_row(disp_image, 0, shape=(IMAGE_SIZE, IMAGE_SIZE,3))
	display_one_row(disp_image, 10, shape=(IMAGE_SIZE, IMAGE_SIZE,3))

# %%
display_faces(validation_dataset, size=10)

#%%
#TODO//
#todo// CREATE Encoder Layer First
#todo// CREATE Sampling Layer
#todo// Integrate it into Encoder Model Cok 
def encoder_model(input_shape , latent_dim):
	"""
	Create Encoder Model

	Args:
		input_shape (Tuple[int,int,int]): onlyShape
		latent_dim (int): Latent Dim units

	Returns:
		tf.keras.Model : Encoder Model
		vis_shape :  Shape before flattened
	"""
	inputs : Layer = tf.keras.layers.Input(shape=input_shape) #* Its Like Starting Point, Data Inputed

	mu, sigma, vis_shape = encoder_layers(inputs, latent_dim) #* This guys right here Will encode the input data

	z = Sampling_layer()([mu, sigma]) #* Feed The Mean and std, to calculate Z

	return tf.keras.Model(inputs, outputs= [mu,sigma,z]), vis_shape #*

def encoder_layers(inputs : Tensor , latent_dim : int) -> Tuple[Tensor, Tensor, Tuple[int,int,int,int]]:

	"""
	Encoder layers,compute the mean and std of the latent vector

	Returns:
			Mu Tensor, sigma : Tensor, and vis.shape : Tuple
	"""
	#TODO
	#todo// 2 2 CNN, each having maxpooling ?
	#todo// Connect it to layer 
	#NOTE 
	#? BEFORE GOING TO LATENT SPACE, SAVE the previous Layer Shape, Fungsine Biar bisa dapet Total Represntasi yang dibutuhkan untuk rekonstruksi
	hidden_layer_1 : Tensor = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2,activation=tf.nn.relu, padding="same",name="encoder_conv1")(inputs)
	hidden_layer_1 : Tensor = tf.keras.layers.BatchNormalization()(hidden_layer_1)
	hidden_layer_2 : Tensor = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2,activation=tf.nn.relu, padding="same",name="encoder_conv2")(hidden_layer_1);hidden_layer_3 : Tensor = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(hidden_layer_1)
	hidden_layer_2 : Tensor = tf.keras.layers.BatchNormalization()(hidden_layer_2)
	hidden_layer_3 : Tensor = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu',)(hidden_layer_2)
	vis : Tensor = tf.keras.layers.BatchNormalization(name="encoder_batchnorm1")(hidden_layer_3) #* YANG INI YA, di return

	fcn : Tensor = tf.keras.layers.Flatten(name="encoder_flatten")(vis)

	fcn : Tensor = tf.keras.layers.Dense(1024, activation=tf.nn.relu, name="encoder_dense1")(fcn) #* Using paper recommendation, 1024
	fcn : Tensor = tf.keras.layers.BatchNormalization()(fcn)

	mu : Tensor = tf.keras.layers.Dense(latent_dim, activation = "linear")(fcn) #* This shit is real
	sigma : Tensor = tf.keras.layers.Dense(latent_dim, activation= "linear")(fcn)
	
	return mu, sigma, vis.shape

class Sampling_layer(tf.keras.layers.Layer):
	def call(self, inputs : Tensor) -> Tensor:
		mu : Tensor = inputs[0]
		sigma : Tensor = inputs[1]
		epsilon : Tensor = tf.keras.backend.random_normal(shape=tf.shape(mu))
		return mu + tf.exp(0.5* sigma) * epsilon


#TODO
#todo CREATE Decoder Layer First
#NOTE
#? OUTPUT LAYER NEED TO HAVE SAME SHAPE with INPUTS
#? GW GANTENG .

def decoder_layer(inputs : Tensor, conv_shape : Tuple[int,int,int,int])-> Tensor: 
	"""
	Rekonsturksi Image

	Args:
		inputs (Tensor): Z, from encoder, but already created the Input Layer
		conv_shape (Tuple[int,int,int,int]): : shape before flattened , used for reshaping

	Returns:
		Tranposed (Tensor) : Reconstructed Image
	"""

#NOTE : #? inputs Berupa Laten Space , ini dari Encoder, Latent space ()
#NOTE : #? Conv_shape isinya : (height, width, channel, filters)

#todo// Ambil Seluruh Representasi dari si Z
#todo// Reshape ke bentuk yang sama dengan input layer
#todo Train dengan Konvolusi Transpose
	hidden_layer_1 : Tensor = tf.keras.layers.Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation=tf.nn.relu,name="decoder_dense1")(inputs)
	hidden_layer_1 : Tensor = tf.keras.layers.BatchNormalization()(hidden_layer_1)
	reshaped_layer : Tensor = tf.keras.layers.Reshape(target_shape=(conv_shape[1],conv_shape[2], conv_shape[3]))(hidden_layer_1)
	
	transposed = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=2, activation=tf.nn.relu, padding="same",name="decoder_Tconv1")(reshaped_layer)
	
	transposed_normalized = tf.keras.layers.BatchNormalization()(transposed)
	transposed = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, activation=tf.nn.relu, padding="same",name="decoder_Tconv2")(transposed_normalized)
	
	transposed_normalized = tf.keras.layers.BatchNormalization()(transposed)
	transposed = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, activation=tf.nn.relu, padding="same",name="decoder_Tconv3")(transposed_normalized)
	
	transposed_normalized = tf.keras.layers.BatchNormalization()(transposed)
	transposed = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3,3), strides=1, activation=tf.nn.sigmoid, padding="same",name="decoder_Tconv4")(transposed_normalized)


	return transposed
def decoder_model(latent_dim, conv_shape) :
	"""
	Decoder Model , Just Integrate stuff for creating model

	Args:
		latent_dim ([Tensor]): its for input layer of decoder, From dense Layer
		conv_shape ([Tuple[int,int,int,int]]): shape before flattened, used for reshaping

	Returns:
		Model : Decoder Model
	"""
	inputs : Layer = tf.keras.layers.Input(shape=latent_dim) #* This guy here is the input layer, Z
	outputs : Layer = decoder_layer(inputs, conv_shape)
	return tf.keras.Model(inputs, outputs)

def k1_reconstruction_loss(inputs : Tensor , outputs : Tensor, mu : Tensor, sigma : Tensor):
	"""
	Reconstruction Loss

	Args:
		inputs (Tensor): Inputs from the original image
		outputs (Tensor): Outputs from the decoder,i.e Reconstruced Image
		mu (Tensor): Mean of the latent vector
		sigma (Tensor): Std of the latent vector
	"""
	k1_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
	k1_loss = tf.reduce_mean(k1_loss) * -0.5

	return k1_loss

def vae(encoder : Model, decoder:Model, input_shape:Model):
	"""
	VAE Model

	Args:
		encoder (Model): Yeah the Model
		decoder (Model): [description]
		input_shape (Model): Input Shape From the original Image

	Returns:
		Model With Configured Loss 
	"""

	input_layer : Layer = tf.keras.layers.Input(shape=input_shape)
	mu, sigma, z  = encoder(input_layer) #*	 This is not encoder layers, but the output of encoder model 
	reconstructed = decoder(z)

	model = tf.keras.models.Model(inputs = input_layer, outputs=reconstructed)

	loss = k1_reconstruction_loss(input_layer, z, mu, sigma)
	model.add_loss(loss)
	return model

def get_models(input_shape, latent_dim):
	encoder, conv_shape = encoder_model(input_shape, latent_dim)
	decoder = decoder_model(latent_dim, conv_shape)
	vae_model = vae(encoder, decoder, input_shape=input_shape)

	return encoder, decoder, vae_model 


#TODO PREPARE THE MODELS 
encoder, decoder, vae = get_models(input_shape=(64,64,3), latent_dim=LATENT_DIM)
#TODO
#// Settingup the optmizers
#// Settingup The Metrics

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
loss_metric = tf.keras.metrics.Mean()  #* We only need how good our model reconstructed the models, So we use Mean
mse_loss = tf.keras.losses.MeanSquaredError() #* This guys here, is for Vae LOSS, this is calculated from the constructed and the real
bce_loss = tf.keras.losses.BinaryCrossentropy() #* This guys here, Same as above, mse, but here using different equation. 

def generate_and_save_images(model : Model, epochs : int, step : int, test_input):

	predictions = model.predict(test_input)

	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]): #* We need to loop for the number of images, Batch
		plt.subplot(4, 4, i+1)
		img = predictions[i, :,:,:] * 255
		img = img.astype(np.int32)
		plt.imshow(img)
		plt.axis("off")
	
	fig.suptitle("Epochs : {} , Step : {}".format(epochs, step))
	plt.savefig("image_at_epoch_{:04d}_step_{:04d}.png".format(epochs, step))
	plt.show()


EPOCHS : int = 100

random_vector_noise : Tensor = tf.random.normal(shape=(16, LATENT_DIM)) #* Just For Visualization
generate_and_save_images(decoder, 0, 0, random_vector_noise)

for epoch in range(EPOCHS):
	print(f"Start Epoch {epoch}")

for step, x_batch_train in enumerate(training_dataset):
	with tf.GradientTape() as tape:

			reconstructed = vae(x_batch_train)

			flattened_inputs : Tensor = tf.reshape(x_batch_train,shape=(-1)) # Flattened IT
			flattened_reconstructed : Tensor = tf.reshape(reconstructed,shape=(-1)) # Flattened ITms
			loss = mse_loss(flattened_inputs, flattened_reconstructed) * 64*64*3 #* Loss is calculated from the constructed and the real ,multiplied by all pixels , for total loss of all pixels

			loss = loss + sum(vae.losses)

	#* Calculating the gradients
	gradients = tape.gradient(loss, vae.trainable_weights)
	optimizer.apply_gradients(zip(gradients, vae.trainable_weights))

	loss_metric(loss)

	if step % 10 == 0:
		display.clear_output(wait=False)
		generate_and_save_images(decoder, epoch, step, random_vector_noise)
		print(f"Epoch {epoch} step {step} mean loss = {loss_metric.result().numpy()}")