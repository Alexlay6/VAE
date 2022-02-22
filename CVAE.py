# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:14:21 2022

@author: alexj
This willbe experimentation of variational autoencoders to assist with quality of networks to produce 
accruate feature maps from latent spaces
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import backend as K
import time
import matplotlib.pyplot as plt
import numpy as np
# Setting Initial parameters
batch_size = 100
beta = 1
n_labels=5
latent_size = 125-n_labels
latent_dim = 2
intermediate_dim =512
input_shape = (1000,1)
label_shape=(n_labels,)
def get_data(): 
    # Importing X(ECGS) and Y(labels) data
    Y_real = pd.read_csv('Y_10s_superclass.csv')
    Y_real=np.array(Y_real)
    print(len(Y_real))
    Y_real=Y_real[:1000,:]
    Y_unique=np.unique(Y_real, axis=0)
    
    
    X_real = np.loadtxt('X_10s_1000.csv')
    print(len(X_real))
    X_real = X_real.reshape(X_real.shape[0], 1000, 1)
    X_real = K.cast(X_real, 'float32')
       
    # Converting these to form suitable for TF
    dataset = tf.data.Dataset.from_tensor_slices((X_real,Y_real)).shuffle(buffer_size=1024).batch(batch_size)
    return dataset




def sampling(args):
    """Reparameterization trick by sampling 
        fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def define_encoder(image, labels, latent_dim):
    
    x = layers.Dense(1000)(labels)
    x = layers.Reshape((1000,1))(x)
    #Concatinate labels and data
    x = layers.concatenate([image, x])
    # Any architecture could go here as long as latent space is achieved
    x = layers.Conv1D(filters=32, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=64, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

   

   

    x = layers.Flatten()(x)
    x = layers.Dense(2, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
  
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = keras.Model([image, labels], [z_mean, z_log_var, z], name='encoder')
    return encoder
def define_decoder(latent_inputs, labels, latent_dim):

    inputs = [latent_inputs, labels]
    #Recombine latent with label vectors
    x = layers.concatenate(inputs, axis = 1)
    
    x = layers.Dense(16000)(x)
    x = layers.Reshape((250,64))(x)
    x = layers.Conv1DTranspose(filters=64, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    
    x = layers.Conv1DTranspose(filters=32, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv1DTranspose(filters = 1, kernel_size = 16, strides=1, padding='same', activation='sigmoid', name='decoder_output')(x)
    decoder = keras.Model([latent_inputs, labels], outputs, name = 'decoder')
    return decoder







optimizer = tf.keras.optimizers.Adam(lr = 0.0005)

def mse_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return 1000 * r_loss

def kl_loss(mean, log_var):
    kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
    return kl_loss

def vae_loss(y_true, y_pred, mean, log_var):
    r_loss = mse_loss(y_true, y_pred)
    kl_losses = kl_loss(mean, log_var)
    return  r_loss + kl_losses


@tf.function
def train_step(images):

    with tf.GradientTape() as enc, tf.GradientTape() as dec:
        image , label  = images
        label = K.cast(label, 'float32')
        image = K.cast(image, 'float32')
        z_mean, z_log_var, z = encoder([image, label], training=True)
        
        
        
    
        generated_images = decoder([z, label], training=True)
        print(generated_images)
        reconstruction_loss = keras.losses.mse(K.flatten(image), K.flatten(generated_images))

        reconstruction_loss *= 1000
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5 * beta
        loss = K.mean(reconstruction_loss + kl_loss)
       
        
      
        
        
    gradients_of_enc = enc.gradient(loss, encoder.trainable_variables)
    gradients_of_dec = dec.gradient(loss, decoder.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients_of_enc, encoder.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_dec, decoder.trainable_variables))
    return loss
loss_array = []

def train(dataset, epochs):

  for epoch in range(epochs):

    start = time.time()

    for image_batch in dataset:

      loss = train_step(image_batch)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Loss is = ', float(loss))
    loss_array.append(loss)
    if epoch % 5 == 0:
        c = np.array(range(1000))
        image, label = image_batch
        z_mean, z_log_var, z = encoder([image, label], training=False)
        generated_images = decoder([z, label], training=False)
        x = np.array(generated_images[1])
        c = c.reshape(1000, 1)
        x = x.reshape(1000,1)
        plot_ex = np.array(image[1]).reshape(1000,1)
        X = np.hstack((c, x))
        Y = np.hstack((c,plot_ex))
        fig, axs = plt.subplots(2, figsize=(17, 7))
        axs[0].plot(X[:,0], X[:,1], color = 'blue', lw=1)
        axs[0].plot(Y[:,0], Y[:,1], color = 'red', lw=1)
        axs[0].grid()
        axs[0].set_title('Epoch Number: {}, Label = {}'.format(epoch, label[1]))
        axs[0].set_xlabel('Time interval')
        axs[0].set_ylabel('Normalised data value')
        axs[1].plot(loss_array, color = 'g', label = 'Gen loss')
        axs[1].grid()
        axs[1].legend(loc="upper left")
        axs[1].set_xlabel('Epoch number')
        axs[1].set_ylabel('Loss')
            
        fig.savefig("Images/Images for CVAE/Image_{}".format(epoch))
        print('Saving image to: Images/Images for CVAE/')
        plt.close(fig)
  
image = keras.Input(shape=input_shape, name = 'encoder_input')
labels = keras.Input(shape=label_shape, name='class_labels')
encoder = define_encoder(image, labels, latent_dim)
    
encoder.summary()
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='decoder_input')
    
decoder = define_decoder( latent_inputs , labels,latent_dim)
decoder.summary()
outputs = decoder([encoder([image, labels])[2], labels])  
train(get_data(), 100)



 

