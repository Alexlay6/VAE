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

    Y_real=Y_real[:1000,:]
    Y_unique=np.unique(Y_real, axis=0)
    
    
    X_real = np.loadtxt('X_10s_1000.csv')
 
    X_real = X_real.reshape(X_real.shape[0], 1000, 1)
    X_real = K.cast(X_real, 'float32')
       
    # Converting these to form suitable for TF
    dataset = tf.data.Dataset.from_tensor_slices((X_real,Y_real)).shuffle(buffer_size=1024).batch(batch_size)
    return dataset, X_real, Y_real




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

def define_encoder(latent_dim):
    image = keras.Input(shape=input_shape, name = 'encoder_input')
    labels = keras.Input(shape=label_shape, name='class_labels')
    x = layers.Dense(1000)(labels)
    x = layers.Reshape((1000,1))(x)
    #Concatinate labels and data
    x = layers.concatenate([image, x])
    # Any architecture could go here as long as latent space is achieved
    
    
    x = layers.Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(image)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=256, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPool1D(pool_size=2)(x)



    x = layers.Flatten()(x)
    x = layers.Dense(2, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
  
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = keras.Model([image, labels], [z_mean, z_log_var, z], name='encoder')
    return encoder
def define_decoder(latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='decoder_input')
    labels = keras.Input(shape=label_shape, name='class_labels')
    inputs = [latent_inputs, labels]
    #Recombine latent with label vectors
    x = layers.concatenate(inputs, axis = 1)
    
    
    
    x = layers.Dense(16000)(x)
    x = layers.Reshape((250,64))(x)
    

    x = layers.Conv1DTranspose(filters=128, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
  
    x = layers.Conv1DTranspose(filters=64, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    
    x = layers.Conv1DTranspose(filters=32, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1DTranspose(filters=16, kernel_size=16, strides=1, padding='same')(x)
    x = layers.ReLU()(x)

    
    x = layers.Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='sigmoid')(x)
    outputs = layers.Conv1DTranspose(filters = 1, kernel_size = 16, strides=1, padding='same', activation='sigmoid', name='decoder_output')(x)
    decoder = keras.Model([latent_inputs, labels], outputs, name = 'decoder')
    return decoder



class CVAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def compile(self, e_optimizer, d_optimizer):
        super(CVAE, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.e_optimizer = e_optimizer
        
    def train_step(self, data):
        '''
        We override the models train step to implement a custom loss function.
         
        '''
        
        with tf.GradientTape() as enc, tf.GradientTape() as dec:
            image , label  = data
            label = K.cast(label, 'float32')
            image = K.cast(image, 'float32')
            z_mean, z_log_var, z = self.encoder([image, label], training=True)
            
            
            
        
            generated_images = self.decoder([z, label], training=True)
         
            reconstruction_loss = keras.losses.mse(K.flatten(image), K.flatten(generated_images))
    
            reconstruction_loss *= 1000
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5 * beta
            loss = K.mean(reconstruction_loss + kl_loss)
           
            
          
            
            
        gradients_of_enc = enc.gradient(loss, self.encoder.trainable_variables)
        gradients_of_dec = dec.gradient(loss, self.decoder.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients_of_enc, self.encoder.trainable_variables))
        optimizer.apply_gradients(zip(gradients_of_dec, self.decoder.trainable_variables))
       
        
        
        self.loss_tracker.update_state(loss)
        
            
        return {
          "loss": self.loss_tracker.result(),

        }
loss_array=[]
dataset, X_real, Y_real = get_data()  

class Save_plot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        
        # Updating loss arrays for plotting
        if epoch%1==0:
            
            
            c = np.array(range(1000))
            image = X_real
            label = Y_real
            z_mean, z_log_var, z = self.model.encoder([image, label], training=False)
            generated_images = self.model.decoder([z, label], training=False)
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

        if epoch%1000==0:
            self.model.decoder.save('Saved Models/CVAE/CVAE-decoder{}.h5'.format(epoch))
            self.model.encoder.save('Saved Models/CVAE/CVAE-encoder{}.h5'.format(epoch))
# Initialising the callback
plotter = Save_plot()


optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
cvae = CVAE(
    encoder=define_encoder(latent_dim),
    decoder=define_decoder(latent_dim),
    latent_dim=2,
)

define_encoder(latent_dim).summary()
define_decoder(latent_dim).summary()

cvae.compile(
    d_optimizer=optimizer,
    e_optimizer=optimizer)


  
cvae.fit(dataset, epochs=3000, callbacks = [plotter])

cvae.generator.save('Saved Models/AC-WGAN/AC-WGAN-Final.h5')







 

