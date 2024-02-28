# -*- coding: utf-8 -*-
"""
This code was adapted fromn Jason Brownlee: 
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

Ashley Lenau

This is code for regularization on the generation by adding the divergence to the GAN loss
"""
# from keras import backend as K
from numpy import load
import os
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import matplotlib.pyplot as plt
# load, split and scale the maps dataset ready for training
from numpy import asarray
from numpy import savez_compressed
from numpy import mean, sqrt, square
import numpy as np
from math import pi
graph_path = 'graphs//'           # where you want loss and RMS(Divergence) information/graphs to be saved
dataset_path = r'/fs/ess/PAS2405/a1lenau/full_and_half_datasets/large_train_high_E_contrast.npz'         # path where dataset is
model_path = 'models//'           # where you want models to be saved
os.makedirs(graph_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
save_npz_name = 'Sigmoid_DLR=0.00003_GLR=0.0002_Dwt=1_MSE=700'

test_dataset = r'/fs/ess/PAS2405/a1lenau/full_and_half_datasets/large_val_high_E_contrast.npz'

#calculate divergence through tensor flow tensorflow
def calc_divergence(input):
    # CP-FFT calculates divegrence in FFT space, so do tht here as well 
    i = input
    print('input shape:', i.shape)
    S1 = i[:,:,:,0]
    S12 = i[:,:,:,1]
    S2 = i[:,:,:,2]
    
    #Take fourier transform of sress directions
    fsig11 = (tf.signal.fft2d(tf.cast(S1, dtype=tf.complex128)))
    fsig12 = (tf.signal.fft2d(tf.cast(S12, dtype=tf.complex128)))
    fsig22 = (tf.signal.fft2d(tf.cast(S2, dtype=tf.complex128)))
    
    Nx = 128 # shape of input images
    a = 0
    b = Nx-1
    
    x1 = tf.linspace(tf.cast(0, dtype=tf.float64), (Nx/2)-1, num=int(64))
    x2 = tf.linspace(-(Nx/2)+1, -1, num=int((Nx/2)-1))
    x0 = tf.linspace(tf.cast(-64, dtype=tf.float64), tf.cast(0, dtype=tf.float64), num=1)
    x = tf.concat([x1, x0, tf.cast(x2, dtype=tf.float64)], axis=0)
    
    y1 = tf.linspace(tf.cast(0, dtype=tf.float64), (Nx/2)-1, num=int(64))
    y2 = tf.linspace(-(Nx/2)+1, -1, num=int((Nx/2)-1))
    y0 = tf.linspace(tf.cast(-64, dtype=tf.float64), tf.cast(0, dtype=tf.float64), num=1)
    y = tf.concat([y1, y0, tf.cast(y2, dtype=tf.float64)], axis=0)
        
    [X, Y] = tf.meshgrid((2*pi/(b-a))*x, (2*pi/(b-a))*y)
    
    dfdx11 = tf.signal.ifft2d(1j*tf.cast(X, dtype=tf.complex128)*fsig11)
    dfdy12 = tf.signal.ifft2d(1j*tf.cast(Y, dtype=tf.complex128)*fsig12)
    dfdx12 = tf.signal.ifft2d(1j*tf.cast(X, dtype=tf.complex128)*fsig12)
    dfdy22 = tf.signal.ifft2d(1j*tf.cast(Y, dtype=tf.complex128)*fsig22)
    
    # Calulate Divergence
    Div1 = dfdx11 + dfdy12
    Div2 = dfdx12 + dfdy22
    
    Div1 = tf.cast(tf.math.real(Div1), dtype=tf.float32)
    Div2 = tf.cast(tf.math.real(Div2), dtype=tf.float32)
   
    return (Div1, Div2)

def sig_expo_fn(stress_fields):
    div_pred = calc_divergence(stress_fields)
    rms0 = tf.math.sqrt(tf.reduce_mean(tf.math.square(div_pred[0])))
    rms1 = tf.math.sqrt(tf.reduce_mean(tf.math.square(div_pred[1])))
    h = 10
    expo0 = (tf.math.log((rms0))/tf.math.log(tf.cast(h, dtype=tf.float32)))
    expo1 = (tf.math.log((rms1))/tf.math.log(tf.cast(h, dtype=tf.float32)))
    T = 1                                                                       # change T to change shape of sigmoid function
    Da = -((1/(1+tf.math.exp(-T*expo0)))-0.5)*2
    Db = -((1/(1+tf.math.exp(-T*expo1)))-0.5)*2
    out = Da*Db
    out = tf.repeat([out], repeats=[64], axis=0)
    out = tf.reshape(out,[-1,8,8,1])
    return out

#### DISCRIMINATOR ####
# define the discriminator model
def define_discrim(image_shape = (128,128,3)):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape, name='comp_image')
    in_target_image = Input(shape=(128,128,3), name='Stress')
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)
    d1_hidden = d1(merged)
    d1a = LeakyReLU(alpha=0.2)
    d1a_hidden = d1a(d1_hidden)
    # C128
    d2 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)
    d2_hidden = d2(d1a_hidden)
    d2a = BatchNormalization()
    d2a_hidden = d2a(d2_hidden)
    d2b = LeakyReLU(alpha=0.2)
    d2b_hidden = d2b(d2a_hidden)
    # C256
    d3 = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)
    d3_hidden = d3(d2b_hidden)
    d3a = BatchNormalization()
    d3a_hidden = d3a(d3_hidden)
    d3b = LeakyReLU(alpha=0.2)
    d3b_hidden = d3b(d3a_hidden)
    # C512
    d4 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)
    d4_hidden = d4(d3b_hidden)
    d4a = BatchNormalization()
    d4a_hidden = d4a(d4_hidden)
    d4b = LeakyReLU(alpha=0.2)
    d4b_hidden = d4b(d4a_hidden)
    # second last output layer
    d5 = Conv2D(512, (4,4), padding='same', kernel_initializer=init)
    d5_hidden = d5(d4b_hidden)
    d5a = BatchNormalization()
    d5a_hidden = d5a(d5_hidden)
    d5b = LeakyReLU(alpha=0.2) 
    d5b_hidden = d5b(d5a_hidden)
    # patch output
    d6 = Conv2D(1, (4,4), padding='same', kernel_initializer=init)
    dout = d6(d5b_hidden)
    patch_out = Activation('sigmoid')
    patch_out_hidden = patch_out(dout)
    ### sigmoid regualrization ###
    div_prob = sig_expo_fn(in_target_image)
    patch_out_new = tf.keras.layers.multiply([div_prob,patch_out_hidden])
    d_model = Model([in_src_image, in_target_image], patch_out_new)
    optd = Adam(learning_rate=0.00003, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=optd, loss_weights=[1])
    return d_model

# define an encoder block for G
def define_encoder_block(layer_in, n_filters, f_size=(4,4), batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, f_size, strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block for G
def decoder_block(layer_in, skip_in, n_filters, f_size=(4,4), dropout=True):
	# weight initialization
    init = RandomNormal(stddev=0.02)
	# add upsampling layer
    g = Conv2DTranspose(n_filters, f_size, strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
        
	# merge with skip connection
    g = Concatenate()([g, skip_in])
	# relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape=(128,128,3)):
	# weight initialization
    init = RandomNormal(stddev=0.02)
	# image input
    in_image = Input(shape=image_shape)
	# encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    
	# bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
    b = Activation('relu')(b)
    
    # decoder model
    d2 = decoder_block(b, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, image_shape):
    # make weights in the discriminator not trainable expect when batch norm is applied (so you can get the batchstatistics)
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
	# define the source image
    in_src = Input(shape=image_shape)
	# connect the source image to the generator input
    gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
    model = Model([in_src], [dis_out, gen_out])
	# compile model
    optg = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mse'], optimizer=optg, loss_weights=[1,700])
    return model


# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
    trainA, trainB = dataset
	# choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
    X = g_model.predict(samples)
	# create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# save the model
def summarize_performance(step, g_model):
	# save the generator model
    filename1 = 'GenModel_%06d.h5' % (step+1)
    g_model.save(model_path + filename1)
    print('>Saved')

# calculate divergence using numpy
def numpy_calc_div(inp):
    i = inp
    S1 = i[:,:,:,0]
    S12 = i[:,:,:,1]
    S2 = i[:,:,:,2]
    
    ## take fft of the files
    fsig11 = (np.fft.fft2(np.squeeze(S1)))
    fsig12 = (np.fft.fft2(np.squeeze(S12)))
    fsig22 = (np.fft.fft2(np.squeeze(S2)))
    
    Nx = 128 # shape of input images
    a = 0
    b = Nx-1

    x1 = np.linspace(0, (Nx/2)-1, num=int(Nx/2))
    x2 = np.linspace(-(Nx/2)+1, -1, num=int(Nx/2-1))
    x = np.concatenate((np.asarray(x1), np.asarray(x2)))
    x = np.insert(x, 64, -64)

    y1 = np.linspace(0, (Nx/2)-1, num=int(Nx/2))
    y2 = np.linspace(-(Nx/2)+1, -1, num=int(Nx/2-1))
    y = np.concatenate((np.asarray(y1), np.asarray(y2)))
    y = np.insert(y, 64, -64)
    
    [X, Y] = np.meshgrid((2*pi/(b-a))*x, (2*pi/(b-a))*y)
    
    dfdx11 = np.fft.ifft2(1j*X*fsig11)
    dfdy12 = np.fft.ifft2(1j*Y*fsig12)
    dfdx12 = np.fft.ifft2(1j*X*fsig12)
    dfdy22 = np.fft.ifft2(1j*Y*fsig22)
    
    Div1 = dfdx11 + dfdy12
    Div2 = dfdx12 + dfdy22
    
    Div1 = np.real(Div1)
    Div2 = np.real(Div2)
    
    return Div1.astype('float32'), Div2.astype('float32')

def calc_mse(true, gen, shape=(128,128,3)):
    mse = np.sum(np.square(true - gen))/(shape[0]*shape[1]*shape[2])
    return mse

loss_list = list()
times = list()
mse = list()
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
	# unpack dataset
    trainA, trainB = dataset
	# calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
	
    # manually enumerate epochs
    for i in range(n_steps):
        start = time.time()
		
		# select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)            # discriminator loss with real samples
		# update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)            # discriminator loss with generated samples
		# update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_realA], [y_real, X_realB])   # G loss
        stop = time.time()
        duration = stop-start
        
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f] time[%.3f]' % (i+1, d_loss1, d_loss2, g_loss, duration))
        tea_time = ((i+1),duration)
        times.append(tea_time)
		
        # make graphs
        losses = ((i+1), d_loss1, d_loss2, g_loss, calc_mse(X_realB, X_fakeB))
        loss_list.append(losses)
        
        # save models every 1000 iterations and save stats to npz files to plot later
        if (i+1)%1000==0:           
            savez_compressed(graph_path + 'loss_data.npz', asarray(loss_list))
            savez_compressed(graph_path + 'time4eachIteration.npz', asarray(times))
            summarize_performance(i, g_model) 
        if (i+1)==n_steps:
            savez_compressed(graph_path + 'loss_data.npz', asarray(loss_list))
            savez_compressed(graph_path + 'time4eachIteration.npz', asarray(times))
            summarize_performance(i, g_model)
        
        
# load image data
dataset = load_real_samples(dataset_path)
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = (128,128,3)
# define the models
d_model = define_discrim(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)



#################### PLOTTING ##################################

# Moving average
def get_MA(loss, hyp_param, period):
    temp = []
    moving_avgs = []
    for j in range(len(loss)):
        if j%period == 0:
            one_avg = np.sum(np.asarray(temp))/period
            moving_avgs.append(one_avg)
            temp = []
        else:
            temp.append(loss[j])

    hp_temp = []
    hp_MA = []
    for j in range(len(hyp_param)):
        if j%period == 0:
            one_avg_hp = np.sum(np.asarray(hp_temp))/period
            hp_MA.append(one_avg_hp)
            hp_temp = []
        else:
            hp_temp.append(hyp_param[j])
    return moving_avgs, hp_MA


loss = np.load(graph_path+'loss_data.npz')['arr_0']
iteration = loss[:,0]
d1 = loss[:,1]
d2 = loss[:,2]
g = loss[:,3]
Mse = loss[:,4]

plt.figure('G loss')
plt.plot(iteration, g)
plt.xlabel('Iteration')
plt.ylabel('G Loss')
plt.savefig(graph_path + 'gloss_iter')
plt.show()

plt.figure('D(real) loss')
plt.plot(iteration, d1)
plt.xlabel('Iteration')
plt.ylabel('D1 Loss')
plt.savefig(graph_path + 'd1loss_iter')
plt.show()

plt.figure('D(fake) loss')
plt.plot(iteration, d2)
plt.xlabel('Iteration')
plt.ylabel('D2 Loss')
plt.savefig(graph_path + 'd2loss_iter')
plt.show()

plt.figure('MSE')
plt.plot(iteration, Mse)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.savefig(graph_path + 'MSE_iter')
plt.show()

gma = get_MA(g, iteration, 200)
plt.figure('G loss')
plt.plot(gma[1], gma[0])
plt.xlabel('Iterations')
plt.ylabel('G Loss')
plt.savefig(graph_path + 'gloss_MA')
plt.show()

d1ma = get_MA(d1, iteration, 200)
plt.figure('D(real) loss')
plt.plot(d1ma[1], d1ma[0])
plt.xlabel('Iterations')
plt.ylabel('D1 Loss')
plt.savefig(graph_path + 'd1loss_MA')
plt.show()

d2ma = get_MA(d2, iteration, 200)
plt.figure('D(fake) loss')
plt.plot(d2ma[1], d2ma[0])
plt.xlabel('Iterations')
plt.ylabel('D2 Loss')
plt.savefig(graph_path + 'd2loss_MA')
plt.show()

msema = get_MA(Mse, iteration, 200)
plt.figure('MSE')
plt.plot(msema[1], msema[0])
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.savefig(graph_path + 'MSE_MA')
plt.show()

import testing_funcs
npz_path = save_npz_name + '//'
os.makedirs(npz_path, exist_ok=True)

testing_funcs.make_npz(model_path, np.load(test_dataset), npz_path, save_npz_name)


