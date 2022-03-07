# -*- coding: utf-8 -*-
"""
This code was adapted fromn Jason Brownlee: 
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

this is code for regularization on the generation by adding the divergence to the GAN loss
"""

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.models import load_model
import tensorflow as tf

# load, split and scale the maps dataset ready for training
from numpy import asarray
from numpy import savez_compressed
from numpy import mean, sqrt, square
import numpy as np
from math import pi

graph_path = r'/users/PAS1064/lenau1/Machine_Learning/stress_equil//'           # where you want loss and RMS(Divergence) information/graphs to be saved
dataset_path = r'/users/PAS1064/lenau1/Machine_Learning/stress_equil//'         # path where dataset is
model_path = r'/users/PAS1064/lenau1/Machine_Learning/stress_equil//'           # where you want models to be saved

#calculate divergence through tensor flow tensorflow
def calc_divergence(input):
    i = input
    print('input shape:', i.shape)
    S1 = i[:,:,:,0]
    S12 = i[:,:,:,1]
    S2 = i[:,:,:,2]
    print('S1 shape:', S1.shape)
    print('S12 shape:', S12.shape)
    
    
    #Take fourier transform of sress directions
    fsig11 = (tf.signal.fft2d(tf.cast(S1, dtype=tf.complex64)))
    fsig12 = (tf.signal.fft2d(tf.cast(S12, dtype=tf.complex64)))
    fsig22 = (tf.signal.fft2d(tf.cast(S2, dtype=tf.complex64)))
    
    Nx = 128 # shape of input images
    a = 0
    b = Nx-1
    
    x1 = tf.linspace(tf.cast(0, dtype=tf.float32), (Nx/2)-1, num=int(64))
    x2 = tf.linspace(-(Nx/2)+1, -1, num=int((Nx/2)-1))
    x0 = tf.linspace(tf.cast(0, dtype=tf.float32), tf.cast(0, dtype=tf.float32), num=1)
    x = tf.concat([x1, x0, x2], axis=0)
    # print('xshape = ',x)
    
    y1 = tf.linspace(tf.cast(0, dtype=tf.float32), (Nx/2)-1, num=int(64))
    y2 = tf.linspace(-(Nx/2)+1, -1, num=int((Nx/2)-1))
    y0 = tf.linspace(tf.cast(0, dtype=tf.float32), tf.cast(0, dtype=tf.float32), num=1)
    y = tf.concat([y1, y0, y2], axis=0)
    # print('yshape = ', y)
        
    [X, Y] = tf.meshgrid((2*pi/(b-a))*x, (2*pi/(b-a))*y)
    # print('X shape:', X.shape)
    # print('Y shape:', Y.shape)    
    
    dfdx11 = tf.signal.ifft2d(1j*tf.cast(X, dtype=tf.complex64)*fsig11)
    dfdy12 = tf.signal.ifft2d(1j*tf.cast(Y, dtype=tf.complex64)*fsig12)
    dfdx12 = tf.signal.ifft2d(1j*tf.cast(X, dtype=tf.complex64)*fsig12)
    dfdy22 = tf.signal.ifft2d(1j*tf.cast(Y, dtype=tf.complex64)*fsig22)
    
    # Calulate Divergence
    Div1 = dfdx11 + dfdy12
    Div2 = dfdx12 + dfdy22
    
    Div1 = tf.math.real(Div1)
    Div2 = tf.math.real(Div2)
   
    return (Div1, Div2)

image_shape = (128,128,3)
############################# DISCRIMINATOR ##################################
# define the discriminator model
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
#print('patch shape B4:', patch_out_hidden.shape)
d_model = Model([in_src_image, in_target_image], patch_out_hidden)
optd = Adam(lr=0.0001, beta_1=0.5)
d_model.compile(loss='binary_crossentropy', optimizer=optd, loss_weights=[0.5])

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
    print('out_image_shape:', out_image.shape)
    # define model
    model = Model(in_image, out_image)
    return model


def g_stress_loss_fn(y_true, y_pred):
    # print('y_true shape:', y_true.shape)
    # print('y_pred shape:', y_pred.shape)
    div_pred = calc_divergence(y_pred)
    L = 100                                                                     # weight of L1 loss term
    LL = 100                                                                    # weight of adding divergence regularization term
    # print('y_true shape:', y_true.shape)
    # print('y_pred shape:', y_pred.shape)
    loss0 = tf.keras.losses.mean_absolute_error(y_true, y_pred)                 # L1 loss term
    # print('loss0 shape:', loss0.shape)

    lossi = (div_pred[0]) 
    lossii = (div_pred[1])
    loss1 = lossi + lossii
    loss = L*loss0 + LL*loss1
    return loss

def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, image_shape):
	# make weights in the discriminator not trainable
    make_trainable(d_model,False)
	# define the source image
    in_src = Input(shape=image_shape)
    print('in_src_img:', in_src.shape)
	# connect the source image to the generator input
    gen_out = g_model(in_src)
    print('gen out shape', gen_out.shape)
	# connect the source input and generator output to the discriminator input
    G_merged = Concatenate()([in_src, gen_out])
    gan1 = d1(G_merged)
    gan2 = d1a(gan1)
    gan3 = d2(gan2)
    gan4 = d2a(gan3)
    gan5 = d2b(gan4)
    gan6 = d3(gan5)
    gan7 = d3a(gan6)
    gan8 = d3b(gan7)
    gan9 = d4(gan8)
    gan10 = d4a(gan9)
    gan11 = d4b(gan10)
    gan12 = d5(gan11)
    gan13 = d5a(gan12)
    gan14 = d5b(gan13)
    gan15 = d6(gan14)
    gan_out = patch_out(gan15)
    # src image as input (in_src), generated image (gen_out) and classification output (gan_out)
    model = Model([in_src], [gan_out, gen_out])
	# compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', g_stress_loss_fn], optimizer=opt, loss_weights=[1,1])
    return model


# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale source from [0,255] to [-1,1]
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
    #print('shapeX2:', X2.shape)
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
    X = g_model.predict(samples)
	# create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    #print('fake shape X, y:', X.shape)
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, gan_model, dataset, n_samples=3):
    # # select a sample of input images
    # [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # # generate a batch of fake samples
    # X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # # scale all pixels from [-1,1] to [0,1] for source image
    # X_realA = (X_realA + 1) / 2.0
    
	# save the generator model
    filename1 = 'GenModel_%06d.h5' % (step+1)
    g_model.save(model_path + filename1)
    filename2 = 'DModel_%06d.h5' % (step+1)
    d_model.save(model_path + filename2)
    # filename3 = 'GanModel_%06d.h5' % (step+1)
    # gan_model.save(filename3)
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
    print('fsig11 shape', fsig11.shape)
    
    Nx = 128 # shape of input images
    a = 0
    b = Nx-1

    x1 = np.linspace(0, (Nx/2)-1, num=int(Nx/2))
    x2 = np.linspace(-(Nx/2)+1, -1, num=int(Nx/2-1))
    x = np.concatenate((np.asarray(x1), np.asarray(x2)))
    x = np.insert(x, 64, 0)

    y1 = np.linspace(0, (Nx/2)-1, num=int(Nx/2))
    y2 = np.linspace(-(Nx/2)+1, -1, num=int(Nx/2-1))
    y = np.concatenate((np.asarray(y1), np.asarray(y2)))
    y = np.insert(y, 64, 0)
    
    [X, Y] = np.meshgrid((2*pi/(b-a))*x, (2*pi/(b-a))*y)
    
    dfdx11 = np.fft.ifft2(1j*X*fsig11)
    dfdy12 = np.fft.ifft2(1j*Y*fsig12)
    dfdx12 = np.fft.ifft2(1j*X*fsig12)
    dfdy22 = np.fft.ifft2(1j*Y*fsig22)
    
    Div1 = dfdx11 + dfdy12
    Div2 = dfdx12 + dfdy22
    
    Div1 = np.real(Div1)
    Div2 = np.real(Div2)
    
    return Div1, Div2

loss_list = list()
RMS = list()
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    print('patch shape:', d_model.output_shape[0])
    print('patch shape:', n_patch)
	# unpack dataset
    trainA, trainB = dataset
	# calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    # Load models or trasfer learning:
    #model_path = r'/users/PAS1064/lenau1/Machine_Learning/stress_equil/higher_E_contrast/No_stress_term_half//'
    ### load old models
    #old_g_model = load_model(model_path + 'GenModel_024600.h5')
    #old_d_model = load_model(model_path + 'DModel_024600.h5')
    #old_gan_model = load_model(model_path + 'GanModel_024600.h5')
    ### set weights from old models
    #g_model.set_weights(old_g_model.get_weights())
    #d_model.set_weights(old_d_model.get_weights())
    #gan_model.set_weights(old_gan_model.get_weights())
    #print('All Weight Set!------------')
	
    # manually enumerate epochs
    for i in range(n_steps):
		# select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        #print('X_realA, X_realB, y_real:', X_realA.shape, X_realB.shape, y_real.shape)
        #print('TYPES: X_realA, X_realB, y_real:', X_realA.dtype, X_realB.dtype, y_real.dtype)
        
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # print('X_fakeB, y_fake:', X_fakeB.dtype, y_fake.dtype)
        
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)            # discriminator loss with real samples
		# update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)            # discriminator loss with generated samples
		# update the generator
        g_loss, _, _ = gan_model.train_on_batch([X_realA], [y_real, X_realB])   # GAN loss
		
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
        if (i+1) % (1000) == 0:    #bat_per_epo * 10
            summarize_performance(i, g_model, d_model, gan_model, dataset) 
        if (i+1) == n_steps:
             summarize_performance(i, g_model, d_model, gan_model, dataset) 
        # make graphs
        if (i+1) % (10) ==0:
           losses = ((i+1), d_loss1, d_loss2, g_loss)
           loss_list.append(losses)
           [Rdiv1, Rdiv2] = numpy_calc_div(X_realB)
           [Fdiv1, Fdiv2] = numpy_calc_div(X_fakeB)
           rmss = ((i+1), sqrt(mean(square(Rdiv1))), sqrt(mean(square(Rdiv2))), 
                   sqrt(mean(square(Fdiv1))), sqrt(mean(square(Fdiv2))))
           RMS.append((rmss))
        
        # save models every 1000 iterations
        if (i+1)%1000==0:           
            savez_compressed(graph_path + 'loss_data.npz', asarray(loss_list))
            savez_compressed(graph_path + 'RMS_data.npz', asarray(RMS))
            print('SAVED-----------------------------------------------------')
            
# load image data
dataset = load_real_samples(dataset_path + 'higher_E_contrast_train.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
#image_shape = dataset[0].shape[1:]
image_shape = (128,128,3)
# define the models
# d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)

import matplotlib.pyplot as plt
path = graph_path
file1 = 'loss_data.npz'
file2 = 'RMS_data.npz'

loss = np.load(path+file1)
rms = np.load(path+file2)

loss = loss['arr_0']
rms = rms['arr_0']

# loss graph
iteration = loss[:,0]
d1 = loss[:,1]
d2 = loss[:,2]
g = loss[:,3]

plt.figure()
plt.plot(iteration, d1)
plt.xlabel('Iteration')
plt.ylabel('d1 Loss')
plt.savefig(path+'d1_plot.png')
plt.show()

plt.figure()
plt.plot(iteration, d2)
plt.xlabel('Iteration')
plt.ylabel('d2 Loss')
plt.savefig(path+'d2_plot.png')
plt.show()

plt.figure()
plt.plot(iteration, g)
plt.xlabel('Iteration')
plt.ylabel('G Loss')
plt.savefig(path+'g_plot.png')
plt.show()


# RMS Graphs
itera = rms[:,0]
rdiv1 = rms[:,1]
rdiv2 = rms[:,2]
fdiv1 = rms[:,3]
fdiv2 = rms[:,4]

plt.figure()
plt.plot(iteration, rdiv1)
plt.xlabel('Iteration')
plt.ylabel('Real Divergence 1')
plt.savefig(path+'rdiv1.png')
plt.show()

plt.figure()
plt.plot(iteration, rdiv2)
plt.xlabel('Iteration')
plt.ylabel('Real Divergence 2')
plt.savefig(path+'rdiv2.png')
plt.show()

plt.figure()
plt.plot(iteration, fdiv1)
plt.xlabel('Iteration')
plt.ylabel('Fake Divergence 1')
plt.savefig(path+'fdiv1.png')
plt.show()

plt.figure()
plt.plot(iteration, fdiv2)
plt.xlabel('Iteration')
plt.ylabel('Fake Divergence 2')
plt.savefig(path+'fdiv2.png')
plt.show()
