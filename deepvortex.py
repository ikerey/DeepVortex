import numpy as np
import platform
import os
from astropy.io import fits
import time
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'vena'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.layers import Input, Convolution2D, Activation, BatchNormalization, add
from keras.models import Model


class deepvortex(object):

    def __init__(self, observations, output):
        """
        Class used to detect vortex flows from vx and vy images

        Parameters
        ----------
        sample : array
            Array of size (n_times, nx, ny,2) with the n_times consecutive images of size nx x ny
        output : string
            Filename were the output is saved
        """

# Only allocate needed memory with Tensorflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        n_timesteps, nx, ny, nv = observations.shape

        self.n_frames = n_timesteps 

        self.nx = nx 
        self.ny = ny 

        self.nv = nv
        
        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 1
        self.n_conv_layers = 5        
        self.observations = observations
        self.output = output

        print("Images without border are of size: {0}x{1}".format(self.nx, self.ny))
        print("Number of predictions to be made: {0}".format(self.n_frames))

    def residual(self, inputs):
        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, inputs])

        return x
            
    def define_network(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.nv))
        conv = Convolution2D(self.n_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, conv])

        final = Convolution2D(1, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(x)

        self.model = Model(inputs=inputs, outputs=final)
        self.model.load_weights('network/deepvortex_weights.hdf5')

    def normalize11(self, ima):
        """Normalize to the interval [-1, 1]"""
        min_v = np.min(ima)
        max_v = np.max(ima)
        nima = 2*(ima - np.min(ima))/(np.max(ima)-np.min(ima))-1
        return nima
                        
    def validation_generator(self):        

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        #self.observations = self.normalize11(self.observations)

        while 1:
            for i in range(self.n_frames):

                input_validation[:,:,:,0] = self.observations[i,:,:,0]

                input_validation[:,:,:,1] = self.observations[i,:,:,1] 

                yield input_validation

        f.close()

    def predict(self):
        print("Detecting vortices with DeepVortex...")

        start = time.time()
        out = self.model.predict_generator(self.validation_generator(), self.n_frames, max_q_size=1)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))
        
        print("Saving data...")
        hdu = fits.PrimaryHDU(out)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(self.output, overwrite=True)
        
    
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='DeepVortex prediction')
    parser.add_argument('-o','--out', help='Output file')
    parser.add_argument('-i','--in', help='Input file')
    parsed = vars(parser.parse_args())

# Open file with observations and read them. We use FITS in our case
    f = fits.open(parsed['in'])
    imgs = f[0].data   
    
    out = deepvortex(imgs, parsed['out'])
    out.define_network()
    out.predict()
