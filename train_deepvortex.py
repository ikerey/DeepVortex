import h5py
import os
import json
import argparse
from contextlib import redirect_stdout

os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Convolution2D, Activation, BatchNormalization, add
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as kerasPlot
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

class LossHistory(Callback):
    def __init__(self, root, losses):
        self.root = root        
        self.losses = losses

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs)
        with open("{0}_loss.json".format(self.root), 'w') as f:
            json.dump(self.losses, f)

    def finalize(self):
        pass

class train_deepvortex(object):

    def __init__(self, root, noise, option):
        """
        Class used to train DeepVortex

        Parameters
        ----------
        option : string
            Indicates what needs to be done (start or continue)
        epochs : integer
            Number of epochs
        root : string
            Name of the output files. Some extensions will be added for different files (weights, loss, etc.)
        noise : float
            Noise standard deviation to be added during training. This helps avoid overfitting and
            makes the training more robust
        """

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = root
        self.option = option

        self.n_filters = 64
        self.kernel_size = 3        
        self.batch_size = 32
        self.n_conv_layers = 5
        
        self.input_file_velocity_training = "/DeepVortex/database/database_velocity_training.h5"
        self.input_file_mask_training = "/DeepVortex/database/database_mask_training.h5"

        self.input_file_velocity_validation = "/DeepVortex/database/database_velocity_validation.h5"
        self.input_file_mask_validation = "/DeepVortex/database/database_mask_validation.h5"

        f = h5py.File(self.input_file_velocity_training, 'r')
        self.n_training_orig, self.nx, self.ny, self.n_times = f.get("velocity").shape        
        f.close()

        f = h5py.File(self.input_file_velocity_validation, 'r')
        self.n_validation_orig, _, _, _ = f.get("velocity").shape        
        f.close()
        
        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)

        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        print("Original training set size: {0}".format(self.n_training_orig))
        print("   - Final training set size: {0}".format(self.n_training))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_training))

        print("Original validation set size: {0}".format(self.n_validation_orig))
        print("   - Final validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))

    def training_generator(self):
        f_velocity = h5py.File(self.input_file_velocity_training, 'r')
        velocity = f_velocity.get("velocity")

        f_mask = h5py.File(self.input_file_mask_training, 'r')
        mask = f_mask.get("mask")

        while 1:        
            for i in range(self.batchs_per_epoch_training):

                input_train = velocity[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')
                output_train = mask[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')

                yield input_train, output_train

        f_velocity.close()
        f_mask.close()

    def validation_generator(self):
        f_velocity = h5py.File(self.input_file_velocity_validation, 'r')
        velocity = f_velocity.get("velocity")

        f_mask = h5py.File(self.input_file_mask_validation, 'r')
        mask = f_mask.get("mask")
        
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                input_validation = velocity[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')
                output_validation = mask[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')

                yield input_validation, output_validation

        f_velocity.close()
        f_mask.close()

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

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv = Convolution2D(self.n_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, conv])

        final = Convolution2D(1, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(x)

        self.model = Model(inputs=inputs, outputs=final)
                
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        #kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))
        
    def read_network(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def train(self, n_iterations):
        print("Training network...")        
        
# Recover losses from previous run
        if (self.option == 'continue'):
            with open("{0}_loss.json".format(self.root), 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, losses)
        
        self.metrics = self.model.fit_generator(self.training_generator(), self.batchs_per_epoch_training, epochs=n_iterations, 
            callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), validation_steps=self.batchs_per_epoch_validation)

        self.history.finalize()

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train DeepVortex')
    parser.add_argument('-o','--out', help='Output files')
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n','--noise', help='Noise to add during training', default=0.0)
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
    parsed = vars(parser.parse_args())

    root = parsed['out']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    noise = parsed['noise']

    out = train_deepvortex(root, noise, option)

    if (option == 'start'):           
        out.define_network()        
        
    if (option == 'continue'):
        out.read_network()

    if (option == 'start' or option == 'continue'):
        out.compile_network()
        out.train(nEpochs)
