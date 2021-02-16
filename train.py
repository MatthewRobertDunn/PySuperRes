import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import audio


class MyModel:

    def __init__(self) -> None:
       gpus = tf.config.experimental.list_physical_devices('GPU')
       if gpus:
            try:
                for gpu in gpus:             # Currently, memory growth needs to be the same across GPUs
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    def get_model(self):
        conv_args = {
            "activation": "tanh",
            "kernel_initializer": "normal",
            "padding": "same",
        }
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(64, 5, input_shape=(None,None,1), **conv_args))
        model.add(layers.Conv1D(64, 3, **conv_args))
        model.add(layers.Conv1D(32, 3, **conv_args))
        model.add(layers.Conv1D(3, 3, **conv_args))
        return model
        
    
    def load(self):
        self.model = keras.models.load_model("mymodel.h5")
    
    def compile(self):
        self.model = self.get_model()
        self.model.compile(loss=tf.keras.losses.mean_squared_error, optimizer="adam", metrics=['mean_squared_error'])

    def train(self,filename, epochs = 150):
        print(f"training {filename}")
        input = audio.get_input_data(filename)
        output = audio.get_output_data(filename, input.shape[1])
        history = self.model.fit(input,output,epochs=epochs, batch_size=32)
        self.model.save('mymodel.h5')

    def upscale(self, input):
        print("upscaling")
        out = self.model.predict(input)
        return audio.undo_output(out)
