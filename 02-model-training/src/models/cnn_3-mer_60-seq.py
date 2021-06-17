# built in libraries
from collections import OrderedDict
from pathlib import Path

# third party libraries
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    Flatten, 
    Dropout,
    MaxPooling1D,
    BatchNormalization,
    LeakyReLU,
)
from tensorflow.keras.models import Model

# Reference name of model
k = 3 # k-mer
N = 60  # length original sequence

MODEL_NAME = str(Path(__file__).resolve().stem)

# Default inputs
# Dictionary with {"Reference-name-of-input": {"len_input": <int>, "len_encoding": <int>}}
INPUTS = OrderedDict(
            input=dict(
                len_input=N-k+1, # len of the input sequence
                len_encoding=4**k, # hot encoding for k-mers
            )
        )

class ModelDNA:
    """Generate an instance of a keras model"""

    def __init__(self, n_output: int, output_layer: str,):
        self.inputs = INPUTS
        self.model_name = MODEL_NAME
        self.n_output = n_output
        self.output_layer = output_layer

    @staticmethod
    def base_conv1d_layer(input_conv, filters, kernel_size):
        output = Conv1D(filters, kernel_size, 
                        padding='same',
                        kernel_initializer='glorot_normal'
                        )(input_conv)
        output = LeakyReLU(alpha=0.1)(output)
        output = Dropout(rate=0.3)(output)
        output = MaxPooling1D(2, padding='same')(output)
        return output

    # Load model
    def get_model(self,):
        f"""keras model for {self.model_name}

        Args:
            n_output (int): number of neurons in the last layer
            output_layer (str): activation function of last layer
            shape_inputs (list of tuples): Eg: For two inputs [(100,1),(1000,1)]
        """    
        shape_inputs = [(value.get("len_input"),value.get("len_encoding")) for value in self.inputs.values()]
        
        # Inputs     
        inputs = Input(shape=shape_inputs[0])
        
        # Conv blocks 
        output = self.base_conv1d_layer(inputs, filters=128, kernel_size=7)
        output = self.base_conv1d_layer(output, filters=64, kernel_size=7)
        output = self.base_conv1d_layer(output, filters=32, kernel_size=7)
        
        # Dense layers 
        output = Flatten()(output)
        output = Dense(128, kernel_initializer='glorot_normal')(output)
        output = LeakyReLU(alpha=0.1)(output)
        output = Dropout(rate=0.3)(output)
        
        # output layer
        output = Dense(self.n_output, activation=self.output_layer, dtype = tf.float32)(output)

        # Create model
        model = Model(inputs = [inputs], outputs = output)

        return model