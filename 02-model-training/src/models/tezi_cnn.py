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
)
from tensorflow.keras.models import Model

# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

# Default inputs
# Dictionary with {"Reference-name-of-input": {"len_input": <int>, "len_encoding": <int>}}
INPUTS = OrderedDict(
            input=dict(
                len_input=60,
                len_encoding=4, # hot encoding for A,C,G,T and N
            )
        )

class ModelDNA:
    """Generate an instance of a keras model"""

    def __init__(self, n_output: int, output_layer: str,):
        self.inputs = INPUTS
        self.model_name = MODEL_NAME
        self.n_output = n_output
        self.output_layer = output_layer

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
        
        # Conv layers
        output = Conv1D(1, 19, padding='same', activation='relu')(inputs)
        
        # Dense layers 
        output = Flatten()(output)
        output = Dense(128, kernel_initializer='glorot_normal', activation='relu')(output) 
        output = Dense(64, kernel_initializer='glorot_normal', activation='relu')(output)
        
        # output layer
        output = Dense(self.n_output, activation=self.output_layer, dtype = tf.float32)(output)

        # Create model
        model = Model(inputs = inputs, outputs = output)

        return model