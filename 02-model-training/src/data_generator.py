
from typing import Callable, Optional, List
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    "Data Generator for DNA sequences"
    def __init__(self,
                    sequences,
                    labels,
                    encoder_input,  # from dna-sequence to array
                    encoder_output, # from label to hot-encode
                    # TODO batch_creator, # from list of arrays to batch for the specific model
                    preprocessing: Optional[Callable] = None, 
                    augmentation: Optional[Callable] = None,
                    shuffle: bool = True,
                    batch_size: int = 32,
            ):

        self.sequences      = sequences
        self.labels         = labels
        self.preprocessing  = preprocessing
        self.augmentation   = augmentation
        self.encoder_input  = encoder_input
        self.encoder_output = encoder_output
        self.shuffle        = shuffle
        self.batch_size     = batch_size

        # Initialize first batch
        self.on_epoch_end()


    def on_epoch_end(self,):
        """Updates indexes after each epoch (starting for the epoch '0')"""
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle indexes in place

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        "Generates one batch of data to train/test the model"
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of sequences for the batch
        sequences_batch = [self.sequences[k] for k in indexes]
        labels_batch = [self.labels[k] for k in indexes] 

        # Input batch (sequences as array)
        X = np.array([self.encoder_input(seq) for seq in sequences_batch])
        
        # Output batch (labels)
        y = np.stack([self.encoder_output([label]) for label in labels_batch], axis=0)
        
        return X, y