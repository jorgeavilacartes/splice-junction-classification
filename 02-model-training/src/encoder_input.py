"""
Generate encoders for sequences
"""
from pathlib import Path
import numpy as np
from collections import OrderedDict
from itertools import product
from gensim.models import KeyedVectors

BASE_DIR = Path(__file__).resolve().parent

class HotEncoder:

    # encoding by nb
    ENCODING = OrderedDict({    
        'A': [1,0,0,0],
        'C': [0,1,0,0],
        'G': [0,0,1,0],
        'T': [0,0,0,1],
        'N': [0,0,0,0]  # aux
    })
    
    def __call__(self, sequence): 
        "Given a sequence, returns an array of dimensions (len_array, 4)"
        return np.array(list(map(lambda nb: self.ENCODING.get(nb), sequence)))

class HotEncoderKmer:
    """Encoding kmers as hot-encode vectors."""

    # TODO include stride
    def __init__(self, k, ): # stride: int = 1,): 
        self.k = k # k-mer to build embeddings
        # self.stride = stride # chars to skip between k-mers in the sequence for the embeddings (same meaning as conv layers)

        self._generate_encoding()


    def __call__(self, sequence):
        "Sequence to k-mer representation as hot-encodings"
        L = len(sequence)
        # sequence to k-mer 
        kmer_sequence = [ sequence[j:j+self.k] for j in range(len(sequence)) if j < L-(self.k-1) ]
        
        # TODO with stride
        # kmer_sequence = [ sequence[j+s:j+k+s] for j in range(1,len(sequence)) if j+s < np.ceil((L-k)/s) + 1 ]

        # k-mers to hot-encoding 
        encoding = [self.ENCODING.get(nb) for nb in kmer_sequence]

        return np.vstack(encoding)

    def _generate_encoding(self,):
        "Create dictionary with kmers and their hot-encoding representation"
        # len of the hot-encoding vectors
        self.len_vector = 4**self.k
        
        # Generate hot-encoding for each k-mer
        self.kmers = ["".join(p) for p in product("ACGT", repeat=self.k)]

        self.ENCODING = OrderedDict([
                            ( kmer, self._canonical_vector(j) ) 
                                for j,kmer in enumerate(self.kmers)
                        ])

    def _canonical_vector(self, index):
        "Get canonical vector e_j = 1 in position j, 0 otherwise"
        x = np.zeros(self.len_vector)
        x[index] = 1.0
        return x

class Word2Vec: 
    "k-mer represented as vector with word2vec model"
    PATH_EMBEDDINGS = BASE_DIR.parent

    def __init__(self, k: int, s: int, size_emb: int = 20):
        self.k = k # k-mer
        self.s = s # stride
        # self.path_model = path_model # path word2vec model
        self.size_emb   = size_emb # size embedding, it depends on the trained models

        self.load_model()

    def __call__(self, sequence: str): 
        "Sequence to k-mer representation as embedding"
        L = len(sequence)
        
        # sequence to k-mer 
        kmer_sequence = self.seq2kmers(sequence)

        # k-mers to embedding 
        encoding = [self.kmer2vec(kmer) for kmer in kmer_sequence]

        return np.vstack(encoding)
    
    def seq2kmers(self, sequence: str): 
        "String sequence to list of kmers based on stride s and length of k-mer k"
        L = len(sequence)
        N_kmer = np.ceil((L-self.k)/self.s) + 1
        return [ sequence[j:j+self.k] for j in range(0, L, self.s) if j < N_kmer ]

    def kmer2vec(self, kmer: str):
        "Given a k-mer, return his vector representation"
        return self.wv[kmer]

    def load_model(self,): 
        "Load word2vec model saved as '<name-model>.wordvectors' file"
        path_emb = self.PATH_EMBEDDINGS.joinpath(f"embeddings/word2vec/{self.k}-mer_{self.size_emb}-emb.wordvectors").as_posix()
        self.wv = KeyedVectors.load(path_emb, mmap='r')