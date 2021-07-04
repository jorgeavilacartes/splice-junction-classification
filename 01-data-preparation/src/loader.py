import json
from functools import lru_cache

class FastaLoader: 
    "Read a genome from a fasta file"
    
    def __call__(self,filename: str) -> str: 
        "Return the loaded sequence"
        self.load(filename)
        return self.sequence

    @lru_cache(maxsize=2)
    def load(self, filename: str):
        "load genome from FASTA format"
        
        # 1. Load your file
        with open(filename, 'r') as fp:
            # Read all lines but the first one
            genome = "".join(line.strip() for line in fp if not line.startswith('>'))
        
        # 2. Assign the genome to the 'sequence' attribute
        self.sequence = genome