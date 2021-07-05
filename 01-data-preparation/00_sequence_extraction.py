import os
import pandas as pd

from collections import (
    defaultdict,
    namedtuple
)

from src import SequencerSpliceJunction

path_db = "data/annotation-004.db"
path_fa = "data/Homo_sapiens-005.fa"

train_chromosome = ['2','4','6','8','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
test_chromosome = ['1','3','5','7','9']
chromosomes = train_chromosome + test_chromosome
chromosomes.sort()
for chromosome in chromosomes: 
    if os.path.exists("data/exoninfo_chr_{}.csv".format(chromosome)):
        continue

    print(chromosome)
    # initialize sequencer and load seq data
    ds = SequencerSpliceJunction(
        path_annotations = path_db, 
        path_fasta = path_fa, 
        chromosomes = chromosomes,
        len_samples = 60,
        k  = 10, # samples per exon (negative) / splice-junction (positive)
    )

    # Save exoninfo for the chromosome
    if not os.path.exists("data/exoninfo_chr_{}.csv".format(chromosome)):
        df_exon = pd.DataFrame(ds.exon_info).to_csv("data/exoninfo_chr_{}.csv".format(chromosome))
    
    if not os.path.exists("data/positive_samples_chr_{}.csv".format(chromosome)):
        ds.generate_positive_samples_fast(chromosome=chromosome,)
        
    if not os.path.exists("data/negative_samples_chr_{}.csv".format(chromosome)):
        ds.generate_negative_samples(chromosome=chromosome, allow_empty=True, save=True)
