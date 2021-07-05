from functools import lru_cache
import pandas as pd
import random 
from collections import (
    defaultdict,
    namedtuple
)
import gffutils
from Bio import SeqIO
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class AnnotationByExon: 
    """
    Get annotations for each exon of a list of chromosomes from a FeatureDB
    """
    
    ExonInfo = namedtuple("ExonInfo",["id_exoninfo","chromosome","gene_id","transcript_id","exon_id","exon_number","acceptor_idx","donor_idx"])
    
    def __init__(self, path_annotations: str, chromosomes: list):
        self.path_annotations = path_annotations
        self.chromosomes = chromosomes
        
        # id for ExonInfo
        self.id = 0

        # load annotations and genome sequence
        self._load_annotations()
        
        # get genes id
        self._get_genes_id()
        
        # exon info
        self._get_exon_info()
    
    @lru_cache
    def _load_annotations(self,):
        print("loading annotations")
        self.db = gffutils.FeatureDB(self.path_annotations, keep_order=True)
        
    def _count_genes(self,):
        return self.db.count_features_of_type('gene')
    
    def _get_genes_id(self,): 
        print("get genes id")
        # dict to save gen_id by chromosome
        gen2chromosome = defaultdict(str)
        
        # create list to save genes id
        genes_id = []
        n_genes = self._count_genes() # to show progress_bar with tqdm
        
        # get all genes id
        for gen in tqdm(self.db.features_of_type('gene'), total=n_genes):
            gen_id = gen.attributes['gene_id'][0]
            genes_id.append(gen_id)
            
            chromosome = self.db[gen_id].chrom
            gen2chromosome[gen_id] = chromosome
            
        self.genes_id = genes_id
        self.gen2chromosome = gen2chromosome
        
    def _feature_from_gen(self, feature, gen_id):
        "returns a generator with the features for the selected gene"
        return self.db.children(gen_id, featuretype = feature, order_by = 'start')
        
    def _get_exon_info(self,): 
        """Get info for each exon"""
        print("get exon info")
        exon_info = []
        for gen_id in tqdm(self.genes_id): 
            # get data only for chromosomes 
            if self.gen2chromosome.get(gen_id) in self.chromosomes: 
                
                for exon in self._feature_from_gen('exon', gen_id):
                    # acceptor and donor positions
                    acceptor_idx = exon.start - 1
                    donor_idx = exon.end - 1
                    
                    # consolidate exon information
                    info = self.ExonInfo(
                            self.new_id(),
                            self.gen2chromosome.get(gen_id),
                            exon["gene_id"][0],      
                            exon["transcript_id"][0], 
                            exon["exon_id"][0], 
                            int(exon["exon_number"][0]), 
                            int(acceptor_idx),
                            int(donor_idx)
                            )
                    
                    # collect exon information
                    exon_info.append(info)

        self.exon_info = exon_info
        #del self.genes_id, self.gen2chromosome, self.db
        
    def new_id(self,):
        "Create new id for ExonInfo"
        self.id += 1
        return self.id

class SequencerSpliceJunction(AnnotationByExon):
    PosExon=namedtuple("PosExon",["start","end"])
    NegativeSample=namedtuple("NegativeSample",["negsample_id", "pos_exon", "id_exoninfo", "seq"])
    PositiveSample=namedtuple("PositiveSample",["possample_id", "pos_donor","pos_acceptor","id_exoninfo_donor","id_exoninfo_acceptor", "seq"])

    def __init__(self, path_annotations: str, path_fasta: str, chromosomes: list, len_samples: int, k: int, seed: int=42): 
        super().__init__(path_annotations, chromosomes)
        self.path_fasta = path_fasta
        self.len_samples = len_samples
        self.k = k
        
        # load sequence by chromosome
        self._load_fasta()

        # initialize id for negative and positive samples
        self.negsample_id = 0
        self.possample_id = 0

        # list to save negative and positive samples
        self.neg_samples = []
        self.pos_samples = []

    @lru_cache
    def _load_fasta(self,): 
        """returns a dictionary with chromosome name as keys and his Sequence as values
        The returned sequence can be queried as a list
        """
        print("collecting sequence by chromosome")
        fasta = SeqIO.parse(self.path_fasta , "fasta")
        seq_by_chromosome = {}
        for record in tqdm(fasta): 
            chromosome = str(record.id)
            if chromosome in self.chromosomes: 
                seq_by_chromosome[chromosome] = record.seq
            
        self.seq_by_chromosome = seq_by_chromosome

    def negative_samples(self, start: int, end: int, allow_empty: bool = False):
        """negative samples for one exon. Returns pairs of indexes (start, end) 
        with the position in the chromosome sequence
        """
        assert start <= end, "start must be lower than end"
        
        # Compute possible samples without replacing
        k = min(end-start-self.len_samples+1, self.k)
        if allow_empty and k<=0:
            return []

        assert k>0,"len_samples too large. max_len = {}".format(end-start)
        
        # sample 'k' starts positions of sequences 
        starts_idx = random.sample(range(start, end+1-self.len_samples), k)
        
        # return (start, end) positions for each sequence
        return [self.PosExon(s, s+self.len_samples) for s in starts_idx]

    def positive_samples(self, donor_start, donor_end, acceptor_start, acceptor_end, alpha, beta,):
        """positive samples from 2 consecutive exons (donor->acceptor) in a transcript.
        Returns pairs of indexes (start, end) for both exons, corresponding 
        to the position in the chromosome sequence
        """
        # range of indexes to sample from
        sample_from, sample_end = donor_end-alpha , donor_end-(self.len_samples-1-beta)+1 

        # redefine k 
        k = min(sample_end-sample_from, self.k)
        if k<=0:
            return []

        # sample 'k' starts positions of sequences
        starts_idx = random.sample(range(sample_from, sample_end), k)

        return [(
            self.PosExon(j, donor_end+1), # subsequence donor exon
            self.PosExon(acceptor_start, self.len_samples+j+acceptor_start-donor_end-1)  # subsequence acceptor exon
            ) for j in starts_idx]

    def sequence_from_chromosome(self, chromosome: str, start: int, end: int,): 
        "get DNA sequence from start (included) to end (not included)"
        return self.seq_by_chromosome.get(chromosome,[])[start:end]

    def new_negsample_id(self):
        self.negsample_id +=1
        return self.negsample_id
    
    def new_possample_id(self):
        self.possample_id +=1
        return self.possample_id
    
    def generate_positive_samples(self,chromosome=None, save=False):
        self.counter_pos = 0
        print("Generating positive samples")
        self.neg_samples = []
        transcripts = self.get_all_transcripts() if chromosome is None else self.get_all_transcripts_by(chromosome)
        
        for transcript in tqdm(transcripts): 
            
            # get all exons of the transcript in ascending order
            exons_transcript = list(filter(lambda exon: exon.transcript_id == transcript , self.exon_info))
            exons_transcript = sorted(exons_transcript, key=lambda exon: exon.exon_number, reverse=False)

            # if 2 or more exons exists, generate positive samples
            for donor_exon, acceptor_exon in zip(exons_transcript[:-1], exons_transcript[1:]):
                
                # length of exons
                len_donor = donor_exon.donor_idx - donor_exon.acceptor_idx
                len_acceptor = acceptor_exon.donor_idx - acceptor_exon.acceptor_idx

                # allows positive samples
                alpha = min(self.len_samples -1, len_donor)# donor exon
                beta  = min(self.len_samples -1, len_acceptor)# acceptor exon

                #Verify if donor and acceptor exons satisfy length to extract sequence samples
                if alpha + beta > self.len_samples:
                    self.counter_pos +=1
                    # indexes for positive samples
                    pos_idx = self.positive_samples(donor_start=donor_exon.acceptor_idx, donor_end=donor_exon.donor_idx, 
                                                    acceptor_start=acceptor_exon.acceptor_idx, acceptor_end=acceptor_exon.donor_idx,
                                                    alpha=alpha, beta=beta,)
                    
                    # generate sequences for each sample
                    for pos_donor_exon, pos_acceptor_exon in pos_idx:
                        seq_donor = self.sequence_from_chromosome(chromosome=donor_exon.chromosome, start=pos_donor_exon.start, end=pos_donor_exon.end)
                        seq_acceptor = self.sequence_from_chromosome(chromosome=acceptor_exon.chromosome, start=pos_acceptor_exon.start, end=pos_acceptor_exon.end)

                        # join donor and acceptor portions        
                        seq = seq_donor + seq_acceptor
                        sample = self.PositiveSample(self.new_negsample_id(), pos_donor_exon, pos_acceptor_exon, donor_exon.id_exoninfo, acceptor_exon.id_exoninfo, seq)
                        self.pos_samples.append(sample)

        # Save results
        if save is True: 
            if chromosome is None:
                path_save = "data/positive_samples.csv"
            else:
                path_save = "data/positive_samples_chr_{}.csv".format(chromosome)

            self.positive_samples_to_csv(path_save)        

    def generate_negative_samples(self, chromosome=None, allow_empty=True, save=False):
        print("Generating negative samples")
        if chromosome is None:
            transcripts = self.get_all_transcripts()
        else: 
            self.neg_samples = []
            transcripts = self.get_all_transcripts_by(chromosome)

        for transcript in tqdm(transcripts): 
            
            # get all exons of the transcript
            exons_transcript = list(filter(lambda exon: exon.transcript_id == transcript , self.exon_info))
            #exons_transcript = sorted(exons_transcript, key=lambda exon: exon.exon_number, reverse=False)

            # 
            for exon in exons_transcript:
                neg_idx = self.negative_samples(start=exon.acceptor_idx, end=exon.donor_idx, allow_empty=allow_empty)
                
                # generate sequences for each sample
                for pos_exon in neg_idx:
                    seq = self.sequence_from_chromosome(chromosome=exon.chromosome, start=pos_exon.start, end=pos_exon.end)
                    sample = self.NegativeSample(self.new_negsample_id(), pos_exon, exon.id_exoninfo, str(seq))
                    self.neg_samples.append(sample)
            
        # Save results
        if save is True: 
            if chromosome is None:
                path_save = "data/negative_samples.csv"
            else:
                path_save = "data/negative_samples_chr_{}.csv".format(chromosome)

            self.negative_samples_to_csv(path_save)

    def allows_neg_samples(self,pos_exon):
        "Returns True if the exon satisfy the length to extract negative samples"
        return pos_exon.end - pos_exon.start > self.len_samples

    def allows_pos_samples(self, pos_exon1, pos_exon2):
        "Returns True if the exon1-exon2 satisfy the length to extract positive samples"
        pass
        

    def get_all_transcripts(self,): 
        "Get a list with all (unique) transcripts"
        return list(set(exon.transcript_id for exon in self.exon_info))

    def get_all_transcripts_by(self, chromosome: str): 
        "Get a list with all (unique) transcripts in the chromosome"
        return list(set(exon.transcript_id for exon in self.exon_info if exon.chromosome==chromosome))
    
    # -- Results to csv --
    def negative_samples_to_csv(self,dirsave: str):
        "Save negative samples to csv"
        df_exon = pd.DataFrame(self.exon_info)
        pd.DataFrame(self.neg_samples).merge(df_exon, on ="id_exoninfo", how="left").to_csv(dirsave)

    def positive_samples_to_csv(self, dirsave: str):
        "Save positive samples to csv"
        df_exon = pd.DataFrame(self.exon_info)
        df = pd.DataFrame(self.pos_samples).merge(df_exon, right_on ="id_exoninfo", left_on = "id_exoninfo_donor",how="left")
        df = df.merge(df_exon, right_on ="id_exoninfo", left_on = "id_exoninfo_acceptor", how="left", suffixes = ("_d","_a"))
        df.to_csv(dirsave)

    # -- Faster generation of positive samples
    def positive_samples_from(self, transcript):
        "Extract positive samples from a transcript (list of exons)"

        # get all exons of the transcript in ascending order
        exons_transcript = list(filter(lambda exon: exon.transcript_id == transcript , self.exon_info))
        exons_transcript = sorted(exons_transcript, key=lambda exon: exon.exon_number, reverse=False)

        # if 2 or more exons exists, generate positive samples
        pairs_da = [(e_d,e_a) for e_d, e_a in zip(exons_transcript[:-1], exons_transcript[1:])]

        if len(pairs_da)>0:
            # collect positive samples for each pair (donor, acceptor)
            list(map(self.extract_positive_samples, pairs_da))

    def extract_positive_samples(self, pair_da):

        # donor and acceptor exons
        donor_exon, acceptor_exon = pair_da

        # length of exons
        len_donor = donor_exon.donor_idx - donor_exon.acceptor_idx
        len_acceptor = acceptor_exon.donor_idx - acceptor_exon.acceptor_idx

        # allows positive samples
        alpha = min(self.len_samples -1, len_donor)# donor exon
        beta  = min(self.len_samples -1, len_acceptor)# acceptor exon

        #Verify if donor and acceptor exons satisfy length to extract sequence samples
        if alpha + beta > self.len_samples:
            self.counter_pos +=1
            # indexes for positive samples
            pos_idx = self.positive_samples(donor_start=donor_exon.acceptor_idx, donor_end=donor_exon.donor_idx, 
                                            acceptor_start=acceptor_exon.acceptor_idx, acceptor_end=acceptor_exon.donor_idx,
                                            alpha=alpha, beta=beta,)
            
            # generate sequences for each sample
            for pos_donor_exon, pos_acceptor_exon in pos_idx:
                seq_donor = self.sequence_from_chromosome(chromosome=donor_exon.chromosome, start=pos_donor_exon.start, end=pos_donor_exon.end)
                seq_acceptor = self.sequence_from_chromosome(chromosome=acceptor_exon.chromosome, start=pos_acceptor_exon.start, end=pos_acceptor_exon.end)

                # join donor and acceptor portions        
                seq = seq_donor + seq_acceptor
                sample = self.PositiveSample(self.new_negsample_id(), pos_donor_exon, pos_acceptor_exon, donor_exon.id_exoninfo, acceptor_exon.id_exoninfo, seq)
                self.pos_samples.append(sample)

        return ""

    def generate_positive_samples_fast(self,chromosome: str, save=True):
        "Generate positive samples using map() for faster computation"
        
        self.counter_pos = 0
        print("Generating positive samples")
        self.neg_samples = []
        transcripts = self.get_all_transcripts() if chromosome is None else self.get_all_transcripts_by(chromosome)

        with ThreadPoolExecutor(max_workers=24) as executor: 
            list(tqdm(executor.map(self.positive_samples_from, transcripts), total=len(transcripts)))
        
        # with ProcessPoolExecutor(max_workers=30) as executor: 
        #     list(tqdm(executor.map(self.positive_samples_from, transcripts), total=len(transcripts)))

        if save is True: 
            path_save = "data/positive_samples_chr_{}.csv".format(chromosome)
            self.positive_samples_to_csv(path_save)        

