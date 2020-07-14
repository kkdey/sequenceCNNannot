import pandas as pd
import h5py
import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.io
import os
from os import path
import argparse
import fnmatch
import math
import pyfasta
import random
import collections

parser = argparse.ArgumentParser(description='Generate encoded sequence data for two classes - corresponding to two annotations')
parser.add_argument('--chr', action="store",
                    dest="chr", type=int,
                    help='The chromosome for which to carry the analysis')
parser.add_argument('--annot1', action="store",
                    dest="Class 1 annotation", type=str, 
                    help='Name of the Class 1 annotation')
parser.add_argument('--annot2', action="store",
                    dest="Class 2 annotation", type=str, 
                    help='Name of the Class 2 annotation')
parser.add_argument('--bed_cell1', action="store",
                    dest="Cell with class 1 annotation", type=str, 
                    help='Directory with the Class 1 annotations')
parser.add_argument('--bed_cell2', action="store",
                    dest="Cell with class 2 annotation", type=str, 
                    help='Directory with the Class 2 annotations')
parser.add_argument('--out_cell', action="store",
                    dest="Cell where the output is saved", type=str, 
                    help='Directory containing the output h5 objects')
parser.add_argument('--inputsize', action="store",
                    dest="inputsize of the sequence context", type=int, default = 1000,
                    help='The size of the sequence context')
parser.add_argument('--num_train', action="store",
                    dest="number of training samples", type=int, default = 25000,
                    help='The number of training samples per chromosome')

args = parser.parse_args()
numchr = args.chr
annot1 = args.annot1
annot2 = args.annot2
bed_cell1 = args.bed_cell1
bed_cell2 = args.bed_cell2
out_cell = args.out_cell
inputsize = args.inputsize
num_train = args.num_train

out_cell="/n/groups/price/kushal/LDSC-Average/data/Sequences"
bed_cell1="/n/groups/price/kushal/LDSC-Average/data/BEDANNOTATIONS/E042"
bed_cell2="/n/groups/price/kushal/LDSC-Average/data/BEDANNOTATIONS/E082"
annot1_name="E042-chrom_EnhA1"
annot2_name="E082-chrom_EnhA1"
inputsize=1000
num_train=25000


def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.

    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output

    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize

    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1
    # get the complementary sequences
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp

def fetchSeqs(chr, pos, ref, alt, shift=0, inputsize=2000):
    """Fetches sequences from the genome.

    Retrieves sequences centered at the given position with the given inputsize.
    Returns both reference and alternative allele sequences . An additional 100bp
    is retrived to accommodate indels.

    Args:
        chr: the chromosome name that must matches one of the names in CHRS.
        pos: chromosome coordinate (1-based).
        ref: the reference allele.
        alt: the alternative allele.
        shift: retrived sequence center position - variant position.
        inputsize: the targeted sequence length (inputsize+100bp is retrived for
                reference allele).

    Returns:
        A string that contains sequence with the reference allele,
        A string that contains sequence with the alternative allele,
        A boolean variable that tells whether the reference allele matches the
        reference genome

        The third variable is returned for diagnostic purpose. Generally it is
        good practice to check whether the proportion of reference allele
        matches is as expected.
    """
    windowsize = inputsize + 0
    mutpos = int(windowsize / 2 - 1 - shift)
    # return string: ref sequence, string: alt sequence, Bool: whether ref allele matches with reference genome
    seq = genome.sequence({'chr': chr, 'start': pos + shift -
                           int(windowsize / 2 - 1), 'stop': pos + shift + int(windowsize / 2)})
    return seq[:mutpos] + ref + seq[(mutpos + len(ref)):], seq[:mutpos] + alt + seq[(mutpos + len(ref)):], seq[mutpos:(mutpos + len(ref))].upper() == ref.upper()

genome = pyfasta.Fasta("/n/groups/price/kushal/ExPecto/test_folder/hg19.fa")
CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']

#out_folder = '/n/groups/price/kushal/DATA/CNN_SELECTION/' + annot + '_and_ComplementVars'
if not os.path.exists(out_cell + "/" + annot1_name + "_" + annot2_name):
    os.mkdir(out_cell + "/" + annot1_name + "_" + annot2_name)

for numchr in range(1, 23, 1):
    with h5py.File(out_cell + '/' + annot1_name + '_' + annot2_name + '/chr' + str(numchr) + '_encode.h5', "a") as hf:
        inputfile = "/n/groups/price/kushal/DATA/BIMS/1000G.EUR.QC." + str(numchr) + ".bim"
        bimfile = pd.read_csv(inputfile, sep='\t', header=None, comment='#')
        bimfile.iloc[:, 0] = 'chr' + bimfile.iloc[:, 0].map(str).str.replace('chr', '')
        bimfile = bimfile[bimfile.iloc[:, 0].isin(CHRS)]
        good_indices_1 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 4])]) == 1))
        good_indices_2 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 5])]) == 1))
        good_indices = np.intersect1d(good_indices_1, good_indices_2)
        bimfile= bimfile.loc[good_indices]
        annot1 = pd.read_csv(bed_cell1 + "/" + annot1_name +"/" + annot1_name + "." + str(numchr)+".annot.gz", compression = "gzip", sep = "\t")
        annot2 = pd.read_csv(bed_cell2 + "/" + annot2_name + "/" + annot2_name + "." + str(numchr)+".annot.gz", compression = "gzip", sep = "\t")
        annot1_indices = np.where(annot1.iloc[:,0] == 1)[0]
        annot2_indices = np.where(annot2.iloc[:,0] == 1)[0]
        ###############################.  Encoding the sequences for variants that are active (coding). #######################################
        np.random.shuffle(annot1_indices)
        num_train_1=np.min(np.asarray([num_train, len(annot1_indices)]))
        bimfile_sub = bimfile.iloc[annot1_indices[:num_train_1],:]
        bimfile_sub.index = range(0, bimfile_sub.shape[0])
        refseqs = []
        altseqs = []
        ref_matched_bools = []
        shift=0
        for i in range(bimfile_sub.shape[0]):
            refseq, altseq, ref_matched_bool = fetchSeqs(bimfile_sub[0][i], bimfile_sub[3][i], bimfile_sub[5][i], bimfile_sub[4][i], shift=shift, inputsize=inputsize)
            refseqs.append(refseq.upper())
            altseqs.append(altseq.upper())
            ref_matched_bools.append(ref_matched_bool)
        encoded_altseqs_1 = encodeSeqs(refseqs, inputsize=inputsize)
        ###############################.  Encoding the sequences for variants that are inactive (not coding). #######################################
        np.random.shuffle(annot2_indices)
        num_train_2=np.min(np.asarray([num_train, len(annot2_indices)]))
        bimfile_sub = bimfile.iloc[annot2_indices[:num_train_2],:]
        bimfile_sub.index = range(0, bimfile_sub.shape[0])
        refseqs = []
        altseqs = []
        ref_matched_bools = []
        shift=0
        for i in range(bimfile_sub.shape[0]):
            refseq, altseq, ref_matched_bool = fetchSeqs(bimfile_sub[0][i], bimfile_sub[3][i], bimfile_sub[5][i], bimfile_sub[4][i], shift=shift, inputsize=inputsize)
            refseqs.append(refseq.upper())
            altseqs.append(altseq.upper())
            ref_matched_bools.append(ref_matched_bool)
        encoded_altseqs_2 = encodeSeqs(refseqs, inputsize=inputsize)
        dset1 = hf.create_dataset('trainxdata1', (encoded_altseqs_1.shape[0], 4, inputsize), maxshape = (None, 4, inputsize), \
                                        data = encoded_altseqs_1, fillvalue=0)
        dset2 = hf.create_dataset('trainxdata2', (encoded_altseqs_2.shape[0], 4, inputsize), maxshape = (None, 4, inputsize), \
                                        data = encoded_altseqs_2, fillvalue=0)
        print(dset1.shape)
        print(dset2.shape)
