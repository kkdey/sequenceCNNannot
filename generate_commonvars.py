import argparse
import math
import pyfasta
import numpy as np
import pandas as pd
import h5py

parser = argparse.ArgumentParser(description='Generate encoded sequence data for common variants of each chromso')
parser.add_argument('--chr', type=int, help='Chromosome number')
parser.add_argument('--inputsize', type = int, default = 1000, help = 'Size of the sequence generated')
parser.add_argument('--num_train', type = int, default = 25000, help = 'Number of training samples per class - common or rare')
args = parser.parse_args()
numchr = args.chr
inputsize = args.inputsize
num_train = args.num_train

## num_train is not used 

genome = pyfasta.Fasta("/n/groups/price/kushal/ExPecto/test_folder/hg19.fa")
CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']


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

with h5py.File('/n/groups/price/kushal/LDSC-Average/data/CommonVars_Alt/chr' + str(numchr) + '_encode.h5', "a") as hf:
    inputfile = "/n/groups/price/kushal/DATA/BIMS/1000G.EUR.QC." + str(numchr) + ".bim"
    bimfile = pd.read_csv(inputfile, sep='\t', header=None, comment='#')
    bimfile.iloc[:, 0] = 'chr' + bimfile.iloc[:, 0].map(str).str.replace('chr', '')
    bimfile = bimfile[bimfile.iloc[:, 0].isin(CHRS)]
    good_indices_1 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 4])]) == 1))
    good_indices_2 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 5])]) == 1))
    good_indices = np.intersect1d(good_indices_1, good_indices_2)
    bimfile= bimfile.loc[good_indices]
    bimfile.index = range(0, bimfile.shape[0])
    refseqs = []
    altseqs = []
    ref_matched_bools = []
    shift=0
    for i in range(bimfile.shape[0]):
        refseq, altseq, ref_matched_bool = fetchSeqs(bimfile[0][i], bimfile[3][i], bimfile[5][i], bimfile[4][i], shift=shift, inputsize=inputsize)
        refseqs.append(refseq.upper())
        altseqs.append(altseq.upper())
        ref_matched_bools.append(ref_matched_bool)
    encoded_altseqs_1 = encodeSeqs(altseqs, inputsize=inputsize)
    dset1 = hf.create_dataset('trainxdata', (encoded_altseqs_1.shape[0], 4, 1000), maxshape = (None, 4, 1000), \
                            data = encoded_altseqs_1, fillvalue=0)
    print(dset1.shape)
