import h5py
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import sklearn.metrics
import scipy.io
import collections
import os
from os import path
import argparse
import fnmatch
import pyfasta
import math

#annot1_name = "E042-chrom_EnhW2"
#annot2_name = "E082-chrom_EnhW2"
#train_cell = "/n/groups/price/kushal/LDSC-Average/data/DeepTrain"
#annot_cell = "/n/groups/price/kushal/LDSC-Average/data/ANNOTATIONS/DeepAnnot"
#bimpath = "/n/groups/price/kushal/1000G/processed_data"
#commonvars_path = "/n/groups/price/kushal/DATA/CNN_SELECTION/CommonVars"


parser = argparse.ArgumentParser(description='Deep Learning predictions for two annotations')
parser.add_argument('--chr', action="store",
                    dest="chr", type=int,
                    help='The chromosome for which to carry the analysis')
parser.add_argument('--annot1', action="store",
                    dest="annot1", type=str,
                    help='The name of Class 1 annotation')
parser.add_argument('--annot2', action="store",
                    dest="annot2", type=str,
                    help='The name of Class 2 annotation')
parser.add_argument('--train_cell', action="store",
                    dest="train_cell", type=str,
                    help='The directory with saved model output files')
parser.add_argument('--annot_cell', action="store",
                    dest="annot_cell", type=str,
                    help='The directory with saved model annotation files')
parser.add_argument('--bimpath', action="store",
                    dest="bimpath", type=str,
                    help='The path of the BIM files')
parser.add_argument('--commonvars_path', action="store",
                    dest="commonvars_path", type=str,
                    help='The path where the common variants data are stored')

args = parser.parse_args()
numchr = args.chr
annot1_name = args.annot1
annot2_name = args.annot2
train_cell = args.train_cell
annot_cell = args.annot_cell
bimpath=args.bimpath
commonvars_path = args.commonvars_path

#train_folder = train_cell + "/" + annot1_name + "_" + annot2_name 


CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']


if numchr % 2 == 0:
    inputfile = bimpath + "/1000G.EUR.QC." + str(numchr) + ".bim"
    bimfile = pd.read_csv(inputfile, sep='\t', header=None, comment='#')
    bimfile.iloc[:, 0] = 'chr' + bimfile.iloc[:, 0].map(str).str.replace('chr', '')
    bimfile = bimfile[bimfile.iloc[:, 0].isin(CHRS)]
    good_indices_1 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 4])]) == 1))
    good_indices_2 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 5])]) == 1))
    good_indices = np.intersect1d(good_indices_1, good_indices_2)
    bimfile= bimfile.loc[good_indices]
    bimfile.index = range(0, bimfile.shape[0])
    file = h5py.File(commonvars_path + "/chr" + str(numchr) + "_encode.h5", 'r')
    trainxdata = file.get('trainxdata')
    trainxdata = np.transpose(trainxdata, (0, 2, 1))
    train_folder = train_cell + "/" + annot1_name + "_" + annot2_name + '_odd'
    list_models = fnmatch.filter(os.listdir(train_folder), '*.index')
    numbers = [int(str.split(str.split(x, "model-tensorflow-")[1], ".")[0]) for x in list_models]
    last_run_index = list_models[np.argmax(numbers)].split("-")[2].split(".ckpt")[0]
    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(train_folder + '/model-tensorflow-' \
                                           + last_run_index + '.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(train_folder))
    graph = tf.get_default_graph()
    x= graph.get_tensor_by_name('Placeholder:0')
    y = graph.get_tensor_by_name('Placeholder_1:0')
    keep_prob = graph.get_tensor_by_name('Placeholder_2:0')
    ypred2 = graph.get_tensor_by_name('fc2/Sigmoid:0')
    split_encoded_x = np.array_split(trainxdata, 100, 0)
    annot_preds_1 = []
    for num in range(len(split_encoded_x)):
        chunk_data= split_encoded_x[num].astype(np.float32)
        preds_file = sess.run(ypred2, feed_dict={x: chunk_data, keep_prob: 1})
        annot_preds_1.append(np.ndarray.tolist(preds_file[:,0]))
        print("We are at chunk" + str(num))
    flat_annot = [item for sublist in annot_preds_1 for item in sublist]
    index_start = 0
    index_end = int(len(flat_annot)/2)
    snp_temp_1 = (np.asarray(flat_annot[index_start:index_end]) +  np.asarray(flat_annot[index_end:(2*index_end)]))/2.0
    if not os.path.exists(annot_cell + "/" +  annot1_name + "_" + annot2_name ):
        os.makedirs(annot_cell + "/" +  annot1_name + "_" + annot2_name )
    df1 = {"CHR" : numchr, "BP" : bimfile.iloc[:,3], "SNP": bimfile.iloc[:,1], "CM": bimfile.iloc[:,2], "AN": snp_temp_1}
    pdcat1 = pd.DataFrame(df1)
    pdcat1.to_csv(annot_cell + "/" + annot1_name + "_" + annot2_name + "/" + annot1_name + "_" + annot2_name + "." + \
                 str(numchr) + ".annot.gz",
                 index=False, sep="\t", compression = "gzip")


if numchr % 2 == 1:
    inputfile =  bimpath + "/1000G.EUR.QC." + str(numchr) + ".bim"
    bimfile = pd.read_csv(inputfile, sep='\t', header=None, comment='#')
    bimfile.iloc[:, 0] = 'chr' + bimfile.iloc[:, 0].map(str).str.replace('chr', '')
    bimfile = bimfile[bimfile.iloc[:, 0].isin(CHRS)]
    good_indices_1 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 4])]) == 1))
    good_indices_2 = np.array(np.where(np.array([len(x) for x in np.array(bimfile.iloc[:, 5])]) == 1))
    good_indices = np.intersect1d(good_indices_1, good_indices_2)
    bimfile= bimfile.loc[good_indices]
    bimfile.index = range(0, bimfile.shape[0])
    file = h5py.File(commonvars_path + "/chr" + str(numchr) + "_encode.h5", 'r')
    trainxdata = file.get('trainxdata')
    trainxdata = np.transpose(trainxdata, (0, 2, 1))
    train_folder = train_cell + "/" + annot1_name + "_" + annot2_name + '_even'
    list_models = fnmatch.filter(os.listdir(train_folder), '*.index')
    numbers = [int(str.split(str.split(x, "model-tensorflow-")[1], ".")[0]) for x in list_models]
    last_run_index = list_models[np.argmax(numbers)].split("-")[2].split(".ckpt")[0]
    tf.reset_default_graph()
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(train_folder + '/model-tensorflow-' \
                                                        + last_run_index + '.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(train_folder))
    graph = tf.get_default_graph()
    x= graph.get_tensor_by_name('Placeholder:0')
    y = graph.get_tensor_by_name('Placeholder_1:0')
    keep_prob = graph.get_tensor_by_name('Placeholder_2:0')
    ypred2 = graph.get_tensor_by_name('fc2/Sigmoid:0')
    split_encoded_x = np.array_split(trainxdata, 100, 0)
    annot_preds_1 = []
    for num in range(len(split_encoded_x)):
        chunk_data= split_encoded_x[num].astype(np.float32)
        preds_file = sess.run(ypred2, feed_dict={x: chunk_data, keep_prob: 1})
        annot_preds_1.append(np.ndarray.tolist(preds_file[:,0]))
        print("We are at chunk" + str(num))
    flat_annot = [item for sublist in annot_preds_1 for item in sublist]
    index_start = 0
    index_end = int(len(flat_annot)/2)
    snp_temp_1 = (np.asarray(flat_annot[index_start:index_end]) +  np.asarray(flat_annot[index_end:(2*index_end)]))/2.0
    if not os.path.exists(annot_cell + "/" +  annot1_name + "_" + annot2_name):
        os.makedirs(annot_cell + "/" +  annot1_name + "_" + annot2_name )
    df1 = {"CHR" : numchr, "BP" : bimfile.iloc[:,3], "SNP": bimfile.iloc[:,1], "CM": bimfile.iloc[:,2], "AN": snp_temp_1}
    pdcat1 = pd.DataFrame(df1)
    pdcat1.to_csv(annot_cell + "/" +  annot1_name + "_" + annot2_name + "/" + annot1_name + "_" + annot2_name + "." + \
                 str(numchr) + ".annot.gz",
                 index=False, sep="\t", compression = "gzip")
