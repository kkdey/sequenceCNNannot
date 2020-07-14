import h5py
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

parser = argparse.ArgumentParser(description='Assemble encoded sequence data of two annotations for even and odd chromosomes')
parser.add_argument('--annot1', action="store",
                    dest="annot1", type=str,
                    help='The name of Class 1 annotation')
parser.add_argument('--annot2', action="store",
                    dest="annot2", type=str,
                    help='The name of Class 2 annotation')
parser.add_argument('--in_cell', action="store",
                    dest="in_cell", type=str,
                    help='The input directory(cell) with encoded sequence data')
parser.add_argument('--out_cell', action="store",
                    dest="out_cell", type=str,
                    help='The output directory(cell) with genomewide encoded sequence data')
parser.add_argument('--inputsize', action="store",
                    dest="inputsize", type=int, default = 1000,
                    help='The size of the sequence context')
parser.add_argument('--num_train', action="store",
                    dest="number of training samples", type=int, default = 25000,
                    help='The number of training samples per chromosome')
args = parser.parse_args()
annot1_name = args.annot1
annot2_name = args.annot2
inputsize = args.inputsize
num_train=args.num_train


#in_cell="/n/groups/price/kushal/LDSC-Average/data/Sequences"
#annot1_name = "E042-chrom_EnhA1"
#annot2_name = "E082-chrom_EnhA1"
#num_train = 25000
#inputsize = 1000
#out_cell = "/n/groups/price/kushal/LDSC-Average/data/Genomewide_Sequences"

in_folder = in_cell + "/" + annot1_name + "_" + annot2_name
out_folder = out_cell +  "/" + annot1_name + "_" + annot2_name
if not os.path.exists(out_folder):
	os.mkdir(out_folder)


chr_ids = range(1, 23, 2)
full_trainxdata = np.empty((0,inputsize,4), float)
full_trainydata = np.empty((0, 2), float)
for numchr in range(len(chr_ids)):
	common_vars = h5py.File(in_folder + "/chr" + str(chr_ids[numchr]) + "_encode.h5", 'r')
	trainxdata_1 = np.asarray(common_vars.get('trainxdata1')) ######. Boolean variables 
	trainxdata_2 = np.asarray(common_vars.get('trainxdata2'))
	trainxdata = np.concatenate((trainxdata_1, trainxdata_2), axis = 0)
	trainydata = np.column_stack((np.concatenate((np.repeat(1, trainxdata_1.shape[0]), np.repeat(0, trainxdata_2.shape[0])), axis = 0),
								 np.concatenate((np.repeat(0, trainxdata_1.shape[0]), np.repeat(1, trainxdata_2.shape[0])), axis = 0)))
	trainxdata = np.transpose(trainxdata, (0, 2, 1))
	take_indices = np.arange(trainxdata.shape[0])
	np.random.shuffle(take_indices)
	trainxdata_shuffled = trainxdata[take_indices,:,:]
	trainydata_shuffled = trainydata[take_indices,:]
	num_train_2=np.min(np.asarray([num_train, trainxdata_shuffled.shape[0]]))
	full_trainxdata = np.append(full_trainxdata, trainxdata_shuffled[:num_train_2,:,:].astype(np.float32), axis = 0)
	full_trainydata = np.append(full_trainydata, trainydata_shuffled[:num_train_2,:].astype(np.float32), axis = 0)
	print("Read training data for chr" + str(chr_ids[numchr]))

hf = h5py.File(out_folder + '/train_odd.h5', "a")
dset1 = hf.create_dataset('trainxdata', data = full_trainxdata)
dset2 = hf.create_dataset('trainydata', data = full_trainydata)
hf.close()

chr_ids = range(2, 23, 2)
full_trainxdata = np.empty((0,inputsize,4), float)
full_trainydata = np.empty((0, 2), float)
for numchr in range(len(chr_ids)):
	common_vars = h5py.File(in_folder + "/chr" + str(chr_ids[numchr]) + "_encode.h5", 'r')
	trainxdata_1 = common_vars.get('trainxdata1') ######. Boolean variables 
	trainxdata_2 = common_vars.get('trainxdata2')
	trainxdata = np.concatenate((trainxdata_1, trainxdata_2), axis = 0)
	trainydata = np.column_stack((np.concatenate((np.repeat(1, trainxdata_1.shape[0]), np.repeat(0, trainxdata_2.shape[0])), axis = 0),
								 np.concatenate((np.repeat(0, trainxdata_1.shape[0]), np.repeat(1, trainxdata_2.shape[0])), axis = 0)))
	trainxdata = np.transpose(trainxdata, (0, 2, 1))
	take_indices = np.arange(trainxdata.shape[0])
	np.random.shuffle(take_indices)
	trainxdata_shuffled = trainxdata[take_indices,:,:]
	trainydata_shuffled = trainydata[take_indices,:]
	num_train_2=np.min(np.asarray([num_train, trainxdata_shuffled.shape[0]]))
	full_trainxdata = np.append(full_trainxdata, trainxdata_shuffled[:num_train_2,:,:].astype(np.float32), axis = 0)
	full_trainydata = np.append(full_trainydata, trainydata_shuffled[:num_train_2,:].astype(np.float32), axis = 0)
	print("Read training data for chr" + str(chr_ids[numchr]))

hf = h5py.File(out_folder + '/train_even.h5', "a")
dset1 = hf.create_dataset('trainxdata', data = full_trainxdata)
dset2 = hf.create_dataset('trainydata', data = full_trainydata)
hf.close()

