#!/usr/bin/env python
#from pyspark import SparkContext
#sc = SparkContext(appName="PythonProject", pyFiles=['lib.zip'])
import numpy as np
import argparse
import time
from scipy.io import arff
from lib.splitter import split
from lib.bagger import get_size_no, partition, bag, pair, cartesian
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE
from lib.scaleKMM import *;
from lib.util import *
from CenKmmProcess import cenKmmProcess
from EnsKmmProcess import ensKmmProcess
from kmmProcess import kmmProcess


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-b', "--bagging", type=int, choices=[1,2,3,4], default=1, help="bagging strategy")
    parser.add_argument("-t", "--training", type=int, default=12000, help="size of training data")
    parser.add_argument("-r", "--reverse", action="store_true", help="set -t as the size of test data")
    #parser.add_argument("-s", "--tr_bsize", type=int, help="the sample size of train set")
    #parser.add_argument("-x", "--te_bsize", type=int, help="the sample size of test set")
    #parser.add_argument("-m", "--train_samples", type=int, help="number of samples from training")
    #parser.add_argument("-n", "--test_samples", type=int, help="number of samples from test")
    #parser.add_argument("-i", "--input", type=str, default='./dataset/powersupply.arff', help="default input file")
    parser.add_argument("-i", "--input", type=str, default='/home/wzyCode/scalablelearning/dataset/kdd.arff', help="default input file")
    #parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/nmseKMM.txt', help="default output file")
    #parser.add_argument("-o1", "--output1", type=str, default='/home/wzyCode/scalablelearning/nmseCenKmm.txt', help="default output file")
    #parser.add_argument("-o2", "--output2", type=str, default='/home/wzyCode/scalablelearning/nmseEnsKmm.txt', help="default output file")
    #parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    args = parser.parse_args()

    training_size = args.training # training set size (small training set)
    reverse = args.reverse # flip training to test (small test set)
    input_file = args.input # input file path
    
    #sc = SparkContext()
    
    # Step 1: Generate biased train and test set, as well as the orginal beta for train set
    start = time.time()

    train, train_beta, test = split(input_file, training_size, reverse)
    #trianBroad = sc.broadcast(train)
    #train_data = np.array(trianBroad.value)
    train_data = np.array(train)
    #testBoard = sc.broadcast(test)
    #test_data = np.array(testBoard.value)
    test_data = np.array(test)
    orig_beta_data = np.array(train_beta)
    
    np.savetxt("train_data.txt",train_data);
    np.savetxt("test_data.txt",test_data);
    np.savetxt("orig_beta_data.txt",orig_beta_data);
    
    end = time.time()
    split_time = end - start
    
    
    #kmmProcess(tr_bsize,m,te_bsize,n,mode,output_file,sc)
    #cenKmmProcess(output_file1)
    #ensKmmProcess(tr_bsize,m,te_bsize,n,mode,output_file2,sc)
    
   
if __name__ == '__main__':
    main()