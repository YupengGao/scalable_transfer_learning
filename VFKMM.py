#!/usr/bin/env python
from pyspark import SparkContext
sc = SparkContext(appName="PythonProject", pyFiles=['lib.zip'])
import numpy as np
import time
import argparse
from lib.bagger import get_size_no, partition, bag, pair, cartesian
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE
from lib.scaleKMM import *;
from lib.util import *



def kmmProcess():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--bagging", type=int, choices=[1,2,3,4], default=1, help="bagging strategy")
    parser.add_argument("-s", "--tr_bsize", type=int, help="the sample size of train set")
    parser.add_argument("-x", "--te_bsize", type=int, help="the sample size of test set")
    parser.add_argument("-m", "--train_samples", type=int, help="number of samples from training")
    parser.add_argument("-n", "--test_samples", type=int, help="number of samples from test")
    #parser.add_argument("-i", "--input", type=str, default='./dataset/powersupply.arff', help="default input file")
    parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/nmseKMM.txt', help="default output file")
    args = parser.parse_args()
    
    mode = args.bagging # bagging strategy
    tr_bsize = args.tr_bsize # By default, the train bag size is dynamic, if specified, the train bag size will fix
    te_bsize = args.te_bsize # By default, the test bag size is dynamic, if specified, the test bag size will fix
    m = args.train_samples # take m samples from training
    n = args.test_samples # take n samples from test
    output_file = args.output
    
    
#1.Bagging process
    train_data = np.loadtxt("train_data.txt")
    test_data = np.loadtxt("test_data.txt")
    orig_beta_data = np.loadtxt("orig_beta_data.txt")
    start = time.time()
    
    # Bagging the train and test data from the sampled index
    tr_bag_size, tr_bag_no = get_size_no(train_data, tr_bsize, m)
    te_bag_size, te_bag_no = get_size_no(test_data, te_bsize, n)
    # bag is the function that put random index of data into the bag
    # partition is the function that can set part the index of data equally, the size of partition is equal to bag size
    if mode == 1:  # if test is too big, provide x or n to partition test set
        tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
        te_n = partition(test_data, part_size=te_bag_size, part_no=te_bag_no)
    elif mode == 2:  # if train is too big, provide s or m to partition train set
        tr_n = partition(train_data, part_size=tr_bag_size, part_no=tr_bag_no)
        te_n = bag(test_data, size=te_bag_size, sample_no=te_bag_no)
    else: # random sample, no partition
        tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
        te_n = bag(test_data, size=te_bag_size, sample_no=te_bag_no)

    if mode < 4:
        # cartisian is to pair each training data and test data put into bag
        bags = cartesian(train_data, test_data, tr_n, te_n)
    else:
        # pair can be used to pair given number pair of train and test data by picking randomly in the pool
        # one tuple of train data pair with one tuple of test data
        bags = pair(train_data, test_data, tr_n, te_n, sample_no=min(tr_bag_no, te_bag_no))

    rdd = sc.parallelize(bags)
    end = time.time()
    bagging_time = end - start
    
    
#2. Compute Beta Process
    start = time.time()
    res = rdd.map(lambda (idx, tr, te): computeBeta(idx, tr, te)).flatMap(lambda x: x)

    rdd1 = res.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),
                              lambda a,b: (a[0] + b[0], a[1] + b[1]))

    est_beta_map = rdd1.mapValues(lambda v: v[0]/v[1]).collectAsMap()
    est_beta_idx = est_beta_map.keys()

    end = time.time()
    compute_time = end - start
    
    
#3. Compute the NMSE between the est_beta and orig_beta through KMM
    start = time.time()
    
    est_beta = [est_beta_map[x] for x in est_beta_idx]
    orig_beta = orig_beta_data[est_beta_idx]
    final_result = computeNMSE(est_beta, orig_beta)

    end = time.time()
    evaluate_time = end - start
    
#4. statistics
    statistics = "In KMM method, mode=%s, train_size=%i, test_size=%i, tr_bag_size=%i, m=%i, te_bag_size=%i, n=%i\n" % \
                 (mode, len(train_data), len(test_data), tr_bag_size, tr_bag_no, te_bag_size, te_bag_no)
    total_time = bagging_time + compute_time + evaluate_time
    time_info = "bagging_time=%s, compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                (bagging_time, compute_time, evaluate_time, total_time)
    print statistics
    print time_info
    
    message = "The final NMSE for KMM is : %s \n" % final_result
    print message
        
    print "---------------------------------------------------------------------------------------------"
    
    with open(output_file, 'a') as output_file:
        output_file.write(statistics)
        output_file.write(time_info)
        output_file.write(message)
