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


def ensKmmProcess():
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-b', "--bagging", type=int, choices=[1,2,3,4], default=1, help="bagging strategy")
    parser.add_argument("-s", "--tr_bsize", type=int, help="the sample size of train set")
    parser.add_argument("-x", "--te_bsize", type=int, help="the sample size of test set")
    parser.add_argument("-m", "--train_samples", type=int, help="number of samples from training")
    parser.add_argument("-n", "--test_samples", type=int, help="number of samples from test")
    #parser.add_argument("-i", "--input", type=str, default='./dataset/powersupply.arff', help="default input file")
    parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/nmseEnsKmm.txt', help="default output file")
    args = parser.parse_args()
    
    tr_bsize = args.tr_bsize # By default, the train bag size is dynamic, if specified, the train bag size will fix
    te_bsize = args.te_bsize # By default, the test bag size is dynamic, if specified, the test bag size will fix
    m = args.train_samples # take m samples from training
    n = args.test_samples # take n samples from test
    output_file = args.output
    
    #1.Bagging process
    train_data = np.loadtxt("train_data.txt")
    test_data = np.loadtxt("test_data.txt")
    orig_beta_data = np.loadtxt("orig_beta_data.txt")


    testDataLength = len(test_data)
    te_bag_size = testDataLength/n
    te_bsizeValue = sc.broadcast(te_bag_size)
    
    # Bagging the train and test data from the sampled index
    start = time.time()
    tr_bag_size_ens = len(train_data)
    tr_bag_no_ens = 1
    te_bag_size_ens, te_bag_no_ens = get_size_no(test_data, te_bsize, n)
    
    tr_n_ens = partition(train_data, part_size=tr_bag_size_ens, part_no=tr_bag_no_ens)
    te_n_ens = partition(test_data, part_size=te_bag_size_ens, part_no=te_bag_no_ens)
    
    bags_ens = cartesian(train_data, test_data, tr_n_ens, te_n_ens)
    rddEns = sc.parallelize(bags_ens)
    
    end = time.time()
    ens_bagging_time = end - start
    
    
#2. Compute Beta Process
    start = time.time()
    #rddEns = rddEns.map(lambda (idx, tr, te): (len(idx), len(tr), len(te)))
    #print "rddEns",rddEns.take(5)
    #print "te_bsizeValue",te_bsizeValue.value
    rddEns = rddEns.map(lambda (idx, tr, te): getEnsKmmBeta(idx, tr, te, te_bsizeValue.value)).flatMap(lambda x: x)
  
    rddEns = rddEns.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),
                              lambda a,b: (a[0] + b[0], a[1] + b[1]))
  
    est_Ensbeta_map = rddEns.mapValues(lambda v: v[0]/v[1]).collectAsMap()
    est_Ensbeta_idx = est_Ensbeta_map.keys()
    end = time.time()
    compute_time_Ens = end - start
    
    
#3. Compute the NMSE between the est_beta and orig_beta through KMM
    start = time.time()
     
    est_Ensbeta = [est_Ensbeta_map[x] for x in est_Ensbeta_idx]
    orig_beta = orig_beta_data[est_Ensbeta_idx]
    final_result_Ens = computeNMSE(est_Ensbeta, orig_beta)
 
    end = time.time()
    evaluateEns_time = end - start
    
#4. statistics
    statisticsEns = "In EnsKMM method, train_size=%i, test_size=%i, tr_bag_size=%i, m=%i, te_bag_size=%i, n=%i\n" % \
                 (len(train_data), len(test_data), tr_bag_size_ens, tr_bag_no_ens, te_bag_size_ens, te_bag_no_ens)
    total_time =  ens_bagging_time + compute_time_Ens + evaluateEns_time
    time_info_Ens = "bagging_time=%s, compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                ( ens_bagging_time, compute_time_Ens, evaluateEns_time, total_time)
    print statisticsEns
    print time_info_Ens
     
    messageEns = "The final NMSE for EnsKMM is : %s \n" % final_result_Ens
    print messageEns
    
    with open(output_file, 'a') as output_file:
        output_file.write(statisticsEns)
        output_file.write(time_info_Ens)
        output_file.write(messageEns)