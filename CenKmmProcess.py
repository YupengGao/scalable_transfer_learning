#!/usr/bin/env python
#from pyspark import SparkContext
import argparse
import numpy as np
import time
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE
from lib.scaleKMM import *;
from lib.util import *

def cenKmmProcess():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default='/home/wzyCode/scalablelearning/nmseCenKMM.txt', help="default output file")
    args = parser.parse_args()
    output_file = args.output
    
    train_data = np.loadtxt("train_data.txt")
    test_data = np.loadtxt("test_data.txt")
    orig_beta_data = np.loadtxt("orig_beta_data.txt")
    
#Step 1: Compute the estimated beta from cenKMM
    start = time.time()
    maxFeature = train_data.shape[1]
    gammab = computeKernelWidth(train_data)
    res = cenKmm(train_data, test_data, gammab, maxFeature)
    est_Cenbeta = res[0]

    end = time.time()
    compute_time_Cen = end-start
    
    
# Step 2: Compute the NMSE between the est_beta and orig_beta through CenKMM
    start = time.time()
    final_result_Cen = computeNMSE(est_Cenbeta, orig_beta_data)
    end = time.time()
    evaluateCen_time = end - start
    
    
# Step 3: statistics
    statisticsCen = "In CenKMM method, train_size=%i, test_size=%i" % \
                 (len(train_data), len(test_data))
    total_time = compute_time_Cen + evaluateCen_time
    time_info_Cen = "compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                (compute_time_Cen, evaluateCen_time, total_time)
    print statisticsCen
    print time_info_Cen
    
    messageCen = "The final NMSE for CenKMM is : %s \n" % final_result_Cen
    print messageCen
    
    print "---------------------------------------------------------------------------------------------"
    
    with open(output_file, 'a') as output_file:
        output_file.write(statisticsCen)
        output_file.write(time_info_Cen)
        output_file.write(messageCen)