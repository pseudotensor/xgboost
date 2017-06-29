# pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
import time
import feather
import pandas as pd

def run_benchmark(args, gpu_algorithm, cpu_algorithm):


    from os.path import expanduser
    #home = str(expanduser("~"))
    filepostfixx="featherx"
    filepostfixy="feathery"

    if False:
        print("Generating dataset: {} rows * {} columns".format(args.rows,args.columns))
        tmp = time.time()
        X, y = make_classification(args.rows, n_features=args.columns, random_state=7)
        print ("Make Data Time: %s seconds" % (str(time.time() - tmp)))
        dfX= pd.DataFrame(np.array(X.data))
        dfy= pd.DataFrame(np.array(y.data))
        
        feather.write_dataframe(dfX, filepostfixx)
        feather.write_dataframe(dfy, filepostfixy)

        
    if False:
        df=np.array(pd.read_csv("HIGGS.csv"))
        #df = np.loadtxt( 'HIGGS.csv', delimiter=',', )

        # Put Y(truth), X(data), W(weight), and I(index) into their own arrays
        print('Assigning data to numpy arrays.')
        # Pick a random seed for reproducible results. Choose wisely!
        np.random.seed(42)
        # Random number for training/validation splitting
        r =np.random.rand(df.shape[0])
        rfrac=1.0
         
        # First 90% are training
        Y_train = df[:,28][r<rfrac]
        X_train = df[:,0:28][r<rfrac]
        # Last r fraction are validation
        Y_valid = df[:,28][r>=rfrac]
        X_valid = df[:,0:28][r>=rfrac]
        
        feather.write_dataframe(pd.DataFrame(X_train), filepostfixx)
        feather.write_dataframe(pd.DataFrame(Y_train), filepostfixy)
        
    if True:
        tmp = time.time()
        dfX = feather.read_dataframe(filepostfixx)
        dfy = feather.read_dataframe(filepostfixy)


    if True:
        dtrain = xgb.DMatrix(dfX.values, dfy.values)
        print ("DMatrix Time: %s seconds" % (str(time.time() - tmp)))
        
        param = { 'max_depth': 6,
                 'silent': 0,
                 'verbose': 2,
                 'n_gpus': 1,
                 'gpu_id': 0,
                 'eval_metric': 'auc'}
        
        param['tree_method'] = gpu_algorithm
        print("Training with '%s'" % param['tree_method'])

        tmp = time.time()
        xgb.train(param, dtrain, args.iterations)
        print ("Time: %s seconds" % (str(time.time() - tmp)))
        
        #param['tree_method'] = cpu_algorithm
        #print("Training with '%s'" % param['tree_method'])
        #tmp = time.time()
        #xgb.train(param, dtrain, args.iterations)
        #print ("Time: %s seconds" % (str(time.time() - tmp)))



parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['all', 'gpu_exact', 'gpu_hist'], default='all')
parser.add_argument('--rows',type=int,default=8000000)
parser.add_argument('--columns',type=int,default=30)
parser.add_argument('--iterations',type=int,default=300)
args = parser.parse_args()

if 'gpu_hist' in args.algorithm:
    run_benchmark(args, args.algorithm, 'hist')
elif 'gpu_exact' in args.algorithm:
    run_benchmark(args, args.algorithm, 'exact')
elif 'all' in args.algorithm:
    run_benchmark(args, 'gpu_exact', 'exact')
    run_benchmark(args, 'gpu_hist', 'hist')

