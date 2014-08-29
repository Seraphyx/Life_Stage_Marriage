# -*- coding: utf-8 -*-
"""
Created on Mon Aug 04 18:03:50 2014

@author: grxpark
"""

#from sklearn.datasets import load_digits

import csv
import numpy as np
#from matplotlib import pyplot as plt

import theano
from pylearn2.models import mlp
from pylearn2.models import maxout
from pylearn2.training_algorithms import sgd
from pylearn2.train_extensions import best_params
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.costs.mlp import dropout
from pylearn2.training_algorithms import learning_rule
from pylearn2 import termination_criteria

#from pylearn2.gui import get_weights_report

import pickle
#from pylearn2 import train as train

#import numpy as np
#from random import randint


# Import Data
#from sklearn import datasets
#iris = datasets.load_iris()

#print np.array(iris.data)[0:10]

# Import PICKLED Files
'''
pkl_file = open('/home/seraphyx/Documents/Python/Data/Marriage - Dataset - Train.pkl', 'rb')

ds_train = pickle.load(pkl_file)

pkl_file = open('/home/seraphyx/Documents/Python/Data/Marriage - Dataset - Test.pkl', 'rb')

ds_test = pickle.load(pkl_file)


'''
# Train
train_file_x = []
train_file_y = []
test_file_x = []
test_file_y = []

###############################################################################
# IMPORT Dataset - Train

###############################################################################
# IMPORT Dataset - Train
class XOR(DenseDesignMatrix):
    def __init__(self):
        
        self.class_names = ['0', '1']
        
        # Import CSV
        with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage/Life Stage - ALTRIX - Marriage 5 - X - Train - Clean.csv', 'rb') as csvfile_train_x:
            spamreader1 = csv.reader(csvfile_train_x, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in spamreader1:
                #print ', '.join(row)
                train_file_x.append([float(x) for x in row])
        with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage/Life Stage - ALTRIX - Marriage 5 - y - Train.csv', 'rb') as csvfile_train_y:
            spamreader2 = csv.reader(csvfile_train_y, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in spamreader2:
                #print ', '.join(row)
                train_file_y.append([float(x) for x in row])
        
        # Make Numpy
        X = np.array(train_file_x[1:10000])
        y_pre = np.array(train_file_y[1:10000])
        y = []
        
        # Form Shape of Response Variable
        for i in train_file_y:
            if i == 1:
                y.append([0, 1])
            else:
                y.append([1, 0])
        
        X = np.array(train_file_x)
        y = np.array(y)
        super(XOR, self).__init__(X=X, y=y)
 
 
# Test Import by Printing
ds_train = XOR()
print ds_train.X
print ds_train.y

###############################################################################
# IMPORT Dataset - Test
class XOR_test(DenseDesignMatrix):
    def __init__(self):
        
        self.class_names = ['0', '1']
        
        # Import CSV
        with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage/Life Stage - ALTRIX - Marriage 5 - X - Test - Clean.csv', 'rb') as csvfile_test_x:
            spamreader1 = csv.reader(csvfile_test_x, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in spamreader1:
                #print ', '.join(row)
                test_file_x.append([float(x) for x in row])
        with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage/Life Stage - ALTRIX - Marriage 5 - y - Test.csv', 'rb') as csvfile_test_y:
            spamreader2 = csv.reader(csvfile_test_y, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in spamreader2:
                #print ', '.join(row)
                test_file_y.append([float(x) for x in row])
        
          
        # Make Numpy
        X = np.array(test_file_x[1:len(test_file_x)])
        y_pre = np.array(test_file_y[1:len(test_file_x)])
        y = []
        
        # Form Shape of Response Variable
        for i in y_pre:
            if i == 1:
                y.append([0, 1])
            else:
                y.append([1, 0])
        
        X = np.array(X)
        y = np.array(y)
        super(XOR_test, self).__init__(X=X, y=y)
 
 
# Test Import by Printing
ds_test = XOR_test()
print ds_test.X
print ds_test.y

###############################################################################
# Layers

'''
--------------------------------------------------------------
Best Results:
--------------------------------------------------------------
-Dropout
            input_include_probs = { 'h0' : .5 , 'h1' : .5},
            input_scales        = { 'h0' : 2 , 'h1' : 2})
-Maxout
Nodes: 500 each
Maxout Linear Pieces: 5 each
max_col_norm = 1.9365

-Learning
init_momentum = .5
Epochs = 500
'''

# Maxout Layer
maxout_layer_h0 = maxout.Maxout(layer_name = 'h0',
                                num_units = 200,
                                num_pieces = 5,
                                irange = .005,        # randomly assigns weights within [-irange, irange]
                                max_col_norm = 2)     # Max-norm Regularization

# Maxout Layer
maxout_layer_h1 = maxout.Maxout(layer_name = 'h1',
                                num_units = 200,
                                num_pieces = 5,
                                irange = .005,
                                max_col_norm = 2)

# Create Softmax output layer
output_layer = mlp.Softmax(n_classes = 2, 
                           layer_name = 'y', 
                           irange = .1,
                           max_col_norm = 2)
                           
# Cost Function: Dropout
cost = dropout.Dropout (
            input_include_probs = { 'h0' : .8 ,  
                                    'h1' : .8 ,  
                                    'y'  : .8},
            input_scales        = { 'h0' : 1.25 ,  
                                    'h1' : 1.25 ,  
                                    'y'  : 1.25})
            
# Learning Rule
learning_rule = learning_rule.Momentum(init_momentum = .5)

# Extensions
#extensions_1 = sgd.MomentumAdjustor(start = 1, 
#                                   saturate = 250,
#                                   final_momentum = .7)

extensions_2 = best_params.MonitorBasedSaveBest(channel_name = 'valid_y_misclass',
                                                save_path = "/Users/jasonpark/Documents/Textbooks/Python/Dataset/model_best.pkl")


# Termination Criteria
termination_criterion = termination_criteria.MonitorBased(channel_name = "valid_y_misclass",
                                                          prop_decrease = 0.,
                                                          N = 100)

# Update Callback
update_callbacks = sgd.ExponentialDecay (decay_factor = 1.000004,
                                         min_lr = .000001)

# Minotring Datasets
monitoring_dataset = {'train' : ds_train,
                      'test'  : ds_test}

###############################################################################
# Create Stochastic Gradient Descent trainer that runs for N epochs
trainer = sgd.SGD(learning_rate = .05, 
                  batch_size = 1000, 
                  learning_rule = learning_rule,
                  cost = cost,
                  update_callbacks = update_callbacks,
                  termination_criterion=EpochCounter(25), 
                  #termination_criterion = termination_criterion,
                  monitoring_dataset = monitoring_dataset
                  )

# Combine Layers                  
layers = [maxout_layer_h0, maxout_layer_h1, output_layer]

# create neural net that takes defined inputs
ann = mlp.MLP(layers, 
              nvis=273)

# Train Setup
trainer.setup(ann, ds_train)

###############################################################################
# train neural net until the termination criterion is true

# Initialize
counter = 0
test_y_misclass_best = 1
test_y_misclass_current = 0

# Execute
while True:
    trainer.train(dataset = ds_train)
    ann.monitor.report_epoch()
    ann.monitor()
    
    # Save Best Model after certain epoch
    if counter >= 2:
        # Current
        test_y_misclass_current = ann.monitor.channels['test_y_misclass'].val_record[counter]    
        
        # If current epoch's model is best, save it
        if test_y_misclass_current < test_y_misclass_best:
            test_y_misclass_best = test_y_misclass_current
            
            # Export
            model_best = ann
            output = open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage - Model - Best Model.pkl', 'wb')
            pickle.dump(model_best, output)
            output.close()
    
    # Increment
    counter += 1
    
    if not trainer.continue_learning(ann):
        break

###############################################################################
# EVALUATE
pred      = ann.fprop(theano.shared(ds_test.X, name='inputs')).eval()
pred_best = model_best.fprop(theano.shared(ds_test.X, name='inputs')).eval()
#print pred

###############################################################################
# Monitor Diagnostics
results = []

channels = ['test_y_misclass','train_y_misclass']
results.append(channels)

# Save Results
for index, res in enumerate(ann.monitor.channels['test_y_misclass'].val_record):
    record = [res, 
              ann.monitor.channels['train_y_misclass'].val_record[index]]
    results.append(record)

print results

print 'get_params'
print ann.get_params()

###############################################################################
# Export Results
with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage - Y - Test - Output - Pylearn.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in pred:
        writer.writerow(val)

with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage - Y - Test - Output - Pylearn - Best.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in pred_best:
        writer.writerow(val)

with open('/Users/jasonpark/Documents/Textbooks/Python/Dataset/Marriage - Y - Test - Output - Pylearn - Monitor.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in results:
        writer.writerow(val)
