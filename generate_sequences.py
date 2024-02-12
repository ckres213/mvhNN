# -*- coding: utf-8 -*-
# !/usr/bin/python
#
"""
@author: Matt Pyper
"""

import pickle
import time
import numpy as np
import os
import scipy.io
from collections import defaultdict
import sequence_generator as seq_generator
import datetime
import argparse
import pdb

__author__ = 'Matt Pyper'

def main():
    # Arguments user specifies about sequence generated
    parser = argparse.ArgumentParser(description='Generating sequences... ')
    
    parser.add_argument(
        '-m', '--ModelGen', #required=True,
        default = 'conttime',
        type = str,
        choices = ['hawkes', 'hawkesinhib', 'conttime'],
        help='Model used to generate data'
    )
    
    parser.add_argument(
        '-st', '--SumForTime',
        default=False, type=bool, choices=[True, False],
        help='Do we use total intensity for time sampling? True -- Yes; False -- No'
    )
    
    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)'
    )
    
    parser.add_argument(
        '-s', '--Seed',
        default = 12345, type = int,
        help='Seed of random state'
    )
    
    parser.add_argument(
        '-k', '--DimProcess',
        default = 5, type = int,
        help='Number of event types'
    )
    
    parser.add_argument(
        '-u', '--LSTMUnits',
        default = 8, type = int,
        help='Number of units in LSTM'
    )
    
    parser.add_argument(
        '-N', '--NumSequences',
        default = 10000, type = int,
        help='Number of sequences to simulate'
    )
    
    parser.add_argument(
        '-min', '--MinLength',
        default = 20, type = int,
        help='Min number of events (length) per sequence'
    )
    
    parser.add_argument(
        '-max', '--MaxLength',
        default = 100, type = int,
        help='Max number of events (length) per sequence'
    )
    
    parser.add_argument(
        '-tts', '--TrainTestSplit',
        default=[0.7, 0.1, 0.2], type=float, nargs='+',
        help='Train, Dev, and Test Splits as decimals'
    )

        
    args = parser.parse_args()
    
    # Specifying datatype of arguments 
    seed = np.int32(args.Seed)
    dim_process = np.int32(args.DimProcess)
    lstm_units = np.int32(args.LSTMUnits)
    num_sequences = np.int32(args.NumSequences)
    min_len = np.int32(args.MinLength)
    max_len = np.int32(args.MaxLength)
    model = str(args.ModelGen)
    pretrain_model_path = args.FilePretrain
    sum_for_time = bool(args.SumForTime)
    
    train_pct = args.TrainTestSplit[0]
    dev_pct = args.TrainTestSplit[0] + args.TrainTestSplit[1]
    test_pct = args.TrainTestSplit[2]
    
    
    # Creating tag for model
    if model == "conttime":
        tag_model = f'{model}_{num_sequences}_Seq_{dim_process}_Dim_{max_len}_MaxLen_{lstm_units}_LSTMUnits'
        
    else:
        tag_model = f'{model}_{num_sequences}_Seq_{dim_process}_Dim_{max_len}_MaxLen'

    # Creating data and model generator files
    file_save = os.path.abspath(f'./data/{tag_model}.pkl')
    file_model = os.path.abspath(f'./gen_models/{tag_model}.pkl')

    # Saving arguments to input into sequence_generator methods
    generator_settings = {
        'dim_process': dim_process,
        'lstm_units': lstm_units,
        'seed': seed,
        'pretrain_model_path': pretrain_model_path,
        'sum_for_time': sum_for_time,
        'args': None  # Update this if 'args' is something specific
    }

    sequence_settings = {
        'num_sequences': num_sequences,
        'min_len': min_len,
        'max_len': max_len
    }

    
    # Initializes parameter space for specified model
    if model == 'hawkes':
        gen_model = seq_generator.HawkesGen(generator_settings)
        
    elif model == 'hawkesinhib':
        gen_model = seq_generator.HawkesInhibGen(generator_settings)
        
    elif model == 'conttime':
        gen_model = seq_generator.NeuralHawkesCTLSTM(generator_settings)
    
    print("initialization done")
    
    # Show user information about the model
    print( "Seed is: %s" % str(seed) )
    print( "FilePretrain is: %s" % pretrain_model_path )
    print( "Generator is: %s" % model )
    print( "FileSave is: %s" % file_save )
    print( "FileModel is: %s" % file_model )
    print( "DimProcess is: %s" % str(dim_process) )
    
    if 'conttime' in model:
        print ("LSTMUnits is: %s" % str(lstm_units) )
    
    print( "Number of Sequences are: %s" % str(num_sequences) )
    print( "MinLen is: %s" % str(min_len) )
    print( "MaxLen is: %s" % str(max_len) )
    print( "SumForTime is: %s" % str(sum_for_time) )

    
    # Saves arguments for sequence generation
    dict_args = {
        'ModelGen': model,
        'FileSave': file_save,
        'FileModel': file_model,
        'DimProcess': dim_process,
        'LSTMUnits': lstm_units,
        'Seed': seed,
        'FilePretrain': pretrain_model_path,
        'NumSequences': num_sequences,
        'MinLen': min_len,
        'MaxLen': max_len,
        'SumForTime': sum_for_time
    }
    
    # Stores arguments in model class
    gen_model.set_args(dict_args)
    
    
    print(f"The number of sequences in the train, dev, and test sets are : {int(num_sequences*train_pct)}, {int(num_sequences*(dev_pct - train_pct))}, {int(num_sequences*test_pct)}")

    # Generate sequence
    time_0 = time.time()
    gen_model.gen_sequences(sequence_settings)
    time_1 = time.time()
    time_taken = time_1 - time_0
    
    
    gen_model.save_model(file_model)
    
    # Splitting up data
    dict_data = {
        'train': gen_model.list_sequences[:int(num_sequences*train_pct)],
        'dev': gen_model.list_sequences[int(num_sequences*train_pct):int(num_sequences*(dev_pct))],
        'test': gen_model.list_sequences[int(num_sequences*(dev_pct)):],
        #'test1': gen_model.list_sequences[cut_test:], #cb
        'args': dict_args
    }
    
    print( "saving ... " )
    
    with open(file_save, 'wb') as f:
        pickle.dump(dict_data, f)

    print( "finished ! Took {} seconds !!!".format(str(round(time_taken, 2))) )

if __name__ == "__main__": main()
