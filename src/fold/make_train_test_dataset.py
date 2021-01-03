# make_train_test_dataset.py
#!/usr/bin/env python
# coding: utf-8

# Import libraries
import logging
import argparse 
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split  #Additional scklearn functions
import config


# DEF FUNCTIONS

# LOAD DATASET
def load_data(file_name):
    X = pd.read_pickle(f"{config.INPUT_PATH}{file_name}.pkl").drop(config.target, axis=1)
    X.columns = [str(c) for c in X.columns]

    y = pd.read_pickle(f"{config.INPUT_PATH}{file_name}.pkl")[config.target]
    return X,y

# SAVE TRAIN & TEST
def save_and_split_data(X,y):
    # SPLIT IN TRAIN/TEST
    # set the seed of random number generator, which is useful for creating simulations 
    # or random objects that can be reproduced.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)

    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape

    X_train.to_pickle(config.OUTPUT_PATH + 'X_train.pkl')
    del X_train

    X_test.to_pickle(config.OUTPUT_PATH + 'X_test.pkl')
    del X_test

    y_train.to_pickle(config.OUTPUT_PATH + 'y_train.pkl')
    del y_train

    y_test.to_pickle(config.OUTPUT_PATH + 'y_test.pkl')
    del y_test

    return X_train_shape, X_test_shape, y_train_shape, y_test_shape
    


def main(file_name):

    logger = logging.getLogger(__name__)
    logger.info('INIT: split dataset in train and test')

    logger.info('RUN: loading data')

    # LOAD DATASET
    X,y = load_data(file_name)

    logger.info(f'RUN: data size before be splited: X {X.shape}, y {y.shape}')

    logger.info(f'RUN: spliting and saving dataset')

    X_train_shape, X_test_shape, y_train_shape, y_test_shape = save_and_split_data(X,y) 
    
    logger.info(f'RUN: data size after be splited: X_train {X_train_shape}, y_train {y_train_shape}, X_test {X_test_shape}, y_test {y_test_shape}')

    logger.info('END: split dataset has finished')

    pass


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file from features folder to be splited without extension.", type=str )
    
    args = vars(parser.parse_args())
    
    main(file_name=args['input_file'])