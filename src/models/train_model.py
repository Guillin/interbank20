# train_model.py
import argparse 
import os
import logging
import warnings
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import gc
import pickle
from datetime import datetime
from time import time
import config
import model_dispatcher
import metric_dispatcher
warnings.filterwarnings("ignore")
#from plot_model import plot_roc_cv

# load datasets
def load_data(kind):
    
    if kind == 'classification':
        # read the training data
        X_train =  pickle.load(open( f'{config.INPUT_PATH}X_train.pkl', "rb" )) 
        y_train = pickle.load(open( f'{config.INPUT_PATH}y_train.pkl', "rb" ))

        # read test data
        X_test = pickle.load(open( f'{config.INPUT_PATH}X_test.pkl', "rb" )) 
        y_test = pickle.load(open( f'{config.INPUT_PATH}y_test.pkl', "rb" )) 

        return X_train, y_train, X_test, y_test

    else :
        # kind equal to regression
        
        # read the training data
        X_train =  pickle.load(open( f'{config.INPUT_PATH}X_train.pkl', "rb" )) 
        y_train = pickle.load(open( f'{config.INPUT_PATH}y_train.pkl', "rb" ))

        # read validation data
        X_val = pickle.load(open( f'{config.INPUT_PATH}X_val.pkl', "rb" )) 
        y_val = pickle.load(open( f'{config.INPUT_PATH}y_val.pkl', "rb" )) 


        # read test data
        X_test = pickle.load(open( f'{config.INPUT_PATH}X_test.pkl', "rb" )) 
        y_test = pickle.load(open( f'{config.INPUT_PATH}y_test.pkl', "rb" )) 

        return X_train, y_train, X_val, y_val, X_test, y_test



def train_classif_model(model, X_train, y_train):

    # fit the model on training data
    clf = model.fit(X_train, y_train)

    return clf    

    pass


def test_classif_model(model, metric, X_test, y_test):


    metricfun = metric_dispatcher.metrics_score[metric]
    y_preds = model.predict_proba(X_test)[:, 1]
   

    return metricfun(y_test, y_preds)
    
    pass


def run_regression(model, X_train_vector, Y_train_vector, X_val_vector, Y_val_vector):

    models = []

    FOLDS = len(X_train_vector)


    for fold in range(FOLDS):
        print(f"\n----- Fold: ({fold + 1} / {FOLDS}) -----\n")
        X_trn, X_val = X_train_vector[fold], X_val_vector[fold]
        y_trn, y_val = y_train_vector[fold], y_val_vector[fold]
        

        train_set = lgb.Dataset(
            X_trn,
            label=y_trn,
            categorical_feature=categorical_feature,
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_feature,
        )

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
        )
        models.append(model)

        del X_trn, X_val, y_trn, y_val
        gc.collect()

    return models

    pass

def save_model(model,model_name):
    pickle.dump(model, open(f'{config.MODEL_PATH}/{model_name}.bin', 'wb'))
    return 0

def run(kind, model, metric):
    logger = logging.getLogger(__name__)
    logger.info(f'INIT: train {kind} model')

    if kind == 'classification':
        # load train and validation datasets
        X_train, y_train, X_test, y_test = load_data(kind)   

        # fetch the model from model_dispatcher
        clf = model_dispatcher.models[model]
        
        logger.info(f'RUN: training model - {model}')
        clf = train_classif_model(clf, X_train, y_train)
        
        # testing final model with test dataset
        score = test_classif_model(clf, metric, X_test, y_test)

        logger.info(f'RUN: test result - {metric} - %.3f' % (score))

        # saving final model
        if config.SAVE_MODEL:
            save_model(clf, model)
            logger.info(f'RUN: save model - {model}' )

        logger.info(f'END: train {kind} model' )


        pass
    else:
        # kind equal to regression

        pass


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--kind",   required=True, 
        help="Specify if problem will be a classification or regression. Values: 'classification' or 'regression'", type=str )
    parser.add_argument( "--model",	 required=True, 
        help="Select kind of model to be used to train from model_dispatcher. Ex: 'rf' equal to Random Forest", type=str )
    parser.add_argument( "--metric", 
        help="Metric that will be used to validate the performance of the model. Values must be provided from metric_dispatcher. Ex: 'accuracy'", required=True, type=str )
    args = vars(parser.parse_args())
    

    assert args['kind'] in ['classification', 'regression'], f"'kind' should be 'classification' or 'regression'. {args['kind']} was provided"
    
    run(kind=args['kind'], model=args['model'], metric=args['metric'])