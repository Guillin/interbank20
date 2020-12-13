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



def cv_classification(model, metric, folds, X, y, plot_roc=False):

    cvscores = []
    cv = StratifiedKFold(n_splits=folds, random_state=config.SEED)

    metricfun = metric_dispatcher.metrics_score[metric]

    if plot_roc and metric=='roc_auc':

        plot_roc_curve_cv(model, X, y, cv)
    
    else:
        cvscores = cross_val_score(model, X, y, scoring=make_scorer(metricfun, greater_is_better=True), cv=cv, n_jobs=config.NJOBS, error_score='raise', verbose=config.VERBOSE)
    
    return cvscores

    pass


def test_classification(model, metric, X_train, y_train, X_test, y_test):


    # fit the model on training data
    model.fit(X_train, y_train)

    metric_function = metric_dispatcher[metric]

    y_preds = []

    # create predictions for validation samples
    if metric in ['roc_auc']:
        y_preds = model.predict_proba(X_test)
    else:
        y_preds = model.predict(X_test)


    return metric_function(y_test, y_preds)
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



def run(kind, model, folds, metric):
    logger = logging.getLogger(__name__)
    logger.info(f'INIT: train {kind} model')

    if kind == 'classification':
        # load train and validation datasets
        X_train, y_train, X_test, y_test = load_data(kind)   

        # fetch the model from model_dispatcher
        clf = model_dispatcher.models[model]
        
        logger.info(f'RUN: training model: {model}')
        cv_scores = cv_classification(model=clf, metric=metric, folds=folds, X=X_train, y=y_train, plot_roc=config.PLOT) 
        logger.info(f'RESULT: {metric} - mean %.3f - std (%.3f)' % (mean(cv_scores), std(cv_scores)))

        logger.info('END: train model' )
       

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
    parser.add_argument( "--folds",  
        help="Number of folds to be used in order to implement Cross Validation. This arg must be provided only if you are using classification, for regression problems, folds must be done using time split methods.", required=False, type=int )
    parser.add_argument( "--metric", 
        help="Metric that will be used to validate the performance of the model. Values must be provided from metric_dispatcher. Ex: 'accuracy'", required=True, type=str )
    args = vars(parser.parse_args())
    

    assert args['kind'] in ['classification', 'regression'], f"'kind' should be 'classification' or 'regression'. {args['kind']} was provided"
    
    run(kind=args['kind'], model=args['model'], folds=args['folds'], metric=args['metric'])