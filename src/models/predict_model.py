# predict_model.py
# desc: load model who has been build and score test (submit) dataset
# @autor: glissa
# created: 2021/01/07
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
from sklearn.model_selection import cross_validate
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


def get_next_filename(model):
    '''Gets the next numeric filename in a sequence.

    All files in the output directory must have the same name format,
    e.g. "txt1.txt".
    '''

    n = 0
    for f in os.listdir(config.SUBMIT_PATH):
        n = max(n, int(get_num_part(os.path.splitext(f)[0])))
    return f"submit_{model}_n{(n + 1)}.csv" 


def get_num_part(s):
    '''Get the numeric part of a string of the form "abc123".

    Quick and dirty implementation without using regex.'''

    for i in range(len(s)):
        if s[i:].isdigit():
            return s[i:]
    return ''


# REDUCE MEMORY USAGE
def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# load datasets
def load_data(file_name):
    
    return pd.read_pickle(f"{config.TEST_PATH}{file_name}.pkl").pipe(reduce_mem_usage)


def load_model(modelname):

    # load the model from disk
    loaded_model = pickle.load(open(f"{config.MODEL_PATH}{modelname}.bin", 'rb'))

    return loaded_model

    pass

def predict():
    # TODO: Regression models predict
    return 0
    pass


def predict_proba(model, data):

    y_pred = model.predict_proba(data)[:,1]

    return y_pred
    
    pass

def save_predic_proba(pred_proba, model, test):
    # you should define this function
    test['target'] = pred_proba
    test.reset_index(inplace=True)
    submit = test[['key_value','target']]

    # get name of next submit file
    submit_filename = get_next_filename(model)

    submit.to_csv(f"{config.SUBMIT_PATH}{submit_filename}", index=False)

    if config.KAGGLE_SUBMIT:
        os.system(f"kaggle competitions submit -c {config.KAGGLE_PROJECT} -f {config.SUBMIT_PATH}{submit_filename} -m 'submit file: {submit_filename} from model: {model}'")
    pass

# def save_models(model_vector,model_name):
#     pickle.dump(model_vector, open(f'{config.MODEL_PATH}/{model_name}.bin', 'wb'))
#     return 0

# def run_regression(model, X_train_vector, Y_train_vector, X_val_vector, Y_val_vector):

#     models = []

#     FOLDS = len(X_train_vector)


#     for fold in range(FOLDS):
#         print(f"\n----- Fold: ({fold + 1} / {FOLDS}) -----\n")
#         X_trn, X_val = X_train_vector[fold], X_val_vector[fold]
#         y_trn, y_val = y_train_vector[fold], y_val_vector[fold]
        

#         train_set = lgb.Dataset(
#             X_trn,
#             label=y_trn,
#             categorical_feature=categorical_feature,
#         )
#         val_set = lgb.Dataset(
#             X_val,
#             label=y_val,
#             categorical_feature=categorical_feature,
#         )

#         model = lgb.train(
#             bst_params,
#             train_set,
#             valid_sets=[train_set, val_set],
#             valid_names=["train", "valid"],
#             **fit_params,
#         )
#         models.append(model)

#         del X_trn, X_val, y_trn, y_val
#         gc.collect()

#     return models

#     pass



def run(kind, model, input_file):
    logger = logging.getLogger(__name__)
    logger.info(f'INIT: predict {model} model')

    if kind == 'classification':
        # load train and validation datasets
        test = load_data(input_file)   

        logger.info(f'RUN: loading model: {model}')
        # fetch the model from model_dispatcher
        clf = load_model(model)

        logger.info(f'RUN: predic model: {model}')
        y_pred = []
        if type(clf)==tuple or type(clf)==list:
            for m in clf:
                y_pred.append(list(predict_proba(m, test)))
            
            y_pred = list(mean(y_pred,axis=0))
        
        else:
            y_pred  = predict_proba(clf,test)


        
        # saving final dataset wiht proba or class
        #if (condition)
        
        logger.info(f'SAVE PREDICTION: {model}' )
        save_predic_proba(y_pred, model, test)

        
        logger.info(f'END: predict {model} model' )
       

        pass
    else:
        #  equal for regression

        pass


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--kind",   required=True, 
        help="Specify if problem will be a classification or regression. Values: 'classification' or 'regression'", type=str )
    parser.add_argument( "--model",	 required=True, 
        help="Select kind of model to be used to score from model_dispatcher. Ex: 'rf' equal to Random Forest", type=str )
    
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file name without extension who will be scored by model.", type=str)
    
    args = vars(parser.parse_args())
    

    assert args['kind'] in ['classification', 'regression'], f"'kind' should be 'classification' or 'regression'. {args['kind']} was provided"

    run(kind=args['kind'], model=args['model'], input_file=args['input_file'])