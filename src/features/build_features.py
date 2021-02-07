# make_dataset.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
import logging
import pandas as pd
from pathlib import Path
import argparse
import pickle
import gc
import os
from google.cloud import storage
from functions import load_data, save_data
import config


# PROCESS DATA
def build_features(data):
    """ You must define here how to build any features engineer from data
        :data: dataset from where features will be made
        :features: final dataset with all features eng
    """
    
    # ****************************************************** # 
    # put here what you think is needed to build features 
    # from your processed data
    # Ex: 
    # features = data.copy()
    #for cat in config.cat_features:
    #    le = LabelEncoder()
    #    le.fit(list(data[cat].unique()))
    #    features[cat] = le.transform(data[cat])
    #features = preprocessor.fit_transform(data)
    #features = pd.DataFrame(data=features)
    # ****************************************************** #

    return features
    pass

def main(output_file):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('INIT: making features data set from processed data')

    logger.info('RUN: loading data')
    se = load_data('se_train', kind='pickle')
    rcc = load_data('rcc_train_featengv6', kind='pickle')
    censo = load_data('censo_train', kind='pickle')
    sunat = load_data('sunat_train_featengv1', kind='pickle')
    target = load_data('y_train', kind='pickle')


    
    #logger.info(f'RUN: data size before be processed: {df.shape}')
    
    logger.info(f'RUN: building features')

    #features = build_features(df)

    gc.collect()  
    features = pd.merge(rcc, se, how='left', on='key_value')
    features = pd.merge(features, censo, how='left', on='key_value')
    features = pd.merge(features, sunat, how='left', on='key_value')
    features = pd.merge(features, target, how='left', on='key_value')
    features = features.fillna(-9999)


    logger.info(f'RUN: features size : {features.shape}')

    del se ,rcc, censo, sunat

    logger.info(f'RUN: saving features')
    save_data(output_file, features)

    logger.info('END: making features data set has finished.')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser()
    #parser.add_argument( "input",   required=False,
#        help="Specify file names without extension from processed folder from where features will be built.", type=str)
    parser.add_argument( "--output_file",   required=True, 
        help="Specify output file name which will be saved into features folder.", type=str)

    args = vars(parser.parse_args())
    
    #main(input_file=args['input_file'], output_file=args['output_file'])
    main(output_file=args['output_file'])