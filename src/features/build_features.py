# make_dataset.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
import logging
import pandas as pd
from pathlib import Path
import argparse
import pickle
import config

# Add here whatever lib you need in order to build features
from sklearn.preprocessing import LabelEncoder


# LOAD DATASET
def load_data(input_file):
    data  = pd.read_pickle(f"{config.INPUT_PATH}{input_file}.pkl")

    return data

# SAVE DATASET
def save_data(output_file, data):
    
    with open(config.OUTPUT_PATH + f'{output_file}.pkl','wb') as f:
        pickle.dump(data, f)
    
    return 0

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

def main(input_file, output_file):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('INIT: making features data set from processed data')

    logger.info('RUN: loading data')
    df = load_data(input_file)
   
    logger.info(f'RUN: data size before be processed: {df.shape}')
    
    logger.info(f'RUN: building features')

    features = build_features(df)
    del df 

    logger.info(f'RUN: features size : {features.shape}')

    logger.info(f'RUN: saving features')
    save_data(output_file, features)

    logger.info('END: making features data set has finished.')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file name without extension from processed folder from where features will be built.", type=str)
    parser.add_argument( "--output_file",   required=True, 
        help="Specify output file name which will be saved into features folder.", type=str)

    args = vars(parser.parse_args())
    
    main(input_file=args['input_file'], output_file=args['output_file'])