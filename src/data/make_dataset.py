# make_dataset.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
import logging
import pandas as pd
from pathlib import Path
import argparse
import config
import gc


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

# LOAD DATASET
def load_data(input_file):
    data  = pd.read_csv(f"{config.INPUT_PATH}{input_file}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)

    return data

# SAVE DATASET
def save_data(output_file, data):
    
    data.to_pickle(config.OUTPUT_PATH + f'{output_file}.pkl')
    
    return 0

# PROCESS DATA
def process_data(data):
    """ You must define here how to process the dataframe before be saved
    """
    # drop duplicates
    data.drop_duplicates(inplace=True)
    
    # ****************************************************** # 
    # put here what you think is needed to process your data
    data["target"] = data["y"].map({"no":0, "yes":1})
    data.drop('y', axis=1, inplace=True)
    # ****************************************************** #

    return data
    pass

def main(input_file, output_file, ):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('INIT: making final data set from raw data')

    logger.info('RUN: Loading TRAIN DATA')
    data = load_data(input_file)
    if 
    y_train = load_data('y_train')


  
    logger.info(f'RUN: Data size before be processed: {rcc_train.shape}')
    
    logger.info(f'RUN: Processing TRAIN DATA')
    rcc_train.drop_duplicates(inplace=True)
    df = pd.merge(rcc_train, se_train, how='left', on='key_value')
    
    df = pd.merge(df, se_train, how='left', on='key_value')

    del rcc_train, se_train, censo_train,y_train
    gc.collect()
    

    logger.info(f'RUN: Data size after be processed: {df.shape}')

    logger.info(f'RUN: Saving data')
    save_data(df)

    logger.info('END: raw data processed.')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_file",   required=True, 
        help="Specify file name without extension from raw folder to be processed.", type=str)

    parser.add_argument( "--type",   required=True, 
        help="Specify file universer from input file [train|test].", type=str)

    parser.add_argument( "--output_file",   required=True, 
        help="Specify output file name which will be saved into folder processed.", type=str)

    args = vars(parser.parse_args())
    assert args['type'] in ['train', 'test'], f"'type' should be 'train' or 'test'. {args['type']} was provided"
    
    main(input_file=args['input_file'], output_file=args['output_file'], type=args['type'])