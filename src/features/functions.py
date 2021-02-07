# functions.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
# many functions useful for scripts in features folder
import pandas as pd
from pathlib import Path
import argparse
import pickle
import config
import gc
import os
from google.cloud import storage


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
def load_data(input_file, kind='csv'):
    data = pd.DataFrame([])

    if kind=='csv':
        data  = pd.read_csv(f"{config.INPUT_PATH}{input_file}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)
    elif kind=='pickle':
        data  = pd.read_pickle(f"{config.INPUT_PATH}{input_file}.pkl").pipe(reduce_mem_usage)
    elif kind=='parquet':
        data  = pd.read_parquet(f"{config.INPUT_PATH}{input_file}.parquet").pipe(reduce_mem_usage)
    else:
        raise Exception(f"`kind` should be csv, pickle or parquet. `{kind}` value is not allowed.") 
    return data

# SAVE DATASET
def save_data(output_file, data):
    data.to_pickle(f"{config.OUTPUT_PATH}{output_file}.pkl")

    if config.USE_GCP:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(config.GCP_BUCKET_NAME)

        blob = bucket.blob(f"{config.GCP_BUCKET_FOLDER_NAME}{output_file}.pkl")
        blob.upload_from_filename(f"{config.OUTPUT_PATH}{output_file}.pkl")

        # remove local file because it is now in the bucket
        os.remove(f"{config.OUTPUT_PATH}{output_file}.pkl")
    
    return 0

