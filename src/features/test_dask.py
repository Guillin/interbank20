import pandas as pd
import pickle as pickle
import gc
import dask.dataframe as dd


import warnings
warnings.filterwarnings('ignore')

from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=2,
                n_workers=4, memory_limit='16GB')
client


INPUT_PATH = "../../data/raw/" #
OUTPUT_PATH = "../../data/processed/" #
NJOBS = -1
VERBOSE = 1
PLOT = False
SEED = 47
CSV_SEP = "," 

# LOAD DATASET
def load_data(input_file):
    data  = dd.read_csv(f"{INPUT_PATH}{input_file}.csv", sep=CSV_SEP, )#.pipe(reduce_mem_usage)
    
    data['index_col'] = data.apply(lambda r: r.key_value.astype(str)+'_'+r.codmes.astype(str), axis=1)
    data = data.set_index('index_col')
    data = data.map_partitions(lambda x: x.sort_index())
    return data

rcc_train = load_data('rcc_train')

print(rcc_train.iloc[:5,:])

print("PIVOT \n")
df = rcc_train.pivot_table(
        values='saldo', 
        index=['kindex_col'], 
        columns='PRODUCTO', 
        aggfunc='sum')

print(df)