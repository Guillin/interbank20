# make_dataset.py
#!/usr/bin/env python
# coding: utf-8
# Import libraries
import logging
import pandas as pd
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from pathlib import Path
import argparse
import pickle
import gc
from tqdm import tqdm
import datetime
from dateutil.relativedelta import relativedelta
import config
import os
from google.cloud import storage

# Add here whatever lib you need in order to build features
from sklearn.preprocessing import LabelEncoder

# GET SIMPLE STATISTICS VALUES
def get_statistics(df, groupby, c, statfunc=['min', 'max']):
    df_group = df.groupby(groupby)
    dfstat = df_group.agg({c:statfunc})
    dfstat.columns = [f"{c}_{v}" for v in statfunc]
    dfstat.reset_index(inplace=True)
    
    return dfstat

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

def add_cat_feateng(df, groupBy='key_value'):

    # count
    df_counts = df.groupby(groupBy).count()
    df_counts.columns = [c + '_count_catvar' for c in df_counts.columns]

    # max and min
    df_max = df.groupby(groupBy).max()
    df_max.columns = [c + '_max_catvar' for c in df_max.columns]
    
    df_min = df.groupby(groupBy).min()
    df_min.columns = [c + '_min_catvar' for c in df_min.columns]

    df_ptp = df.groupby(groupBy).agg(np.ptp)
    df_ptp.columns = [c + '_ptp_catvar' for c in df_ptp.columns]

 
    return df_counts.join(df_max).join(df_min).join(df_ptp)
    pass

def add_num_feateng(df):

    c = df.columns[0]
    df.reset_index(inplace=True)
    groupBy='key_value'  # TODO: harcodeado, ver como pasar como parametro

    for p in [1, 3]:
        df[f"{c}_diff_p{p}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.diff(p)
        )
    
    # # sum 
    # df[f"{c}_sum"] = df.groupby(groupBy)[c].transform(
    #         lambda x: x.sum()
    #     )
    
    # count
    df[f"{c}_count"] = df.groupby(groupBy)[c].transform(
            lambda x: x.count()
        )

    # Moving sum 
    for window in [3, 6]:
        df[f"{c}_rolling_sum_t{window}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.expanding(window).sum(skipna=True)
        )

    # Moving average
    for window in [3, 6]:
        df[f"{c}_rolling_mean_w{window}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.rolling(window).mean(skipna=True)
        )

    # Moving std 
    for window in [3, 6]:
        df[f"{c}_rolling_std_w{window}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.rolling(window).std(skipna=True)
        )
    
    # Percentage change between the current and a prior element.
    for p in [1, 3, 6]:
        df[f"{c}_pct_p{p}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.pct_change(periods=p, fill_method='pad').replace(np.inf,0).replace(-np.inf,0).replace(np.nan,0)
        )

    
    df[f"{c}_skew"] = df.groupby(groupBy)[c].skew(skipna = True)
    

    # TODO: harcodeado, ver como pasar como parametro
    df.set_index(['key_value', 'codmes'], inplace=True) 

    # para saber si es una variable numerica o categorica
    df.columns =  [c + '_num' for c in df.columns]
    return df

# replace outliers with a default value
def replace_outliers(df):
    for x in df.columns:
        q75,q25 = np.percentile(df.loc[:,x],[75,25])
        intr_qr = q75-q25
    
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
    
        df.loc[df[x] < min,x] = min
        df.loc[df[x] > max,x] = max
    
    return df


# LOAD DATASET
def load_data(input_file):
    data  = pd.read_csv(f"{config.INPUT_PATH}{input_file}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)

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

# PROCESS DATA
def build_features(data, output_file):
    """ You must define here how to build any features engineer from data
        :data: dataset from where features will be made
        :features: final dataset with all features eng
    """
    NCORES = 0
    if config.NJOBS != -1 and config.NJOBS <= cpu_count():
        NCORES = config.NJOBS
    else:
        NCORES = cpu_count()
    
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

    data.sort_values(['key_value', 'codmes'], inplace=True, ascending=True)

    # id producto from test now equal to train 
    data.PRODUCTO.fillna(255,inplace=True)
    data.loc[:,'PRODUCTO'] = data.PRODUCTO.astype(int)


    # codmes
    codmes_df = get_statistics(data, 'key_value', 'codmes', statfunc=['min', 'max'])
    codmes_df['key_value_timespam'] = codmes_df['codmes_max'] - codmes_df['codmes_min'] 

    # condicion
    condicion_df = get_statistics(data, 'key_value', 'condicion', statfunc=['min', 'max','mean','std'])


    features = pd.merge(codmes_df, condicion_df, how='inner', on='key_value')

    del codmes_df, condicion_df
    gc.collect()


    # saldo
    saldo_df = get_statistics(data, 'key_value', 'saldo', statfunc=['sum','count','min', 'max','mean','std'])

    saldo_x_tab = []
    for col in ['PRODUCTO']:
        dfxtab = data.pivot_table(
                values='saldo', 
                index=['key_value', 'codmes'], 
                columns=col, 
                aggfunc=np.sum)
        
        # rename columns witch the value of each var
        dfxtab.columns = [f"{col}_{c}_saldo_" for c in dfxtab.columns]
        saldo_x_tab.append(dfxtab)


    gc.collect()

    saldo_xtab_df = pd.concat(saldo_x_tab, axis=1)

    saldo_xtab_df.dropna(axis=1,how='all',inplace=True)

    saldo_xtab_df = replace_outliers(saldo_xtab_df)



    # we split the dataframe in n pieces (for each column) in order to process it in each CPU core
    df_split = np.array_split(saldo_xtab_df, saldo_xtab_df.shape[1], axis=1)

    del saldo_xtab_df
    gc.collect()


    saldo_num_feateng = []
 
    saldo_num_feateng = []
    for col in tqdm(range(0,len(df_split), NCORES )):
        pool = Pool(NCORES)
        result = pd.concat(pool.map(add_num_feateng, df_split[col:NCORES+col]), axis=1)
        saldo_num_feateng.append(result)
        del result
        gc.collect()
        pool.close()
        pool.join()

    
    saldo_num_feateng = pd.concat(saldo_num_feateng, axis=1)
    saldo_num_feateng.reset_index(inplace=True)

    

    # keep only last reg of each key_value
    saldo_num_feateng = saldo_num_feateng.sort_values('codmes', ascending=True).groupby('key_value').last()
    saldo_num_feateng.reset_index(inplace=True)
    gc.collect()

    saldo_df = pd.merge(saldo_df, saldo_num_feateng, how='left', on='key_value') 
    
    # ==============================================================



    gc.collect()

    features = pd.merge(features, saldo_df, how='left', on='key_value')
    save_data(output_file + '_part1',features)


    del saldo_df, saldo_num_feateng
    gc.collect()

    # =============== Productos by condicion ===================
    condicion_cross_products = data.pivot_table(
                values='condicion', 
                index=['key_value', 'codmes'], 
                columns='PRODUCTO', 
                aggfunc=np.max)


    # rename columns witch the value of each var
    condicion_cross_products.columns = [f"PRODUCTO_{c}_condicion_" for c in condicion_cross_products.columns]
    
    
    
    #condicion_cross_products.reset_index(inplace=True)

    condicion_cross_products.dropna(axis=1,how='all',inplace=True)

    condicion_cross_products = replace_outliers(condicion_cross_products)


    # we split the dataframe in n pieces (for each column) in order to process it in each CPU core
    df_split = np.array_split(condicion_cross_products, condicion_cross_products.shape[1], axis=1)
    gc.collect()


    condicion_num_feateng = []
    for col in tqdm(range(0,len(df_split), NCORES )):
        pool = Pool(NCORES)
        result = pd.concat(pool.map(add_num_feateng, df_split[col:NCORES+col]), axis=1)
        condicion_num_feateng.append(result)
        del result
        gc.collect()
        pool.close()
        pool.join()


    condicion_num_feateng = pd.concat(condicion_num_feateng, axis=1)
    condicion_num_feateng.reset_index(inplace=True)


    #condicion_cross_products = condicion_cross_products.sort_values('codmes', ascending=True).groupby('key_value').last()

    #features = pd.merge(features, condicion_cross_products, how='left', on=['key_value','codmes'])
    

    # keep only last reg of each key_value
    condicion_num_feateng = condicion_num_feateng.sort_values('codmes', ascending=True).groupby('key_value').last()
    condicion_num_feateng.reset_index(inplace=True)
    gc.collect()

    

    #features = pd.merge(features, condicion_num_feateng, how='left', on='key_value')

    save_data(output_file + '_part2', condicion_num_feateng)


    del condicion_cross_products, condicion_num_feateng
    gc.collect()

    # ==============================================================

    # =============== Productos by RIESGO_DIRECTO ===================
    data['RIESGO_DIRECTO'] = data['RIESGO_DIRECTO'].replace(-1,0)
    riesgodir_cross_products = data.pivot_table(
                values='RIESGO_DIRECTO', 
                index=['key_value', 'codmes'], 
                columns='PRODUCTO', 
                aggfunc=np.max)


    # rename columns witch the value of each var
    riesgodir_cross_products.columns = [f"PRODUCTO_{c}_RIESGO_DIRECTO_" for c in riesgodir_cross_products.columns]

    riesgodir_cross_products.dropna(axis=1,how='all',inplace=True)    
    riesgodir_cross_products.reset_index(inplace=True)

    riesgodir_cat_feateng = add_cat_feateng(riesgodir_cross_products)
    riesgodir_cat_feateng.reset_index(inplace=True)

    gc.collect()

    #features = pd.merge(features, riesgodir_cat_feateng, how='left', on='key_value')
    save_data(output_file + '_part3', riesgodir_cat_feateng)


    del riesgodir_cat_feateng, riesgodir_cross_products
    gc.collect()


    # ==============================================================
    

    # =============== Productos by tipo_credito ===================
    tipodcred_cross_products = data.pivot_table(
                values='tipo_credito', 
                index=['key_value', 'codmes'], 
                columns='PRODUCTO', 
                aggfunc=np.max)


    # rename columns witch the value of each var
    tipodcred_cross_products.columns = [f"PRODUCTO_{c}_tipo_credito_cat" for c in tipodcred_cross_products.columns]

    tipodcred_cross_products.dropna(axis=1,how='all',inplace=True)    
    tipodcred_cross_products.reset_index(inplace=True)

    tipocred_cat_feateng = add_cat_feateng(tipodcred_cross_products)
    tipocred_cat_feateng.reset_index(inplace=True)
    gc.collect()


    #features = pd.merge(features, tipocred_cat_feateng, how='left', on='key_value')
    save_data(output_file + '_part4', tipocred_cat_feateng)


    del tipodcred_cross_products, tipocred_cat_feateng
    gc.collect()

    # ==============================================================


    
    # =============== Productos by cod_instit_financiera ===================
    codinstfin_cross_products = data.pivot_table(
                values='tipo_credito', 
                index=['key_value', 'codmes'], 
                columns='PRODUCTO', 
                aggfunc=np.max)


    # rename columns witch the value of each var
    codinstfin_cross_products.columns = [f"PRODUCTO_{c}_cod_instit_financiera_cat" for c in codinstfin_cross_products.columns]
    codinstfin_cross_products.reset_index(inplace=True)

    codinstfin_cross_products = codinstfin_cross_products.sort_values('codmes', ascending=True).groupby('key_value').last()
    codinstfin_cross_products.reset_index(inplace=True)
    codinstfin_cross_products.drop('codmes', axis=1,inplace=True)

    #features = pd.merge(features, codinstfin_cross_products, how='left', on='key_value')
    save_data(output_file + '_part5', codinstfin_cross_products)


    del codinstfin_cross_products
    gc.collect()

    # ==============================================================

    # =============== Productos by COD_CLASIFICACION_DEUDOR ===================
    codclassdeudor_cross_products = data.pivot_table(
                values='COD_CLASIFICACION_DEUDOR', 
                index=['key_value', 'codmes'], 
                columns='PRODUCTO', 
                aggfunc=np.max)


    # rename columns witch the value of each var
    codclassdeudor_cross_products.columns = [f"PRODUCTO_{c}_COD_CLASIFICACION_DEUDOR_" for c in codclassdeudor_cross_products.columns]
    codclassdeudor_cross_products.dropna(axis=1,how='all',inplace=True)    
    codclassdeudor_cross_products.reset_index(inplace=True)
    codclassdeudor_cross_products.drop('codmes', axis=1,inplace=True)

    codclassdeudor_cat_feateng = add_cat_feateng(codclassdeudor_cross_products)
    codclassdeudor_cat_feateng.reset_index(inplace=True)
    gc.collect()

    #features = pd.merge(features, codclassdeudor_cat_feateng, how='left', on='key_value')
    save_data(output_file + '_part6', codclassdeudor_cat_feateng)


    del codclassdeudor_cross_products, codclassdeudor_cat_feateng
    gc.collect()

    # ==============================================================

    return features
    pass

def main(input_file, output_file):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('INIT: making features data set from processed data')

    start = datetime.datetime.now()

    logger.info('RUN: loading data')
    df = load_data(input_file)
    
    logger.info(f'RUN: data size before be processed: {df.shape}')
    
    logger.info(f'RUN: building features')

    features = build_features(df,output_file)
    del df 

    logger.info(f'RUN: features size : {features.shape}')

    logger.info(f'RUN: saving features')
    #save_data(output_file, features)

    logger.info('END: making features data set has finished.')
    ends = datetime.datetime.now()
    diff = relativedelta(start, ends)
    print("The process has toke %d year %d month %d days %d hours %d minutes to complete the task" % (diff.years, diff.months, diff.days, diff.hours, diff.minutes))

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
