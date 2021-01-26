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

def add_cat_feateng(df):

    c = df.columns[0]
    df.reset_index(inplace=True)
    groupBy='key_value'  # TODO: harcodeado, ver como pasar como parametro

    for p in [1, 3]:
        df[f"{c}_diff_p{p}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.diff(p)
        )
    
    # count
    df[f"{c}_count"] = df.groupby(groupBy)[c].transform(
            lambda x: x.count()
        )


    for window in [3, 6, 9]:
        df[f"{c}_rolling_min_t{window}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.rolling(window).min()
        )

    for window in [3, 6, 9]:
        df[f"{c}_rolling_max_t{window}"] = df.groupby(groupBy)[c].transform(
            lambda x: x.rolling(window).max()
        )

 
    

    # TODO: harcodeado, ver como pasar como parametro
    df.set_index(['key_value', 'codmes'], inplace=True)

    # para saber si es una variable numerica o categorica
    df.columns =  [c + '_cat' for c in df.columns]

    return df

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
            lambda x: x.pct_change(periods=p)
        )
    
    # Moving Acum average
    # for window in [3, 6, 9]:
    #     df[f"{c}_rolling_acum_mean_t{window}"] = df.groupby(groupBy)[c].transform(
    #         lambda x: x.expanding(min_periods=window).mean()
    #     )

    # for window in [3, 6, 9]:
    #     df[f"{c}_rolling_min_t{window}"] = df.groupby(groupBy)[c].transform(
    #         lambda x: x.rolling(window).min()
    #     )

    # for window in [3, 6, 9]:
    #     df[f"{c}_rolling_max_t{window}"] = df.groupby(groupBy)[c].transform(
    #         lambda x: x.rolling(window).max()
    #     )

    # for window in [3, 6, 9]:
    #     df[f"{c}_rolling_skew_t{window}"] = df.groupby(groupBy)[c].transform(
    #         lambda x: x.rolling(window).skew()
    #     )

    # for window in [3, 6, 9]:
    #     df[f"{c}_rolling_kurt_t{window}"] = df.groupby(groupBy)[c].transform(
    #         lambda x: x.rolling(window).kurt()
    #     )

    

    # TODO: harcodeado, ver como pasar como parametro
    df.set_index(['key_value', 'codmes'], inplace=True) 

    # para saber si es una variable numerica o categorica
    df.columns =  [c + '_num' for c in df.columns]
    return df

# LOAD DATASET
def load_data(input_file):
    data  = pd.read_csv(f"{config.INPUT_PATH}{input_file}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)

    return data

# SAVE DATASET
def save_data(output_file, data):
    data.to_pickle(f"{config.OUTPUT_PATH}{output_file}.pkl")
    
    return 0

# PROCESS DATA
def build_features(data):
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
    # saldo by producto
    # saldo_xtab_df = data.pivot_table(
    #                         values='saldo', 
    #                         index=['key_value', 'codmes'], 
    #                         columns='PRODUCTO', 
    #                         aggfunc=np.sum)

    # saldo_xtab_df.columns = [f"PRODUCTO_{c}" for c in saldo_xtab_df.columns]
    # #saldo_xtab_df.sort_values(['key_value','codmes'], inplace=True, ascending=True)
    
    # =============== saldo by each var ===================
    # ['condicion','tipo_credito','PRODUCTO', 'RIESGO_DIRECTO', 'COD_CLASIFICACION_DEUDOR']
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

    

    features = pd.merge(features, condicion_num_feateng, how='left', on='key_value')

    del condicion_cross_products, condicion_num_feateng
    gc.collect()

    # ==============================================================

    # =============== Productos by RIESGO_DIRECTO ===================
    riesgodir_cross_products = data.pivot_table(
                values='RIESGO_DIRECTO', 
                index=['key_value', 'codmes'], 
                columns='PRODUCTO', 
                aggfunc=np.max)


    # rename columns witch the value of each var
    riesgodir_cross_products.columns = [f"PRODUCTO_{c}_RIESGO_DIRECTO_" for c in riesgodir_cross_products.columns]

    riesgodir_cross_products.dropna(axis=1,how='all',inplace=True)

    # we split the dataframe in n pieces (for each column) in order to process it in each CPU core
    df_split = np.array_split(riesgodir_cross_products, riesgodir_cross_products.shape[1], axis=1)
    gc.collect()


    riesgodir_cat_feateng = []
    for col in tqdm(range(0,len(df_split), NCORES )):
        pool = Pool(NCORES)
        result = pd.concat(pool.map(add_cat_feateng, df_split[col:NCORES+col]), axis=1)
        riesgodir_cat_feateng.append(result)
        del result
        gc.collect()
        pool.close()
        pool.join()


    riesgodir_cat_feateng = pd.concat(riesgodir_cat_feateng, axis=1)
    riesgodir_cat_feateng.reset_index(inplace=True)


    riesgodir_cat_feateng = riesgodir_cat_feateng.sort_values('codmes', ascending=True).groupby('key_value').last()
    riesgodir_cat_feateng.reset_index(inplace=True)
    gc.collect(2)

    features = pd.merge(features, riesgodir_cat_feateng, how='left', on='key_value')

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
    tipodcred_cross_products.reset_index(inplace=True)

    tipodcred_cross_products = tipodcred_cross_products.sort_values('codmes', ascending=True).groupby('key_value').last()

    features = pd.merge(features, tipodcred_cross_products, how='left', on=['key_value','codmes'])

    del tipodcred_cross_products
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

    features = pd.merge(features, codinstfin_cross_products, how='left', on=['key_value','codmes'])

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
    codclassdeudor_cross_products.reset_index(inplace=True)

    codclassdeudor_cross_products = codclassdeudor_cross_products.sort_values('codmes', ascending=True).groupby('key_value').last()

    features = pd.merge(features, codclassdeudor_cross_products, how='left', on=['key_value','codmes'])

    del codclassdeudor_cross_products
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

    features = build_features(df)
    del df 

    logger.info(f'RUN: features size : {features.shape}')

    logger.info(f'RUN: saving features')
    save_data(output_file, features)

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