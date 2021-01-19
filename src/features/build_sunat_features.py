# build_sunat_features.py
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


# GET DICT DATE
def get_date_dict(date_field='fecalta'):
    enum = enumerate(list(np.sort(pd.read_csv(f"{config.INPUT_PATH}sunat_test.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)[date_field].unique())))
    date_dict =dict((j,i) for i,j in enum)

    return date_dict


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

    data.drop_duplicates(inplace=True)

    data.sort_values(['key_value', 'fecalta', 'fecbaja'], inplace=True, ascending=True)

    

    # ============================== FECALTA ========================

    # get dates dict
    fecalta_dict = get_date_dict('fecalta')

    data['fecalta'] = data['fecalta'].map(fecalta_dict)


    
    fecalta_x_var = []
    for val in ['tipcontribuyente', 'tippersona', 'ciiu', 'ubigeo', 'condiciondomicilio','estadocontribuyente', 'codvia', 'codzona','contabilidad', 'facturacion', 'domiciliado', 'comercioexterior', 'cargorele', 'codentidadtributo', 'estadotributo']:
        dfxtab = data.pivot_table(
                values=val, 
                index=['key_value'], 
                columns='fecalta', 
                aggfunc=np.max,
                fill_value=-9999)
        
        # rename columns witch the value of each var
        dfxtab.columns = [f"fecalta_{c}_{val}" for c in dfxtab.columns]
        fecalta_x_var.append(dfxtab)


    gc.collect()

    fecalta_xtab_df = pd.concat(fecalta_x_var, axis=1)

    fecalta_xtab_df.dropna(axis=1,how='all',inplace=True)

    # ============================== FECBAJA ========================

    # get dates dict
    fecbaja_dict = get_date_dict('fecbaja')

    data['fecbaja'] = data['fecbaja'].map(fecbaja_dict)


    
    fecbaja_x_var = []
    for val in ['tipcontribuyente', 'tippersona', 'ciiu', 'ubigeo', 'condiciondomicilio','estadocontribuyente', 'codvia', 'codzona','contabilidad', 'facturacion', 'domiciliado', 'comercioexterior', 'cargorele', 'codentidadtributo', 'estadotributo']:
        dfxtab = data.pivot_table(
                values=val, 
                index=['key_value'], 
                columns='fecbaja', 
                aggfunc=np.max,
                fill_value=-9999)
        
        # rename columns witch the value of each var
        dfxtab.columns = [f"fecbaja_{c}_{val}" for c in dfxtab.columns]
        fecbaja_x_var.append(dfxtab)


    gc.collect()

    fecbaja_xtab_df = pd.concat(fecbaja_x_var, axis=1)

    fecbaja_xtab_df.dropna(axis=1,how='all',inplace=True)



    # joing bouth features
    features = fecalta_xtab_df.join(fecbaja_xtab_df)



    # tipo x estado contribuyente 
    val = 'estadocontribuyente'
    col = 'tipcontribuyente'
    dfxtab = data.pivot_table(
        values=val, 
        index=['key_value'], 
        columns=col, 
        aggfunc=np.max,
        fill_value=-9999)

    # rename columns witch the value of each var
    dfxtab.columns = [f"{col}_{c}_{val}_max" for c in dfxtab.columns]

    # joing bouth features
    features = features.join(dfxtab)


    gc.collect()


    # tipo x estado contribuyente 
    val = 'estadocontribuyente'
    col = 'tipcontribuyente'
    dfxtab = data.pivot_table(
        values=val, 
        index=['key_value'], 
        columns=col, 
        aggfunc='count',
        fill_value=-9999)

    # rename columns witch the value of each var
    dfxtab.columns = [f"{col}_{c}_{val}_count" for c in dfxtab.columns]

    # joing bouth features
    features = features.join(dfxtab)


    gc.collect()



    # tributo
    val = 'estadotributo'
    col = 'codentidadtributo'
    dfxtab = data.pivot_table(
        values=val, 
        index=['key_value'], 
        columns=col, 
        aggfunc=np.max,
        fill_value=-9999)

    # rename columns witch the value of each var
    dfxtab.columns = [f"{col}_{c}_{val}_max" for c in dfxtab.columns]

    # joing bouth features
    features = features.join(dfxtab)


    gc.collect()

    #  tributo
    dfxtab = data.pivot_table(
        values=val, 
        index=['key_value'], 
        columns=col, 
        aggfunc='count',
        fill_value=-9999)

    # rename columns witch the value of each var
    dfxtab.columns = [f"{col}_{c}_{val}_count" for c in dfxtab.columns]

    # joing bouth features
    features = features.join(dfxtab)

    features.reset_index(inplace=True)


    gc.collect()

    del fecalta_xtab_df,fecbaja_xtab_df, dfxtab


    # cat variables count
    # col_dummies = []
    # for col in ['tipcontribuyente', 'tippersona', 'ciiu', 'ubigeo', 'condiciondomicilio', 'estadocontribuyente']:
    #     col_dummies.append(pd.get_dummies(data[col], prefix=col))
    
    # col_dummies = pd.concat(col_dummies, axis=1)

    # cat_features =data[['key_value']].join(pd.get_dummies(data['tipcontribuyente'], prefix='tipcontribuyente_'))
    # cat_features = cat_features.groupby('key_value').sum().reset_index()

    # features = pd.merge(codmes_df, condicion_df, how='left', on='key_value')

    # del codmes_df, condicion_df
    # gc.collect()


    return features
    pass


# LOAD DATASET
def load_data(input_file):
    data  = pd.read_csv(f"{config.INPUT_PATH}{input_file}.csv", sep=config.CSV_SEP).pipe(reduce_mem_usage)

    return data


# SAVE DATASET
def save_data(output_file, data):
    data.to_pickle(f"{config.OUTPUT_PATH}{output_file}.pkl")
    
    return 0


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