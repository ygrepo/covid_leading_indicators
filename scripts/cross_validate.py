import pickle

import pandas as pd
from pathlib2 import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

import datetime as dt
from functions import load_parquet_file

DATA_PATH = 'data'
# PATH = 'C:\\cygwin64\\home\\gryv9001\\code\\tv_dmp_viewers_breakdown\\data\\'
DATE_FORMAT = '%Y%m%d'
AGG_VIEW_FILENAME = 'onehot_qr_hr_smarttv_df.gzip'
CV_RESULTS_FILENAME = 'cvresults.p'
START_DATE = dt.datetime(2019, 1, 7)
START_DATE_STR = START_DATE.strftime(DATE_FORMAT)
END_DATE = dt.datetime(2019, 1, 21)
END_DATE_STR = END_DATE.strftime(DATE_FORMAT)
DATE_SAMPLING = '20190319'
NOW = dt.datetime.now()
NOW_STR = NOW.strftime(DATE_FORMAT)
N_ESTIMATORS = 464

def get_data_filename():
    date_str = DATE_SAMPLING + '_' + START_DATE_STR + '_' + END_DATE_STR
    file_name = '{0}'.format(date_str) + '_' + AGG_VIEW_FILENAME
    p = Path('.')
    return p / DATA_PATH / file_name

def cv_result_filename():
    date_str = NOW_STR + '_' + START_DATE_STR + '_' + END_DATE_STR
    file_name = '{0}'.format(date_str) + '_' + CV_RESULTS_FILENAME
    p = Path('.')
    return p / DATA_PATH / file_name

def load_data():
    df = load_parquet_file(get_data_filename())
    print(df.head(3))
    df['n_smart_tvs_tuned'] = df['n_smart_tvs_tuned'].fillna(0)
    df['n_non_smart_tvs_tuned'] = df['n_non_smart_tvs_tuned'].fillna(0)