# Packages
import datetime as dt

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import warnings

warnings.filterwarnings('ignore')
from pathlib2 import Path

from sklearn.preprocessing import StandardScaler

# Model libraries

from sklearn.linear_model import Ridge

from functions import load_parquet_file, convert_to_type, ColumnMapperArg, get_consolidated_predictions

PATH = 'C:\\cygwin64\\home\\gryv9001\\code\\tv_dmp_viewers_breakdown\\data\\'
DATA_PATH = 'data'
BUCKET_NAME = 'useast1-nlsn-w-digital-dsci-dev'
FOLDER_NAME = 'users/gryv9001/'
DATE_FORMAT = '%Y%m%d'
NOW = dt.datetime.now()
NOW_STR = NOW.strftime(DATE_FORMAT)
TRAINING_START_DATE = dt.datetime(2019, 1, 7)
TRAINING_START_DATE_STR = TRAINING_START_DATE.strftime(DATE_FORMAT)
TRAINING_END_DATE = dt.datetime(2019, 1, 13)
TRAINING_END_DATE_STR = TRAINING_END_DATE.strftime(DATE_FORMAT)
TEST1_START_DATE = dt.datetime(2019, 1, 14)
TEST1_START_DATE_STR = TEST1_START_DATE.strftime(DATE_FORMAT)
TEST1_END_DATE = dt.datetime(2019, 1, 20)
TEST1_END_DATE_STR = TEST1_END_DATE.strftime(DATE_FORMAT)
TEST2_START_DATE = dt.datetime(2019, 1, 21)
TEST2_START_DATE_STR = TEST2_START_DATE.strftime(DATE_FORMAT)
TEST2_END_DATE = dt.datetime(2019, 1, 27)
TEST2_END_DATE_STR = TEST2_END_DATE.strftime(DATE_FORMAT)
DATA_GENERATION_DATE = dt.datetime(2019, 3, 26)
DATA_GENERATION_DATE_STR = DATA_GENERATION_DATE.strftime(DATE_FORMAT)
ONEHOT_TOP200_VIEW_FILENAME = 'onehot_top200.gzip'
MODEL_FILENAME = 'bridge.joblib'

NETWORK_START_POSITION = 0
NETWORK_END_POSITION = -27
DAY_OF_WEEK_START_POSITION = -26
DAY_OF_WEEK_END_POSITION = -19
AGE_GEN_START_POSITION = -18
AGE_GEN_END_POSITION = -2
TOP200_PREDICTIONS_FILENAME = 'top200_predictions.gzip'


def get_file_name(root, start, end, file_name):
    date_str = root + '_' + start + '_' + end
    file_name = '{0}'.format(date_str) + '_' + file_name
    p = Path('.')
    return p / DATA_PATH / file_name


def main():
    file_name = get_file_name(DATA_GENERATION_DATE_STR, \
                              TRAINING_START_DATE_STR, TRAINING_END_DATE_STR, ONEHOT_TOP200_VIEW_FILENAME)
    print(file_name)
    train_df = load_parquet_file(file_name)
    train_df['n_smart_tvs_tuned'] = train_df['n_smart_tvs_tuned'].fillna(0)
    train_df['n_non_smart_tvs_tuned'] = train_df['n_non_smart_tvs_tuned'].fillna(0)
    train_df['n_smart_tvs_tuned'] = train_df['n_smart_tvs_tuned'].astype(np.int64)
    train_df['n_non_smart_tvs_tuned'] = train_df['n_non_smart_tvs_tuned'].astype(np.int64)

    X = train_df.iloc[:, :-1]
    X.drop(['date'], axis=1, inplace=True)
    print('X has {} rows'.format(len(X)))
    y = train_df.iloc[:, -1]
    print('y has {} rows'.format(len(y)))

    scaler = StandardScaler()
    scaler.fit(convert_to_type(X, np.int64))
    X_scaled = scaler.transform(X)
    print('X_scaled={}'.format(X_scaled.shape))

    file_name = get_file_name(DATA_GENERATION_DATE_STR, \
                              TEST2_START_DATE_STR, TEST2_END_DATE_STR, ONEHOT_TOP200_VIEW_FILENAME)
    print(file_name)
    test_df = load_parquet_file(file_name)
    test_df['n_smart_tvs_tuned'] = test_df['n_smart_tvs_tuned'].fillna(0)
    test_df['n_non_smart_tvs_tuned'] = test_df['n_non_smart_tvs_tuned'].fillna(0)
    test_df['n_smart_tvs_tuned'] = test_df['n_smart_tvs_tuned'].astype(np.int64)
    test_df['n_non_smart_tvs_tuned'] = test_df['n_non_smart_tvs_tuned'].astype(np.int64)

    X_test = test_df.iloc[:, :-1]
    X_test.drop(['date'], axis=1, inplace=True)
    print('X_test has {} rows'.format(len(X_test)))
    y_test = test_df.iloc[:, -1]
    print('y_test has {} rows'.format(len(y_test)))
    X_test_scaled = scaler.transform(X_test)

    model = Ridge()
    model.fit(X_scaled, y)
    predictions_df = pd.DataFrame(model.predict(X_test_scaled))
    print('{} predictions'.format(len(predictions_df)))

    columns = ['network', 'date', 'day_of_week', 'quarter_hour', 'age_gen', 'n_smart_tvs_tuned', 'truth', 'pred']
    column_mapper_arg_list = []
    column_mapper_arg_list.append(ColumnMapperArg(column_name='network', start_col_index=NETWORK_START_POSITION, \
                                                  end_col_index=NETWORK_END_POSITION))
    column_mapper_arg_list.append(ColumnMapperArg('date', 'date'))
    column_mapper_arg_list.append(ColumnMapperArg(column_name='day_of_week', start_col_index=DAY_OF_WEEK_START_POSITION, \
                                                  end_col_index=DAY_OF_WEEK_END_POSITION))
    column_mapper_arg_list.append(ColumnMapperArg('quarter_hour', 'quarter_hour'))
    column_mapper_arg_list.append(ColumnMapperArg(column_name='age_gen', start_col_index=AGE_GEN_START_POSITION, \
                                                  end_col_index=AGE_GEN_END_POSITION))
    column_mapper_arg_list.append(ColumnMapperArg('n_smart_tvs_tuned', 'n_smart_tvs_tuned'))
    column_mapper_arg_list.append(ColumnMapperArg('truth', values=y_test))
    column_mapper_arg_list.append(ColumnMapperArg('pred', values=predictions_df))
    consolidated_prediction_df = get_consolidated_predictions(test_df, columns, column_mapper_arg_list)
    print('Consolidated {} predictions'.format(len(consolidated_prediction_df)))

    file_name = get_file_name(NOW_STR, \
                              TEST1_START_DATE_STR, TEST1_END_DATE_STR, TOP200_PREDICTIONS_FILENAME)
    print('Saving predictions to {}'.format(file_name))
    consolidated_prediction_df.to_parquet(file_name, compression='gzip')


if __name__ == '__main__':
    main()
