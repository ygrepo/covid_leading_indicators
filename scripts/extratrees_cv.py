import datetime as dt
import pickle

import pandas as pd
from pathlib2 import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

DATA_PATH = 'data'
# PATH = 'C:\\cygwin64\\home\\gryv9001\\code\\tv_dmp_viewers_breakdown\\data\\'
AGG_VIEW_FILENAME = 'onehot_qr_hr_smarttv_df.gzip'
CV_RESULTS_FILENAME = 'cvresults.p'

DATE_FORMAT = '%Y%m%d'
N_ESTIMATORS = 464

def get_data_filename():
    start_date = dt.datetime(2018, 2, 1)
    start_date_str = start_date.strftime(DATE_FORMAT)
    end_date = dt.datetime(2018, 2, 3)
    end_date_str = end_date.strftime(DATE_FORMAT)
    date_str = '20190311_' + start_date_str + '_' + end_date_str
    file_name = '{0}'.format(date_str) + '_' + AGG_VIEW_FILENAME
    p = Path('.')
    return p / DATA_PATH / file_name

def cv_result_filename():
    start_date = dt.datetime(2018, 2, 1)
    start_date_str = start_date.strftime(DATE_FORMAT)
    end_date = dt.datetime(2018, 2, 3)
    end_date_str = end_date.strftime(DATE_FORMAT)
    current_dt = dt.datetime.now()
    current_dt_str = current_dt.strftime(DATE_FORMAT)
    date_str = current_dt_str + '_' + start_date_str + '_' + end_date_str
    file_name = '{0}'.format(date_str) + '_' + CV_RESULTS_FILENAME
    p = Path('.')
    return p / DATA_PATH / file_name


def save_cv_results(results):
    print("Saving cross-validation results")
    path = str(cv_result_filename())
    print(path)
    pickle.dump(results, open(path, "wb"))


def load_cv_results():
    print("Loading cross-validation results")
    path = str(cv_result_filename())
    results = pickle.load(open(path, "rb"))
    return results


def load_file(path):
    df = pd.read_parquet(str(path))
    print('Loaded {} rows'.format(len(df)))
    for c in df.columns:
         df[c] = df[c].astype(float)
    return df



def generate_dataset(df):
    list_cols = df.columns.values.tolist()
    x_list = list(list_cols[0])
    x_list.extend(list_cols[2:])
    X = df[x_list]
    y = df['hh_smrt_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    print('X_train={}, X_test={}, y_train={}, y_test={}'.format(X_train.shape, X_test.shape, y_train.shape,
                                                                y_test.shape))
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test


def main():
    # param_grid = {
    #     'min_samples_leaf': [1]
    # }
    param_grid = {
        #'n_estimators':np.logspace(2, 3, 10, endpoint=True).astype(np.int),
        'min_samples_leaf':[1,5,10,20]
    }

    model = ExtraTreesRegressor(n_estimators=N_ESTIMATORS)
    gsc = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',
        refit='mean_squared_error',
        return_train_score=True,
        n_jobs=10,
        cv=5,
        verbose=True
    )
    df = load_file(get_data_filename())
    print(df.head(3))
    X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test = generate_dataset(df)
    grid_results = gsc.fit(X_train_scaled, y_train)
    print("Completed cross-validation, saving results")
    save_cv_results(grid_results)

if __name__ == "__main__":
    #print(np.logspace(2, 3, 10, endpoint=True).astype(np.int))
    main()
