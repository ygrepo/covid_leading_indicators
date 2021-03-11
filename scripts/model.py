# Packages
import numpy as np
from scipy import stats
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from datetime import date, datetime
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train(df):
    list_cols = df.columns.values.tolist()
    x_list = [list_cols[0]]
    x_list.extend(list_cols[2:-1])
    X = df[x_list]
    y = df['device_id_count']
    print('X={}, y={}'.format(X.shape, y.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    print('X_train={}, X_test={}, y_train={}, y_test={}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    alphas = list(np.logspace(-5, 5, 5))
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(r2_score(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))

def main():
    # PATH='C:\\cygwin64\\home\\gryv9001\\code\\tv_dmp_viewers_breakdown\\data\\'
    PATH = '~/code/tv_dmp_viewers_breakdown/data/'
    FILENAME = 'hr_net_df.gzip'
    df = pd.read_parquet(PATH + FILENAME)
    train(df)


if __name__ == "__main__":
    main()
