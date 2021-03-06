import io

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


class ColumnMapperArg:
    def __init__(self, column_name, mapped_column_name=None, values=pd.Series(), \
                 start_col_index=None, end_col_index=None):
        self.column_name = column_name
        self.mapped_column_name = mapped_column_name
        self.values = values
        self.start_col_index = start_col_index
        self.end_col_index = end_col_index

    def __repr__(self):
        return "ColumnMapperArg({},{},{},{})".format(self.column_name, self.mapped_column_name,
                                                     str(self.start_col_index), str(self.end_col_index))


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    df.drop(columns=cols, axis=1, inplace=True)
    return df


# Read single parquet file from S3
def pd_read_s3_parquet(key, bucket, s3_client=None, **args):
    if s3_client is None:
        s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()), **args)

# Read single csv file from S3
def pd_read_s3_csv(key, bucket, s3_client=None, **args):
    if not s3_client:
        s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()), **args)

# Read multiple parquets from a folder on S3 generated by spark
def pd_read_s3_multiple_parquets(filepath, bucket, s3=None,
                                 s3_client=None, verbose=False, **args):
    if not filepath.endswith('/'):
        filepath = filepath + '/'  # Add '/' to the end
    if s3_client is None:
        #session = boto3.session.Session(profile_name='dev')
        #s3 = session.resource('s3')
        s3_client = boto3.client('s3')
    if s3 is None:
        s3 = boto3.resource('s3')
    s3_keys = [item.key for item in s3.Bucket(bucket).objects.filter(Prefix=filepath)
               if item.key.endswith('.parquet')]
    if not s3_keys:
        print('No parquet found in', bucket, filepath)
    elif verbose:
        print('Load parquets:')
        for p in s3_keys:
            print(p)
    dfs = [pd_read_s3_parquet(key, bucket=bucket, s3_client=s3_client, **args)
           for key in s3_keys]
    return pd.concat(dfs, ignore_index=True)


def pd_read_s3(filepath, bucket, file_type= "parquet", s3=None,
                                 s3_client=None, verbose=False, **args):
    assert file_type == "parquet" or file_type == "csv"
    if not filepath.endswith('/'):
        filepath = filepath + '/'  # Add '/' to the end
    if s3_client is None:
        s3_client = boto3.client('s3')
    if s3 is None:
        s3 = boto3.resource('s3')
    s3_keys = [item.key for item in s3.Bucket(bucket).objects.filter(Prefix=filepath)
               if (item.key.endswith('.parquet') or item.key.endswith('.csv'))]
    if not s3_keys:
        print('No parquet or csv files found in', bucket, filepath)
    elif verbose:
        print('Load files:')
        for p in s3_keys:
            print(p)
    if file_type == "parquet":
        dfs = [pd_read_s3_parquet(key, bucket=bucket, s3_client=s3_client, **args) for key in s3_keys]
    else:
        dfs = [pd_read_s3_csv(key, bucket=bucket, s3_client=s3_client, **args) for key in s3_keys]
    return pd.concat(dfs, ignore_index=True)


def load_parquet_file(path):
    df = pd.read_parquet(str(path))
    print('Loaded {} rows from {}'.format(len(df), path))
    return df


def scores_report(model_name, model, X_test, y_test):
    columns = ['MSE', 'R2 Score', 'MAE', 'Expl.Var', ]
    rows = [model_name]
    results = pd.DataFrame(0.0, columns=columns, index=rows)
    y_pred = model.predict(X_test)
    results.iloc[0, 0] = np.sqrt(mean_squared_error(y_test, y_pred))
    results.iloc[0, 1] = r2_score(y_test, y_pred)
    results.iloc[0, 2] = mean_absolute_error(y_test, y_pred)
    results.iloc[0, 3] = explained_variance_score(y_test, y_pred)
    return results


def convert_to_type(df, type):
    for c in df.columns:
        df[c] = df[c].astype(type)
    return df.copy()


def get_inverse_series(row, cols, keyword):
    keyword_length = len(keyword)
    for c in cols:
        c = c.strip()
        if row[c] == 1 and c.startswith(keyword):
            return c[keyword_length + 1:]
    raise Exception('Invalid row:{}'.format(row))


def age_gender_bucket16(gender, age, gender_age=12):
    buckets = [6, 12, 18, 25, 35, 50, 55, 65]
    bk = np.digitize(age, buckets)
    if ((gender == 'F') | (gender == 'Female')) & (age >= 12):
        bk += 7
    return int(bk)


def get_consolidated_predictions(df, cols, column_mapper_arg_list):
    results = pd.DataFrame(None, columns=cols)
    df_columns = list(df.columns.values)
    for arg in column_mapper_arg_list:
        if arg.mapped_column_name:
            results[arg.column_name] = df[arg.mapped_column_name]
        if not arg.mapped_column_name and arg.values.empty:
            results[arg.column_name] = \
                df.apply(lambda r:
                         get_inverse_series(r, df_columns[arg.start_col_index: arg.end_col_index], arg.column_name),
                         axis=1)
        elif not arg.values.empty:
            results[arg.column_name] = arg.values
    return results


def compute_ratio_per_quarter(df):
    data = []
    for i in range(1, 5):
        seen_ad_df = df[
            (df.view_time_start <= df.ad_start_time_adj)
            & (df.commercial_start_time <= df.view_time_end)
            & (df.advertisement_quarter == i)
            ]
        seen_ad_df = seen_ad_df.drop_duplicates()
        seen_ad_count = len(seen_ad_df)
        not_seen_ad_df = df[
            (df.view_time_start <= df.ad_start_time_adj)
            & (df.view_time_end > df.ad_start_time_adj)
            & (df.view_time_end < df.commercial_start_time)
            & (df.advertisement_quarter == i)]
        not_seen_ad_df = not_seen_ad_df.drop_duplicates()
        not_seen_ad_count = len(not_seen_ad_df)
        total_count = seen_ad_count + not_seen_ad_count
        ratio = 0
        if total_count != 0:
            ratio = (seen_ad_count / total_count) * 100.0
        data.append((str(i), "{:.2f}".format(ratio)))
    df = pd.DataFrame(data, columns=['Quarter', 'Ratio'])
    return df


if __name__ == "__main__":
    print(age_gender_bucket16('F', 14))
