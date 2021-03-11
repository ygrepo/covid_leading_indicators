import datetime as dt
import pickle

from pathlib2 import Path

DATA_PATH = 'data'
AGG_VIEW_FILENAME = 'onehot_qr_hr_smarttv_df.gzip'
CV_RESULTS_FILENAME = 'cvresults.p'

DATE_FORMAT = '%Y%m%d'


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


def load_cv_results():
    print("Loading cross-validation results")
    path = str(cv_result_filename())
    results = pickle.load(open(path, "rb"))
    return results


def main():
    grid_results = load_cv_results()
    print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))

    for test_mean, train_mean, param in zip(
            grid_results.cv_results_['mean_test_score'],
            grid_results.cv_results_['mean_train_score'],
            grid_results.cv_results_['params']):
        print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))


if __name__ == "__main__":
    main()
