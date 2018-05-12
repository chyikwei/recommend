from __future__ import print_function

import os
import logging
import zipfile
from six.moves import urllib
from numpy.random import RandomState
from recommend.bpmf import BPMF
from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings
import pandas as pd

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
# ML_1M_FOLDER = "ml-1m"
ML_1M_FOLDER = "/home/lorenzo/BPMF/my_code/data/1M-dataset"
ML_1M_ZIP_SIZE = 24594131

rand_state = RandomState(0)

# download MovieLens 1M dataset if necessary
def ml_1m_download(folder, file_size):
    file_name = "ratings.dat"
    file_path = os.path.join(os.getcwd(), folder, file_name)
    if not os.path.exists(file_path):
        print("file %s not exists. downloading..." % file_path)
        zip_name, _ = urllib.request.urlretrieve(ML_1M_URL, "ml-1m.zip")
        with zipfile.ZipFile(zip_name, 'r') as zf:
            file_path = zf.extract('ml-1m/ratings.dat')

    # check file
    statinfo = os.stat(file_path)
    if statinfo.st_size == file_size:
        print('verify success: %s' % file_path)
    else:
        raise Exception('verify failed: %s' % file_path)
    return file_path

# load or download MovieLens 1M dataset
rating_file = ml_1m_download(ML_1M_FOLDER, file_size=ML_1M_ZIP_SIZE)
ratings = load_movielens_1m_ratings(rating_file)
n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])

# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1

# split data to training & testing
train_pct = 0.9

rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]

# models settings; do now the loop over several n_features. 
results = pd.DataFrame(columns=['Number of features', 'Train RMSE', 'Test RMSE'])
n_features_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
eval_iters = 50
for n_feature in n_features_list: 
    print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
        n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
                max_rating=5., min_rating=1., seed=0)

    train_rmse_list, test_rmse_list = bpmf.fit(train, validation, n_iters=eval_iters)
    
    row = pd.DataFrame({'Number of features' : n_feature, 
                         'Train RMSE': train_rmse_list, 
                         'Test RMSE': test_rmse_list}) 
    results = results.append(row)
    results.to_csv("results/1M_movielens_features{}_iterations{}.csv".format(n_features_list, eval_iters))    
    
    # print("after %d iteration, train RMSE: %.6f, validation RMSE: %.6f" %
    #     (eval_iters, train_rmse, val_rmse))
