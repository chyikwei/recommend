from __future__ import print_function

import os
import logging
import zipfile

from six.moves import urllib
from numpy.random import RandomState
from recommend.pmf import PMF
from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_1M_FOLDER = "ml-1m"
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

# models settings
n_feature = 10
eval_iters = 20
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
pmf = PMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
          epsilon=25., max_rating=5., min_rating=1., seed=0)

pmf.fit(train, n_iters=eval_iters)
train_preds = pmf.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = pmf.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])
print("after %d iterations, train RMSE: %.6f, validation RMSE: %.6f" % \
      (eval_iters, train_rmse, val_rmse))

