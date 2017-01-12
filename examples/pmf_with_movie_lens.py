from __future__ import print_function

import logging
import numpy as np
from recommend.utils.load_data import load_movielens_ratings
from recommend.pmf import PMF
from recommend.utils.evaluation import RMSE

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

RATINGS_FILE = '/Users/chyikwei/github/recommend/data/ratings.dat'

# load MovieLens data
ratings = load_movielens_ratings(RATINGS_FILE)
n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])

# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1
n_feature = 10
eval_iters = 5

# split data to training & testing
train_pct = 0.9
np.random.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]

# models
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
pmf = PMF(n_user=n_user, n_item=n_item, n_feature=n_feature, max_rating=5, min_rating=1)

for i in xrange(1, 5):
    pmf.fit(train, n_iters=eval_iters)
    train_preds = pmf.predict(train[:, :2])
    train_rmse = RMSE(train_preds, np.float16(train[:, 2]))
    val_preds = pmf.predict(validation[:, :2])
    val_rmse = RMSE(val_preds, np.float16(validation[:, 2]))
    print("after %d iteration, train RMSE: %.6f, validation RMSE: %.6f" % (
        i * eval_iters, train_rmse, val_rmse))

