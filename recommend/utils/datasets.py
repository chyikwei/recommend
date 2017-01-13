"""
load data set

"""
import numpy as np
from numpy.random import RandomState
import scipy.sparse as sparse


ML_100K_DOWNLOAD_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'


def make_ratings(n_users, n_items, min_rating_per_user, max_rating_per_user,
                 rating_choices, seed=None, shuffle=True):
    """Randomly generate a (user_id, item_id, rating) array

    Return
    ------
        ndarray with shape (n_samples, 3)

    """
    if not (isinstance(rating_choices, list) or isinstance(rating_choices, tuple)):
        raise ValueError("'rating_choices' must be a list or tuple")
    if min_rating_per_user < 0 or min_rating_per_user >= n_items:
        raise ValueError("invalid 'min_rating_per_user' invalid")
    if min_rating_per_user > max_rating_per_user or max_rating_per_user >= n_items:
        raise ValueError("invalid 'max_rating_per_user' invalid")

    rs = RandomState(seed=seed)
    user_arrs = []
    for user_id in xrange(n_users):
        item_count = rs.randint(min_rating_per_user, max_rating_per_user) 
        item_ids = rs.choice(n_items, item_count, replace=False)
        ratings = rs.choice(rating_choices, item_count)
        arr = np.stack([np.repeat(user_id, item_count), item_ids, ratings], axis=1)
        user_arrs.append(arr)

    ratings = np.array(np.vstack(user_arrs))
    ratings[:, 2] = ratings[:, 2].astype('float')
    if shuffle:
        rs.shuffle(ratings)
    return ratings


def load_movielens_ratings(ratings_file):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
            line = line.split('::')[:3]
            line = [int(l) for l in line]
            ratings.append(line)
        ratings = np.array(ratings)
    return ratings


def build_ml_1m():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """
    num_user = 6040
    num_item = 3952
    print("\nloadind movie lens 1M data")
    with open("data/ratings.dat", "rb") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line.split('::')[:3]
            line = [int(l) for l in line]
            ratings.append(line)

            if line_num % 100000 == 0:
                print line_num

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)] - 1
    print "max user id", max(ratings[:, 0])
    print "max item id", max(ratings[:, 1])
    return num_user, num_item, ratings


def load_ml_1m():
    """load Movie Lens 1M ratings from saved gzip file"""
    import gzip
    import cPickle

    file_path = 'data/ratings.gz'
    with gzip.open(file_path, 'rb') as f:
        print "load ratings from: %s" % file_path
        num_user = cPickle.load(f)
        num_item = cPickle.load(f)
        ratings = cPickle.load(f)

        return num_user, num_item, ratings


def build_user_item_matrix(n_users, n_items, ratings):
    """Build user-item matrix

    Return
    ------
        sparse matrix with shape (n_users, n_items)
    """
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_users, n_items)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
