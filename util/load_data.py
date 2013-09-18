"""
load data set

"""
import numpy as np
import scipy.sparse as sparse


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


def build_rating_matrix(num_user, num_item, ratings):
    """
    build dense ratings matrix from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print '\nbuild matrix'
    # sparse matrix
    #matrix = sparse.lil_matrix((num_user, num_item))
    # dense matrix
    matrix = np.zeros((num_user, num_item), dtype='int8')
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            matrix[data[:, 0], item_id] = data[:, 2]

        if item_id % 1000 == 0:
            print item_id

    return matrix


def load_rating_matrix():
    """
    load Movie Lens 1M ratings from saved gzip file
    Format is numpy dense matrix
    """
    import gzip
    import cPickle

    file_path = 'data/rating_matrix.gz'
    with gzip.open(file_path, 'rb') as f:
        print "load ratings matrix from: %s" % file_path
        return cPickle.load(f)


def build_sparse_matrix(num_user, num_item, ratings):
    # TODO: have not tested it yet. will test after algorithm support sparse
    # matrix
    print '\nbuild sparse matrix'
    # sparse matrix
    matrix = sparse.lil_matrix((num_user, num_item))
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            # for sparse matrix
            matrix[data[:, 0], item_id] = np.array([data[:, 2]]).T

        if item_id % 1000 == 0:
            print item_id

    return matrix
