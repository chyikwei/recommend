"""
load data set

"""
import numpy as np
import scipy.sparse as sparse

def load_ml_1m():
    """
    load movie lens 1M rating
    Source: http://www.grouplens.org/
    """
    num_user = 6040
    num_item = 3952

    with open("data/ratings.dat", "rb") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line.split('::')[:3]
            line = [int(l) for l in line]
            ratings.append(line)

            if line_num % 10000 == 0:
                print line_num

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0 
    ratings[:,(0, 1)] = ratings[:, (0, 1)] - 1
    
    return num_user, num_item, ratings


def build_matrix(num_user, num_item, ratings):
    matrix = sparse.lil_matrix(num_user, num_item, dtype='float16')
