import numpy as np
from ..exceptions import DimensionError


def check_ratings(ratings, n_user, n_item, max_rating=None, min_rating=None):
    """Check rating array

    'ratings' must be a matrix with shape (n_sample, 3)
    and each row is (user_id, item_id, rating).
    Both user_id and item_id start from 0.

    Return
    ------
        None

    """
    if ratings.shape[1] != 3:
        raise DimensionError(
            "Invalid rating format. number of column must be 3")

    if not np.all(ratings[:, :2] >= 0):
        raise ValueError("negative user_id or item_id")

    max_user_id = ratings[:, 0].max()
    if max_user_id >= n_user:
        raise ValueError("max user_id >= %d", n_user)

    max_item_id = ratings[:, 1].max()
    if max_item_id >= n_item:
        raise ValueError("max n_item >= %d", n_item)

    if max_rating is not None and np.any(ratings[:, 2] > max_rating):
        raise ValueError("max rating >= %d", max_rating)

    if min_rating is not None and np.any(ratings[:, 2] < min_rating):
        raise ValueError("min rating >= %d", min_rating)
