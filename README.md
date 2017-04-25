[![Build Status](https://circleci.com/gh/chyikwei/recommend.png?&style=shield)](https://circleci.com/gh/gh/chyikwei/recommend)
[![Coverage Status](https://coveralls.io/repos/github/chyikwei/recommend/badge.svg?branch=master)](https://coveralls.io/github/chyikwei/recommend?branch=master)
Recommend
=========

Simple recommendatnion system implementation with Python

Current model:
--------------
- Probabilistic Matrix Factorization
- Bayesian Matrix Factorization
- Alternating Least Squares with Weighted Lambda Regularization (ALS-WR)

Reference:
----------
- "Probabilistic Matrix Factorization", R. Salakhutdinov and A.Mnih., NIPS 2008
- "Bayesian Probabilistic Matrix Factorization using MCMC", R. Salakhutdinov and A.Mnih., ICML 2008
- Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
- "Large-scale Parallel Collaborative Filtering for the Netflix Prize", Y. Zhou, D. Wilkinson, R. Schreiber and R. Pan, 2008

Install:
--------
```
# clone repoisitory
git clone git@github.com:chyikwei/recommend.git
cd recommend

# install numpy & scipy
pip install -r requirements.txt
pip install .
```

Getting started:
----------------

To run BPMF with MovieLens 1M dataset:
First, download MovieLens 1M dataset and unzip it (data will be in `ml-1m` folder).
Then run:

```python
>>> import numpy as np
>>> from recommend.bpmf import BPMF
>>> from recommend.utils.evaluation import RMSE
>>> from recommend.utils.datasets import load_movielens_1m_ratings

# load user ratings
>>> ratings = load_movielens_1m_ratings('ml-1m/ratings.dat')
>>> n_user = max(ratings[:, 0])
>>> n_item = max(ratings[:, 1])
>>> ratings[:, (0, 1)] -= 1 # shift ids by 1 to let user_id & movie_id start from 0

# fit model
>>> bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=10,
                max_rating=5., min_rating=1., seed=0).fit(ratings, n_iters=20)
>>> RMSE(bpmf.predict(ratings[:, :2]), ratings[:,2]) # training RMSE
0.79784331768263683

# predict ratings for user 0 and item 0 to 9:
>>> bpmf.predict(np.array([[0, i] for i in xrange(10)]))
array([ 4.35574067,  3.60580936,  3.77778456,  3.4479072 ,  3.60901065,
        4.29750917,  3.66302187,  4.43915423,  3.85788772,  4.02423073])
```

Complete example can be found in `examples/` folder. The script will download MovieLens 1M dataset automatically, run PMF(BPMF) model and show training/validation RMSE.


Running Test:
-------------
```
python setup.py test
```

or run test with coverage:
```
coverage run --source=recommend setup.py test
coverage report -m
```

Uninstall:
----------
```
pip uninstall recommend
```

Notes:
------
- Old version code can be found in `v0.0.1`. It contains a Probabilistic Matrix Factorization model with theano implementation.
