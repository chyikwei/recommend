![Build Status](https://circleci.com/gh/chyikwei/recommend.png?&style=shield&circle-token=41f0b88bfbe0c34a269b522ffacf3da80d9a9b20)

Recommend
=========

Simple recommendatnion system implementation with Python

Current method:
- Probabilistic Matrix Factorization
- Bayesian MF Matrix Factorization

Reference:
- "Probabilistic Matrix Factorization", R. Salakhutdinov and A.Mnih., Neural Information Processing Systems 21 (NIPS 2008). Jan. 2008.
- "Bayesian Probabilistic Matrix Factorization using MCMC", R. Salakhutdinov and A.Mnih., 25th International Conference on Machine Learning (ICML-2008) 
- Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html

Install:
```
# clone repoisitory
git clone git@github.com:chyikwei/recommend.git
cd recommend

# install numpy & scipy
pip install -r requirement
python setup.py install
```

Test:
```
python setup.py test
```
TODO:
- Add examples
- download movie lens data automatically

Notes:
- Old version code can be found in `v0.0.1`. It contains a Probabilistic Matrix Factorization model with theano implementation.
