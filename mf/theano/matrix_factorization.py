"""
Matrix Factorization with Theano
"""

import numpy as np
from ..base import Base
from util.load_data import load_ml_1m
from util.evaluation_metrics import RMSE

import time
import theano
from theano import shared
import theano.tensor as T


class MatrixFactorization(Base):

    def __init__(self, num_user, num_item, num_feature, train, validation, **params):
        super(MatrixFactorization, self).__init__()
        self._num_user = num_user
        self._num_item = num_item
        self._num_feature = num_feature
        self.train_size = train.shape[0]

        # mean rating
        mean_rating = np.mean(train[:, 2])
        self._mean_rating = mean_rating

        self.user_train = T.cast(shared(
            np.asarray(train[:, 0], dtype=theano.config.floatX), borrow=True), 'int32')
        self.item_train = T.cast(shared(
            np.asarray(train[:, 1], dtype=theano.config.floatX), borrow=True), 'int32')

        self.rating_train = shared(
            np.asarray(train[:, 2], dtype=theano.config.floatX), borrow=True)

        self.user_validation = T.cast(shared(
            np.asarray(validation[:, 0], dtype=theano.config.floatX), borrow=True), 'int32')

        self.item_validation = T.cast(shared(
            np.asarray(validation[:, 1], dtype=theano.config.floatX), borrow=True), 'int32')

        self.rating_validation = shared(
            np.asarray(validation[:, 2], dtype=theano.config.floatX), borrow=True)

        # batch size
        self.batch_size = int(params.get('batch_size', 100000))

        # learning rate
        self.epsilon = float(params.get('epsilon', 100.0))
        # regularization parameter (lambda)
        self.lam = float(params.get('lam', 0.001))

        self.max_rating = params.get('max_rating')
        self.min_rating = params.get('min_rating')
        if self.max_rating:
            self.max_rating = float(self.max_rating)
        if self.min_rating:
            self.min_rating = float(self.min_rating)

        # latent variables
        # random generate features
        user_features = 0.3 * np.random.rand(num_user, num_feature)
        item_features = 0.3 * np.random.rand(num_item, num_feature)

        self._user_features = shared(
            value=np.asarray(user_features, dtype=theano.config.floatX), name='user_features', borrow=True)
        self._item_features = shared(
            value=np.asarray(item_features, dtype=theano.config.floatX), name='item_features', borrow=True)

    @property
    def user(self):
        return self._num_user

    @property
    def items(self):
        return self._num_item

    def cost(self, user_index, item_index, ratings):
        norm_ratings = ratings - self._mean_rating
        user_features = self._user_features[user_index]
        item_features = self._item_features[item_index]
        preds = T.sum(user_features * item_features, axis=1)
        regularization = T.sum(
            (item_features ** 2 + user_features ** 2))
        return (T.sum((norm_ratings - preds) ** 2) + self.lam * regularization) / self.batch_size

    def estimate(self, iterations=50, converge=1e-4):
        #last_rmse = None
        batch_num = int(
            np.ceil(float(self.train_size / self.batch_size)))
        print "batch_num =", batch_num + 1

        batch = T.lscalar()
        user_index = T.ivector()
        item_index = T.ivector()
        ratings = T.dvector()
        cost = self.cost(user_index, item_index, ratings)
        user_grad = T.grad(cost=cost, wrt=self._user_features)
        item_grad = T.grad(cost=cost, wrt=self._item_features)

        updates = [(
            self._user_features, self._user_features - self.epsilon * user_grad),
            (self._item_features, self._item_features - self.epsilon * item_grad)]

        # training
        train_func = theano.function(inputs=[batch],
                                     outputs=[],
                                     givens={
                                         user_index: self.user_train[batch * self.batch_size: (batch + 1) * self.batch_size],
                                         item_index: self.item_train[batch * self.batch_size: (batch + 1) * self.batch_size],
                                         ratings: self.rating_train[batch * self.batch_size: (
                                                                    batch + 1) * self.batch_size]},
                                     updates=updates)

        for iteration in xrange(iterations):
            for batch_index in xrange(batch_num):
                train_func(batch_index)
                
            # compute RMSE
            # train errors
            train_preds = self.predict(self.user_train, self.item_train)
            train_rmse = RMSE(train_preds, self.rating_train.get_value(borrow=True))

            # validation errors
            validation_preds = self.predict(
                self.user_validation, self.item_validation)
            validation_rmse = RMSE(
                validation_preds, self.rating_validation.get_value(borrow=True))
            self.train_errors.append(train_rmse)
            # self.validation_erros.append(validation_rmse)
            print "iterations: %3d, train RMSE: %.6f, validation RMSE: %.6f" % \
                (iteration + 1, train_rmse, validation_rmse)

            # stop if converge
            #if last_rmse:
                #if abs(train_rmse - last_rmse) < converge:
                #    break

            #last_rmse = train_rmse

    def predict(self, user_idex, item_index):
        u_features = self._user_features[user_idex]
        i_features = self._item_features[item_index]
        preds = T.sum(u_features * i_features, axis=1) + self._mean_rating
        preds = preds.eval()

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating
        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def suggestions(self, user_id, num=10):
        # TODO
        pass

    def save_model(self):
        # TODO
        pass

    def load_model(self):
        # TODO
        pass


def test():
    # TODO: move all test function into a separate file and measure performance
    num_user, num_item, ratings = load_ml_1m()
    # suffle_data
    np.random.shuffle(ratings)

    # split data to training & validation
    train_pct = 0.9
    train_size = int(train_pct * len(ratings))
    train = ratings[:train_size]
    validation = ratings[train_size:]

    # params
    num_feature = 10
    start_time = time.clock()
    mf_model = MatrixFactorization(
        num_user, num_item, num_feature, train, validation, max_rating=5, min_rating=1)
    mf_model.estimate(10)
    end_time = time.clock()
    print "time spend = %.3f" % (end_time - start_time)
   
    return mf_model
