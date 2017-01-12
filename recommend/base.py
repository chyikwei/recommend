"""
Base class of recommendation System
"""

from abc import ABCMeta, abstractmethod


class ModelBase(object):

    """base class of recommendations"""

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train, n_iters):
        """training models"""
        pass

    @abstractmethod
    def predict(self, data):
        """save model"""
        pass

    def __reper__(self):
        return self.__class__.__name__
