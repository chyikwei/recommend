"""
Base class of recommendation System
"""

from abc import ABCMeta, abstractmethod


class NotImplementedError(Exception):

    def __init__(self):
        super(NotImplementedError, self).__init__()


class DimensionError(Exception):

    def __init__(self):
        super(DimensionError, self).__init__()


class Base(object):

    """base class of recommendations"""

    __metaclass__ = ABCMeta

    def __init__(self):
        self.train_errors = []
        self.validation_erros = []

    @abstractmethod
    def estimate(self, iter=1000):
        """training models"""
        raise NotImplementedError

    @abstractmethod
    def suggestions(self, user_id, num=10):
        """suggest items for given user"""
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path):
        raise NotImplementedError

    @classmethod
    def load_model(cls, path):
        """load saved models"""
        raise NotImplementedError

    def __reper__(self):
        return self.__class__.__name__
