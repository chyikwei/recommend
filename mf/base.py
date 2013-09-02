"""
Base class of recommendation System
"""

class NotImplementedError(Exception):
    def __init__(self):
        super(NotImplementedError, self).__init__()


class Base(object):
    """base class of recommendations"""
    def __init__(self):
        self.train_errors = []
        self.validation_erros = []

    def Estimate(self, iter=1000):
        raise NotImplementedError

    def suggestions(self, user_id, num=10):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    @classmethod
    def load_model(cls, file):
        raise NotImplementedError          

    def __reper__(self):
        return self.__class__.__name__
