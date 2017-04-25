class DimensionError(ValueError):
    """Exception to raise if data dimension mismatched
    """


class NotFittedError(ValueError):
    """Exception to raise if a model is used before fitting
    """
