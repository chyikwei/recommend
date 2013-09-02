import numpy as np

def RMSE(estimation, truth):
    """Root Mean Square Error"""

    num_sample = len(estimation)

    # sum square error 
    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1.0))
