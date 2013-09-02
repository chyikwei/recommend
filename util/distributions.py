"""
Copy from https://gist.github.com/jfrelinger/2638485
"""

import numpy as np
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
 
def invwishartrand_prec(nu,phi):
    return inv(wishartrand(nu,phi))
 
def invwishartrand(nu, phi):
    return inv(wishartrand(nu, inv(phi)))
 
def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = npr.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))
    
    
if __name__ == '__main__':
    npr.seed(1)
    nu = 5
    a = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
    #print invwishartrand(nu,a)
    x = np.array([ invwishartrand(nu,a) for i in range(20000)])
    nux = np.array([invwishartrand_prec(nu,a) for i in range(20000)])
    print x.shape
    print np.mean(x,0),"\n", inv(np.mean(nux,0))
    #print inv(a)/(nu-a.shape[0]-1)
    