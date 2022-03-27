cimport cython
from numpy cimport ndarray
cimport numpy as np
import numpy as np
from scipy import spatial

from libc.stdlib cimport abs  # int abs

cdef int DELMAX = 1000

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False)
def bw_phi4(n, d, cnt, h):
    
    cdef double sum = 0.0, u
    cdef int nbin = cnt.shape[0]

    cdef np.ndarray[double, ndim=1] delta = (np.arange(nbin)*d/h)**2
    
    if (delta > DELMAX).any():
        idx = np.where(delta > DELMAX)[0]
        delta = np.delete(delta, idx)
        cnt = np.delete(cnt, idx)

    cdef np.ndarray[double, ndim=1] term = np.exp(-0.5*delta) * (delta**2 - 6*delta + 3)
    sum = np.sum(term * cnt)

    sum = 2 * sum + n * 3	 # add in diagonal
    u = sum / (n * (n - 1) * h**5 * np.sqrt(2*np.pi))
    return u

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False)
def bw_phi6(n, d, cnt, h):
    
    cdef double sum = 0.0, u
    cdef int nbin = cnt.shape[0]

    cdef np.ndarray[double, ndim=1] delta = (np.arange(nbin)*d/h)**2
    if (delta > DELMAX).any():
        idx = np.where(delta > DELMAX)[0]
        delta = np.delete(delta, idx)
        cnt = np.delete(cnt, idx)

    cdef np.ndarray[double, ndim=1] term = np.exp(-0.5*delta)*(delta**3 - 15*delta**2 + 45*delta - 15)
    sum = np.sum(term * cnt)

    sum = 2*sum - 15*n  # add in diagonal
    u = sum/(n*(n-1)* h**7 * np.sqrt(2*np.pi))

    return u

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False)
def bw_den(nbin, x):

    cdef int n = x.shape[0]
    cdef double xmin, xmax, rang, dd

    xmin = np.inf
    xmax = -np.inf

    if(np.isfinite(x).all() == False):
        print(f"non-finite x in bandwidth calculation")

    xmin = np.min(x)
    xmax = np.max(x)

    rang = (xmax-xmin)*1.01
    dd = rang/nbin

    # PAIRWISE BINNED DISTANCE
    cdef long[:] arr = (x/dd).astype(int)
    cdef long[:] cnt = np.zeros(nbin, dtype=int)
    cdef int dists

    cdef int i, j
    for i in range(n):
        dists = arr[i]
        for j in range(i):
            cnt[abs(dists - arr[j])] += 1

    return dd, cnt
