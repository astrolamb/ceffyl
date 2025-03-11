"""
A set of functions to calculate the bandwidth of a normal kernel to create
a kernel density estimator of some data x

Portions of bandwidths.py contain code derived from, or inspired by that
Stats package for R under the GNU General Public License (GPL) version 2 or
later.
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth

#  File src/library/stats/R/bandwidths.R
#  Part of the R package, https://www.R-project.org
#
#  Copyright (C) 1994-2001 W. N. Venables and B. D. Ripley
#  Copyright (C) 2001-2014 The R Core Team
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  https://www.R-project.org/Licenses/

"""

from functools import partial
import numpy as np
from jax import jit
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from scipy import optimize


@jit
def _SD(n: float, d: ArrayLike, cnt: ArrayLike, h: float) -> float:
    """
    Equation 12.1 from [1] for a Gaussian kernel, where r`$\phi^\mathrm{iv}$ is
    the fourth derivitive of the Gaussian.

    Parameters
    ----------
    n : float
        Number of samples
    d : ArrayLike
        Array of pairwise distances
    cnt : ArrayLike
        Array of counts of pairwise distances
    h : float
        Proposed bandwidth
    
    Returns
    -------
    float
        The value of r`$\hat{S}_D(h)` from [1]

    References
    ----------
    [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """

    delta = (np.arange(cnt.shape[0]) * d / h)**2

    term = (jnp.exp(-0.5*delta) *
            (delta**2 - 6*delta + 3))
    s = 2 * jnp.sum(term * cnt) + (n * 3)
    u = s / (n * (n-1) * h**5 * jnp.sqrt(2*jnp.pi))

    return u


@jit
def _TD(n: float, d: ArrayLike, cnt: ArrayLike, h: float) -> float:
    """
    Equation 12.2 from [1] for a Gaussian kernel, where r`$\phi^\mathrm{iv}$ is
    the fourth derivitive of the Gaussian.

    Parameters
    ----------
    n : float
        Number of samples
    d : ArrayLike
        Array of pairwise distances
    cnt : ArrayLike
        Array of counts of pairwise distances
    h : float
        Proposed bandwidth
    
    Returns
    -------
    float
        The value of r`$\hat{T}_D(h)` from [1]

    References
    ----------
    [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """

    delta = (jnp.arange(cnt.shape[0]) * d / h)**2

    term = (jnp.exp(-0.5*delta) *
            (delta**3 - 15*(delta**2) + 45*delta - 15))
    s = 2 * jnp.sum(term * cnt) - 15 * n
    u = s / (n * (n-1) * h**7 * jnp.sqrt(2*jnp.pi))

    return u

@partial(jit, static_argnums=[0])
def _pairwise_binned_distance(
        nbin: int,
        x: ArrayLike
        ) -> tuple[Array, Array]:
    """
    Compute the pairwise distance between samples. For memory efficiency, the
    distances are binned and counted.

    Parameters
    ----------
    nbin : int
        Number of bins
    x : ArrayLike
        Array of samples to compute KDE bandwidth
    
    Returns
    -------
    tuple[Array, Array]
        The first array is the binned distances, the second array is the number
        of distances within that bin.
    """

    n = x.shape[0]

    xmin = jnp.min(x)
    xmax = jnp.max(x)

    rang = (xmax-xmin) * 1.01
    dd = rang / nbin

    # pairwise binned distance
    arr = jnp.rint(x / dd).astype(int)
    cnt = jnp.zeros(nbin).astype(int)
    idx = jnp.arange(n-1)

    print('Computing pairwise distances...')
    for i in range(1, n):
        dists = jnp.abs(arr[i] - arr[idx[:i]])
        cnt = cnt.at[dists].add(1)

    return dd, cnt


def sj_dpi(x: ArrayLike, nbin: int=10000):
    """
    `sj_dpi` implements most of the methods of Sheather & Jones (1991) to select
    the bandwidth using pilot estimation of derivatives. This is the 'direct
    plug-in' method as defined by an R implementation. With sufficiently large
    number of samples, this should give the same bandwidth as `sj_ste.`

    This method uses pairwise binned distances: they are of complexity
    (O(n^2)) up to n = nb/2 and (O(n)) thereafter. Because of the binning,
    the results differ slightly when x is translated or sign-flipped.

    Parameters
    ----------
    x : ArrayLike
        array of values to calculate bandwidth from
    bin : int
        number of bins to find pairwise distances

    Returns
    -------
        float
            The bandwidth that solves Equation 12 of [1]

    Raises
    ------
    ValueError
        If x has non-finite elements, is too sparse, or has less than two
        elements, or if nbin is negative.
    
    References
    ----------
    [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """
    if jnp.isfinite(x).all() is False:
        raise ValueError("Input has non-finite elements")

    n = len(x)
    if n < 2:
        raise ValueError("Array must have two or more elements")

    if nbin < 0:
        raise ValueError("`nbin` must be a positive number")

    c1 = 1 / (2 * jnp.sqrt(jnp.pi) * n)
    iqr = jnp.subtract(*jnp.percentile(x, jnp.array([75, 25])))
    scale = jnp.min(jnp.array([jnp.std(x), iqr/1.349]))
    b = 1.230 * scale * n**(-1/9)

    d, cnt = _pairwise_binned_distance(nbin, x)

    TDb = -_TD(n, d, cnt, b)  # TDh

    if((np.isfinite(TDb).all() is False) or (TDb <= 0).any()):
        raise ValueError("Sample is too sparse")

    SDa = _SD(n, d, cnt, (2.394/(n * TDb))**(1/7))

    res = (c1 / SDa)**0.2
    return res


@jit
def fSD(
        h: float,
        c1: float,
        alph2: float,
        n: float,
        d: ArrayLike,
        cnt: ArrayLike
        ) -> float:
    """Function to call _SD for Newton-Raphson method to solve Eq. 12"""
    SDh = _SD(n, d, cnt, alph2 * h**(5/7))
    return (c1 / SDh)**0.2 - h


def sj_ste(x: ArrayLike,
           nbin: float=10000,
           hmin: bool=None,
           hmax: bool=None,
           tol: float=None
           ) -> float:
    """
    `sj_ste` implements the methods of Sheather & Jones (1991) to select the
    bandwidth using pilot estimation of derivatives. The algorithm for method
    "ste" solves an equation (via newton) and because of that, enlarges the
    interval (lower, upper) when the boundaries were not user-specified and do
    not bracket the root.

    This method uses pairwise binned distances: they are of complexity
    (O(n^2)) up to n = nb/2 and (O(n)) thereafter. Because of the binning,
    the results differ slightly when x is translated or sign-flipped.

    Parameters
    ----------
    x : ArrayLike
        array of values to calculate bandwidth from
    bin : int
        number of bins to find pairwise distances

    Returns
    -------
        float
            The bandwidth that solves Equation 12 of [1]

    Raises
    ------
    ValueError
        If x has non-finite elements, is too sparse, or has less than two
        elements, or if nbin is negative.
    
    References
    ----------
    [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    """

    if jnp.isfinite(x).all() is False:
        raise ValueError("Input has non-finite elements")

    if nbin <= 0:
        raise ValueError("`nbin` must be a positive number")

    n = len(x)
    if n < 2:
        raise ValueError("Array must have two or more elements")

    c1 = 1 / (2*jnp.sqrt(jnp.pi) * n)
    iqr = jnp.subtract(*jnp.percentile(x, jnp.array([75, 25])))
    scale = jnp.min(jnp.array([jnp.std(x), iqr/1.349]))
    a = 1.241 * scale * n**(-1/7)
    b = 1.230 * scale * n**(-1/9)

    d, cnt = _pairwise_binned_distance(nbin, x)

    TDb = -_TD(n, d, cnt, b)
    SDa = _SD(n, d, cnt, a)

    if((np.isfinite(TDb).all() is False) or (TDb <= 0).any()):
        raise ValueError("Sample is too sparse")

    alph2 = 1.357*(SDa / TDb)**(1/7)  # equation 12 from [1]

    if np.isfinite(alph2).all() is False:
        raise ValueError("Sample is too sparse")

    if hmax is None:
        hmax = 1.144 * scale * n**(-0.2)

    if hmin is None:
        hmin = 0.1 * hmax

    lower = hmin
    upper = hmax

    itry = 0
    while(
        fSD(lower, c1, alph2, n, d, cnt) * fSD(upper, c1, alph2, n, d, cnt) > 0
        ):

        if itry > 99:
            raise ValueError(
                "No solution in the specified range of bandwidths"
            )

        if itry % 2 == 0:
            upper *= 1.2
        else:
            lower /= 1.2

        print(f"Increasing bw.SJ() search interval {itry} to " +
                f"[{lower}, {upper}]")

        itry += 1
    
    if tol is None:
        tol = 0.1 * lower

    args = (c1, alph2, n, d, cnt)
    res = optimize.newton(fSD, (lower+upper)/2, tol=tol, args=args)
    return res
