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
A set of functions to calculate the bandwidth of a normal kernel to create
a kernel density estimator of some data x

This file is mostly a pythonic translation of R code from
src/library/stats/R/bandwidths.R. Documentation comes from
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth

Some functions have been cythonised and can be found in cbandwidths.pyx
"""

import numpy as np
import scipy
from cbandwidths import bw_phi4, bw_phi6, bw_den


def nrd0(x):
    """
    bw.nrd0 implements a rule-of-thumb for choosing the bandwidth of a
    Gaussian kernel density estimator. It defaults to 0.9 times the minimum of
    the standard deviation and the interquartile range divided by 1.34 times
    the sample size to the negative one-fifth power (= Silverman's
    ‘rule of thumb’, Silverman (1986, page 48, eqn (3.31))).

    @param x: array of values to calculate bandwidth from
    @return bandwidth: an bandwidth suggestion
    """
    if len(x) < 2:
        print("need at least 2 data points")
        exit

    std = np.std(x)  # calculate std dev
    q75, q25 = np.percentile(x, [75, 25])  # calculate IQR

    lo = min(std, (q75-q25)/1.34898)  # minimum of these two values

    return 0.9 * lo * len(x)**-0.2


def nrd(x):
    """
    bw.nrd is the more common variation given by Scott (1992), using factor
    1.06.

    @param x: array of values to calculate bandwidth from
    @return bandwidth: an bandwidth suggestion
    """
    if len(x) < 2:
        print("need at least 2 data points")
        exit

    std = np.std(x)  # calculate std dev
    q75, q25 = np.percentile(x, [75, 25])  # calculate IQR

    lo = min(std, (q75-q25)/1.34898)  # minimum of these two values

    return 1.06 * lo * len(x)**-0.2


def sj(x, nbin=1000, method='dpi', hmax=None):
    """
    bw.SJ implements the methods of Sheather & Jones (1991) to select the
    bandwidth using pilot estimation of derivatives. The algorithm for method
    "ste" solves an equation (via uniroot) and because of that, enlarges the
    interval c(lower, upper) when the boundaries were not user-specified and do
    not bracket the root.

    This method uses pairwise binned distances: they are of complexity
    (O(n^2)) up to n = nb/2 and (O(n)) thereafter. Because of the binning,
    the results differ slightly when x is translated or sign-flipped.

    @param x: array of values to calculate bandwidth from
    @param nbin: number of bins to find pairwise distances
    @param method: use the direct plug-in ('dpi') or solve-the-equation ('ste')
                   method
    @param hmax: maximum bandwidth to search over

    @return bandwidth: an bandwidth suggestion
    """
    n = len(x)
    if n < 2:
        print("need at least 2 data points")
        exit

    if nbin < 0:
        print('nbin must be a positive number')
        exit

    c1 = 1/(2*np.sqrt(np.pi)*n)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    scale = min(np.std(x), iqr/1.349)
    a = 1.24 * scale * n**(-1/7)
    b = 1.23 * scale * n**(-1/9)

    alph2 = 0.
    d, cnt = bw_den(nbin, x)
    SDh = lambda h: bw_phi4(n, d, cnt, h)
    TDh = lambda h: bw_phi6(n, d, cnt, h)
    fSD = lambda h: (c1/SDh(alph2 * h*(5/7)))**0.2 - h

    TD = -TDh(b)

    if((np.isfinite(TD) is False) or (TD <= 0)):
        print("sample is too sparse to find TD")
        exit

    # direct plug-in method
    if method == 'dpi':
        res = (c1/SDh((2.394/(n * TD))**(1/7)))**0.2
        return res

    # solve-the-equation
    elif method == 'ste':

        if hmax is None:
            hmax = 1.144 * scale * n**(-0.2)

        lower = 0.1 * hmax
        upper = hmax
        tol = 0.1 * lower

        alph2 = 1.357*(SDh(a)/TD)**(1/7)
        if np.isfinite(alph2) is False:
            print("sample is too sparse to find alph2")
            return

        itry = 0
        while(fSD(lower) * fSD(upper) > 0):

            if(itry > 99):
                print("no solution in the specified range of bandwidths")
                exit

            if(itry % 2 == 0):
                upper *= 1.2
            else:
                lower /= 1.2

            print(f"increasing bw.SJ() search interval {itry} to " +
                  f"[{lower},{upper}]")

            itry += 1

        res = scipy.optimize.newton(fSD, (lower+upper)/2, tol=tol)
        return res

    else:
        print("invalid method")
        return
