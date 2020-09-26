# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-814839
# author: Andrea Chiang (andrea4@llnl.gov)
"""
Utility functionality for tdmtpy
"""

import numpy as np
from scipy.signal import fftconvolve


def xcorr(data,template,normalize=True):
    """
    Compute cross-correlation coefficient

    :param data: first time series
    :type data: :class:`numpy.ndarray`
    :param template: second time series
    :type template: :class:`numpy.ndarray`
    :param normalize: If ``True`` normalizes correlations
    :type normalize: bool
    :return: cross-correlation function
    :rtype: :class:`numpy.ndarray`
    """
    c = fftconvolve(data,template[::-1],mode="valid")   
    if normalize:
        lent = len(template)
        # Zero-padding
        pad = len(c) - len(data) + lent
        pad1 = (pad+1) // 2
        pad2 = pad // 2
        data = np.hstack([np.zeros(pad1,dtype=data.dtype),
                          data,
                          np.zeros(pad2,dtype=data.dtype)])
        # Rolling sum
        window_sum = np.cumsum(data**2)
        np.subtract(window_sum[lent:],window_sum[:-lent],out=window_sum[:-lent])
        norm = window_sum[:-lent]
        
        norm *= np.sum(template ** 2)
        norm = np.sqrt(norm)
        mask = norm <= np.finfo(float).eps
        #    c[:] = 0
        if c.dtype == float:
            c[~mask] /= norm[~mask]
        else:
            c = c/norm
        c[mask] = 0
    return c

def gaussj(A,b):
    """
    Solves the linear matrix equation Ax = b

    Computes the solution of a well-determined linear matrix equation
    by performing Gaussian-Jordan elimination for a given matrix and
    transforming it to a reduced echelon form.

    :param A: coefficient matrix
    :type A: :class:`numpy.ndarray`
    :param b: vector of dependent variables
    :type b: :class:`numpy.ndarray`
    :return: solution vector x so that Ax = b
    :rtype: :class:`numpy.ndarray`
    """
    # Make a copy to avoid altering input
    x = np.copy(b)
    n = len(x)

    # Gaussian-Jordan elimination
    for k in range(0,n-1):
        for i in range(k+1,n):
            if A[i,k] != 0.0:
                lam = A[i,k]/A[k,k]
                A[i,k+1:n] = A[i,k+1:n] - lam*A[k,k+1:n]
                x[i] = x[i]-lam*x[k]

    # Back substitution
    for k in range(n-1,-1,-1):
        x[k] = (x[k]-np.dot(A[k,k+1:n],x[k+1:n]))/A[k,k]

    return x
