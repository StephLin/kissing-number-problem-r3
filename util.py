# Copyright 2020 Yu-Kai Lin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
import warnings
import numpy as np
import scipy.optimize as opt
import sympy

OptResutType = typing.Tuple[float, float]


def fminbound(func: callable,
              left: float,
              right: float,
              method: str = 'Bounded') -> OptResutType:
    """Minimum of the function in a closed interval.
    
    Arguments:
        func {callable} -- Callable function.
        left {float} -- Lower bound of the interval.
        right {float} -- Higher bound of the interval.

    Keyword Arguments:
        method {str} -- name of algorithm (default: {'Bounded'}).
                        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy-optimize-minimize-scalar

    Returns:
        arg {float} -- argument minimum of the function.
        val {float} -- minimum of the function.
    
    Notes:
        This function is based on `scipy.optimize.minimize_scalar`.
    """
    bounds = (left, right)

    res = opt.minimize_scalar(func,
                              bracket=bounds,
                              bounds=bounds,
                              method=method)

    if not res.success:
        warnings.warn(res.message)

    if res.x < left or res.x > right:
        warnings.warn('x out of bounds')

    val = func(res.x)
    return res.x, val


def fmaxbound(func: callable,
              left: float,
              right: float,
              method: str = 'Bounded') -> OptResutType:
    """Maximum of the function in a closed interval.
    
    Arguments:
        func {callable} -- Callable function.
        left {float} -- Lower bound of the interval.
        right {float} -- Higher bound of the interval.

    Keyword Arguments:
        method {str} -- name of algorithm (default: {'Bounded'}).
                        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy-optimize-minimize-scalar

    Returns:
        arg {float} -- argument maximum of the function.
        val {float} -- maximum of the function.
    
    Notes:
        This function is based on `scipy.optimize.minimize_scalar`.
    """
    wrapper = lambda x: -func(x)
    arg, _ = fminbound(wrapper, left, right)
    val = func(arg)
    return arg, val


def deg2red(deg):
    return deg * np.pi / 180


def red2deg(red):
    return red * 180 / np.pi


def verify_f_function(f, z, N):
    a_positive = []
    a_negative = []

    epsilon = (z + 1) / N
    for j in range(N + 1):
        aj = -1 + epsilon * j
        if aj <= -1 * f.t0:
            a_positive.append(aj)
        else:
            a_negative.append(aj)

    for a in a_positive:
        fa = f(a)
        if fa < 0:
            return (False, a, fa)

    for i in range(len(a_positive) - 1):
        fa1 = f(a_positive[i])
        fa2 = f(a_positive[i + 1])
        if fa1 <= fa2:
            return (False, a_positive[i], fa1)

    for a in a_negative:
        fa = f(a)
        if fa > 0 and np.abs(a + f.t0) > 1e-2:
            return (False, a, fa)
    return (True, None, None)