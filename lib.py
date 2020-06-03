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

import abc
from typing import Union, List

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import sympy
from sympy.abc import t as sympy_t

import util


class Function(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def fminbound(self, a, b, **kwargs):
        argmin = util.fminbound(self, a, b)
        return argmin, self(argmin)

    def fmaxbound(self, a, b, **kwargs):
        negative_func = lambda x: -1 * self.__call__(x, **kwargs)
        argmax, _ = util.fminbound(negative_func, a, b)
        return argmax, self(argmax)


class Polynomial(Function):
    def __init__(self, coeffs: Union[List[float], np.ndarray]):
        """Generator of Polynomial.
        
        Arguments:
            coeffs {Union[List[float], np.ndarray]} -- Coefficients, start with degree 0.
        """
        self._coeffs = np.asanyarray(coeffs).flatten()

    def __call__(self, x: float) -> float:
        """Caller of Polynomial.
        
        Arguments:
            x {float} -- Input
        
        Returns:
            float -- Value of Polynomial(x)
        """
        if self.dim < 0:
            return 0

        value = x * 0
        for idx, coeff in enumerate(self.coeffs):
            value += x**(idx) * coeff

        return value

    def symp(self, x):
        return self.__call__(x)

    def plot(self, left, right, density=100):
        plt.figure()
        xs = np.linspace(left, right, density)
        ys = [self.__call__(x) for x in xs]
        plt.plot(xs, ys)

    @property
    def coeffs(self) -> np.array:
        return self._coeffs

    @property
    def dim(self) -> int:
        return self._coeffs.shape[0] - 1


class GegenbauerPolynomial(Polynomial):
    def __init__(self, n, k):
        self.symbols = None
        coeffs = self._get_coeffs(n, k)
        super(GegenbauerPolynomial, self).__init__(coeffs)

    def _get_coeffs(self, n, k):
        self.symbols = self._get_symbols(n, k)
        if isinstance(self.symbols, int) or isinstance(self.symbols, float):
            return [self.symbols]
        return self.symbols.all_coeffs()[::-1]

    def _get_symbols(self, n, k):
        if k == 0:
            return 1
        if k == 1:
            return sympy.poly(sympy_t)
        numerator = (2 * k + n - 4) * sympy_t * self._get_symbols(n, k - 1)
        numerator -= (k - 1) * self._get_symbols(n, k - 2)
        denominator = k + n - 3
        return sympy.poly(numerator / denominator)
