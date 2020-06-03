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

import numpy as np
import scipy.optimize as opt
import sympy
from sympy.abc import theta as sympy_theta
from sympy.abc import psi as sympy_psi
from sympy.abc import u as sympy_u
import matplotlib.pyplot as plt

import lib
import util

T0 = 0.590689689324012
THETA0 = float(np.arccos(T0))
GAMMA = np.arccos(np.sqrt(2 / 3))
R0 = GAMMA
F_COEFF = [
    -1 / 200, 1 / 10, -213 / 100, -83 / 10, 343 / 40, 18333 / 400, 0,
    -1287 / 20, 0, 2431 / 80
]

Symbol = sympy.Symbol
deg2red = util.deg2red


def rho(s: float) -> float:
    return 2 * np.arccos(1 / (2 * np.cos(s / 2)))


class F(lib.Polynomial):
    def __init__(self, coeffs=F_COEFF, t0=T0, theta0=THETA0):
        self.t0 = t0
        self.theta0 = theta0
        super(F, self).__init__(coeffs)


f = F()


class CapitalF1Tilde(lib.Function):
    def __init__(self, f_function=f):
        self.f = f_function

    def __call__(self, theta: float, psi: float) -> float:
        return self.f(-1 * np.cos(theta)) + self.f(-1 * np.cos(psi - theta))

    def symbolic(self, theta: Symbol, psi: Symbol) -> sympy.Function:
        return self.f(-1 * sympy.cos(theta)) + self.f(
            -1 * (sympy.cos(psi) * sympy.cos(theta) +
                  sympy.sin(psi) * sympy.sin(theta)))

    def dtheta_symbolic(self, theta: Symbol, psi: Symbol) -> sympy.Function:
        return sympy.diff(self.symbolic(theta, psi), theta)

    def dtheta(self, theta: float, psi: float) -> float:
        subs = {sympy_theta: theta, sympy_psi: psi}
        return self.dtheta_symbolic(sympy_theta, sympy_psi).evalf(subs=subs)


class F1(lib.Function):
    def __init__(self, f_function=f):
        self.ftilde = CapitalF1Tilde(f_function)

    def __call__(self, psi, visible_evaluate=False, visible_name=''):
        wrapper = lambda theta: self.ftilde(theta, psi)

        left = psi / 2
        right = self.ftilde.f.theta0

        x, val = util.fmaxbound(wrapper, left, right)

        if visible_evaluate:
            plt.figure('{} - function'.format(visible_name))
            ymin, ymax = self.plot_func(psi, left, right)
            plt.plot([x, x], [ymin, ymax], '--')
            y = self.ftilde(x, psi)
            plt.plot([left, right], [y, y], '--')
            plt.text(x, ymax, '{:.5f}'.format(y))

            plt.figure('{} - derivative'.format(visible_name))
            ymin, ymax = self.plot_diff(psi, left, right)
            plt.plot([x, x], [ymin, ymax], '--')
            plt.text(x, ymax, '{:.5f}'.format(self.ftilde.dtheta(x, psi)))

        return val

    def plot_func(self, psi, left, right):
        xs = np.linspace(left, right, 100)
        ys = [self.ftilde(x, psi) for x in xs]
        plt.plot(xs, ys)

        return np.min(ys), np.max(ys)

    def plot_diff(self, psi, left, right):
        xs = np.linspace(left, right, 100)
        ys = [self.ftilde.dtheta(x, psi) for x in xs]
        plt.plot(xs, ys)
        plt.plot([left, right], [0, 0], '--')

        return np.min(ys), np.max(ys)


class CapitalF2Tilde(lib.Function):
    def __init__(self, f_function=f):
        self.f = f_function

    def __call__(self, u, psi):
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        sin_deg60 = np.sin(deg2red(60))
        cos_deg60 = .5
        base = cos_deg60 * cos_psi
        cos_theta1 = base + sin_deg60 * sin_psi * np.cos(R0 - u)
        cos_theta2 = base + sin_deg60 * sin_psi * np.cos(R0 + u)

        return self.f(1) + self.f(-1 * cos_theta1) + self.f(-1 * cos_theta2)

    def symbolic(self, u: Symbol, psi: Symbol) -> sympy.Function:
        sin_psi = sympy.sin(psi)
        cos_psi = sympy.cos(psi)
        sin_deg60 = np.sin(deg2red(60))
        cos_deg60 = .5
        base = cos_deg60 * cos_psi
        cos_theta1 = base + sin_deg60 * sin_psi * sympy.cos(R0 - u)
        cos_theta2 = base + sin_deg60 * sin_psi * sympy.cos(R0 + u)
        return self.f(1) + self.f(-1 * cos_theta1) + self.f(-1 * cos_theta2)

    def dtheta_symbolic(self, u: Symbol, psi: Symbol) -> sympy.Function:
        return sympy.diff(self.symbolic(psi, u), u)

    def dtheta(self, u: float, psi: float) -> float:
        subs = {sympy_psi: psi, sympy_u: u}
        return self.dtheta_symbolic(sympy_psi, sympy_u).evalf(subs=subs)


class F2(lib.Function):
    def __init__(self, f_function=f):
        self.ftilde = CapitalF2Tilde(f_function)

    def __call__(self, psi, visible_evaluate=False, visible_name=''):
        wrapper = lambda u: self.ftilde(u, psi)

        left = 0
        right = np.arccos(1 / (np.sqrt(3) * np.tan(psi))) - R0

        x, val = util.fmaxbound(wrapper, left, right)

        if visible_evaluate:
            plt.figure('{} - function'.format(visible_name))
            ymin, ymax = self.plot_func(psi, left, right)
            plt.plot([x, x], [ymin, ymax], '--')
            y = self.ftilde(x, psi)
            plt.plot([left, right], [y, y], '--')
            plt.text(x, ymax, '{:.5f}'.format(y))

            plt.figure('{} - derivative'.format(visible_name))
            ymin, ymax = self.plot_diff(psi, left, right)
            plt.plot([x, x], [ymin, ymax], '--')
            plt.text(x, ymax, '{:.5f}'.format(self.ftilde.dtheta(x, psi)))

        return val

    def plot_func(self, psi, left, right):
        xs = np.linspace(left, right, 100)
        ys = [self.ftilde(x, psi) for x in xs]
        plt.plot(xs, ys)

        return np.min(ys), np.max(ys)

    def plot_diff(self, psi, left, right):
        xs = np.linspace(left, right, 100)
        ys = [self.ftilde.dtheta(x, psi) for x in xs]
        plt.plot(xs, ys)
        plt.plot([left, right], [0, 0], '--')

        return np.min(ys), np.max(ys)