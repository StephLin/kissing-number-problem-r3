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

# handle import issue
# pylint: disable=import-error,no-name-in-module
import os
import sys
libpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(libpath)

import numpy as np
# import matplotlib.pyplot as plt

# import lib
# import util
from util import deg2red
import s2

f = s2.F()
f1 = s2.F1()

h2 = f1(deg2red(60)) + f(1)
print('h2:', h2)

h4_1 = f(1) + f1(s2.rho(2 * f.theta0)) + f1(s2.rho(deg2red(77)))
# use `visible_evaluate=True` to inspect visualized optimization result
# f1(s2.rho(2 * f.theta0), visible_evaluate=True, visible_name='4-1')
h4_2 = f(1) + f1(deg2red(90)) + f1(deg2red(77))
print('h4:', end=' ')
print(h4_1, h4_2, sep='\n    ')

f2 = s2.F2()

h3_1 = f2(deg2red(38)) + f(-np.cos(s2.GAMMA))
# use `visible_evaluate=True` to inspect visualized optimization result
# f2(deg2red(38), visible_evaluate=True, visible_name='3-1')
h3_2 = f2(deg2red(41)) + f(-np.cos(deg2red(38)))
h3_3 = f2(deg2red(44)) + f(-np.cos(deg2red(41)))
h3_4 = f2(deg2red(48)) + f(-np.cos(deg2red(44)))
h3_5 = f2(s2.THETA0) + f(-np.cos(deg2red(48)))

print('h3:', end=' ')
print(h3_1, h3_2, h3_3, h3_4, h3_5, sep='\n    ')

# if `visible_evaluate=True` used, you are able to save them
# import matplotlib._pylab_helpers as pylab_helpers

# figures = [
#     manager.canvas.figure
#     for manager in pylab_helpers.Gcf.get_all_fig_managers()
# ]

# for figure in figures:
#     print('saving figure {}'.format(figure.canvas.get_window_title()))
#     figure.savefig('{}.png'.format(figure.canvas.get_window_title()), dpi=1000)