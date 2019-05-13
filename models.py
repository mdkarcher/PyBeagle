from collections import namedtuple
import numpy as np

Model = namedtuple("Model", ['D', 'U', 'U_inv', 'pi'])
pi_JC = np.array([0.25, 0.25, 0.25, 0.25])
D_JC = np.array([0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])
U_JC = np.array([[1.0,  2.0,  0.0,  0.5],
                 [1.0, -2.0,  0.5,  0.0],
                 [1.0,  2.0,  0.0, -0.5],
                 [1.0, -2.0, -0.5,  0.0]])
U_inv_JC = np.array([[0.25,   0.25,  0.25,   0.25],
                     [0.125, -0.125, 0.125, -0.125],
                     [0.0,    1.0,   0.0,   -1.0],
                     [1.0,    0.0,  -1.0,    0.0]])
JC = Model(D_JC, U_JC, U_inv_JC, pi_JC)
