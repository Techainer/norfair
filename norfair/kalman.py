from __future__ import absolute_import, division

import numba
import numpy as np

from numpy import zeros, eye
from numba import float64

spec = [
    ('dim_x', float64),
    ('dim_z', float64),
    ('dim_u', float64),
    ('x', float64[:, ::1]),
    ('P', float64[:, ::1]),
    ('Q', float64[:, ::1]),
    ('B', float64[:, ::1]),
    ('F', float64[:, ::1]),
    ('H', float64[:, ::1]),
    ('R', float64[:, ::1]),
    ('_alpha_sq', float64),
    ('M', float64[:, ::1]),
    ('K', float64[:, ::1]),
    ('y', float64[:, ::1]),
    ('S', float64[:, ::1]),
    ('_I', float64[:, ::1]),
]


@numba.experimental.jitclass(spec=spec)
class KalmanFilter(object):
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1), dtype=np.float64)        # state
        self.P = eye(dim_x, dtype=np.float64)               # uncertainty covariance
        self.Q = eye(dim_x, dtype=np.float64)               # process uncertainty
        # self.B = None                     # control transition matrix
        self.F = eye(dim_x, dtype=np.float64)               # state transition matrix
        self.H = zeros((dim_z, dim_x), dtype=np.float64)    # Measurement function
        self.R = eye(dim_z, dtype=np.float64)               # state uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = zeros((dim_z, dim_z), dtype=np.float64) # process-measurement cross correlation

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = zeros((dim_x, dim_z), dtype=np.float64) # kalman gain
        self.y = zeros((dim_z, 1), dtype=np.float64)
        self.S = zeros((dim_z, dim_z), dtype=np.float64) # system uncertainty

        # identity matrix. Do not alter this.
        self._I = eye(dim_x, dtype=np.float64)


    def predict(self):
        F = self.F
        self.x = self.F@self.x
        
        # P = FPF' + Q
        self.P = self._alpha_sq * (F@self.P)@F.T + self.Q


    def update(self, z, H=None):
        R = self.R
        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - H@self.x

        # common subexpression for speed
        PHT = self.P@H.T

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = H@PHT + R
        SI = np.linalg.inv(self.S) # inverse system uncertainty

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = PHT@SI

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + self.K@self.y

        I_KH = self._I - self.K@H

        # NOTE: How will this impact accuracy?
        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # self.P = (I_KH@self.P)@I_KH.T + (self.K@R)@self.K.T

        # On the other hand,
        # P = (I-KH)P usually seen in the literature
        # and also it must be faster
        self.P = I_KH@self.P


    @property
    def alpha(self):
        return self._alpha_sq**.5

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2
