import pandas as pd
import numpy as np
import scipy.stats as sci
from math import pi, sin


def rolling_mean(arr, n, smoothing):
    a = pd.Series(arr)
    a = a.rolling(n).mean()
    a = a.rolling(smoothing).mean()
    return a.to_numpy()


def threshold(i, upper=1, bottom=0):
    if i < bottom:
        return bottom
    elif i > upper:
        return upper
    return i


def sinact(i, alpha=0.43, beta=0.1):
    if i <= 0:
        return 0
    elif i >= 1:
        return sin(1 / alpha)
    return sin(i ** (pi - i) / (alpha * i) + beta)


def qact(x, p=0.5, h=1, w=1, beta=0.0):
    if x <= 0:
        return 0
    elif x >= w:
        return w * h
    return h * (abs(x / w) ** p + beta) / (1 + beta)


def euclid(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class LinearFunc:
    def __init__(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        # y = kx + b
        # kx1 + b - y1 = kx2 + b - y2
        # kx1 - y1 = kx2 - y2
        # kx1 - kx2 = y1 - y2
        # k(x1 - x2) = y1 - y2
        # k = (y1 - y2)/(x1 - x2), (x1 - x2) != 0
        # y - kx = b
        if (x1 - x2) == 0:
            # вертикальная линия
            def yvert(self, x):
                return np.inf
            self.y = yvert
        else:
            self.k = (y1 - y2) / (x1 - x2)
            self.b = y1 - self.k * x1

    def y(self, x):
        return x * self.k + self.b


class LinRegression:
    def __init__(self, y):
        N = len(y)
        x = np.arange(N)
        self.k, self.b, self.r_value, self.p_value, self.std_err = sci.linregress(x, y)

    def y(self, x):
        return self.k * x + self.b


class DTR:
    """
    Double Trend Roller
    """
    def __init__(self, l1_len, l2_len, f1, f2, dtype=np.float):
        self.f1 = f1
        self.f2 = f2
        self.l1_len = l1_len
        self.l2_len = l2_len
        self.layer1 = np.empty(shape=l1_len + l2_len - 1, dtype=dtype)
        self.layer1[:] = np.nan
        self.layer2 = np.empty(shape=l2_len, dtype=dtype)
        self.layer2[:] = np.nan
        self.l1_fill = 0
        self.l2_fill = 0
        self.act_len = self.l1_len + self.l2_len - 1

    def push(self, i):
        self.layer1[1:] = self.layer1[:-1]
        self.layer1[0] = i

        j = 0
        for n in range(self.l1_len - self.l2_len + 2, self.act_len):
            if not np.isnan(self.layer1[n]):
                self.layer2[j] = self.f1(self.layer1[j:n+j])
            else:
                break
            j += 1

    def put(self, arr):
        arr = arr[:self.act_len]
        self.layer1[:] = np.nan
        self.layer2[:] = np.nan

        self.layer1[:len(arr)] = arr
        j = 0
        for n in range(self.l1_len - self.l2_len + 2, self.act_len):
            if not np.isnan(self.layer1[n]):
                self.layer2[j] = self.f1(self.layer1[j:n + j])
            else:
                break
            j += 1

    def roll(self):
        if np.all(~np.isnan(self.layer2)):
            return self.f2(self.layer2)
        return np.nan

