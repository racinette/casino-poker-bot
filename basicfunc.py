import numpy as np
import numba as nb
import time


class vstacker:
    def __init__(self, w):
        self.w = w
        self.h = 0
        self.stack = []

    @staticmethod
    def stack(l, w=2, dtype=np.int):
        a = vstacker(w)
        for arr in l:
            a.append(arr)
        return a.build(dtype=dtype)

    def append(self, ar):
        if len(ar.shape) == 0:
            raise ValueError("Shape is 0.")
        elif len(ar.shape) == 1:
            h, w = 1, ar.shape[0]
        else:
            h, w = ar.shape
        if w != self.w:
            raise ValueError("Shape of array doesn't fit the shape of the stack. " + str(w) + " != " + str(self.w))
        self.h += h
        self.stack.append(ar)

    def build(self, dtype=np.int):
        r = np.empty(shape=(self.h, self.w), dtype=dtype)
        y = 0
        for ar in self.stack:
            if len(ar.shape) == 1:
                arh, arw = 1, ar.shape[0]
            else:
                arh, arw = ar.shape

            r[y:y+arh, :self.w] = ar
            y += arh
        return r


def eq(arr1, arr2):
    e = np.zeros(shape=arr1.shape, dtype=bool)
    for a in arr2:
        e = e | (arr1 == a)
    return e


def pred(c):
    if c == 0:
        return 51
    return c - 1


def succ(c):
    if c == 51:
        return 0
    return c + 1


def dist(a, b):
    """
    :param a:
    :param b:
    :return: returns distance between the maximum value a and the minimum value b (standard)
    """
    a = a % 13
    b = b % 13
    if a == b:
        return 0
    if a == 12:
        return b + 1
    return abs(b - a)


def conseq(a, b):
    """
    :param a: card a
    :param b: card b
    :return: -1 -> cards are of the same rank
              0 -> cards are consequent
             >0 -> number of gap cards between non-consequent cards
             то есть то, сколько можно вместить карт между двумя данными картами
    """
    return dist(a, b) - 1


def find(ar, i):
    n = 0
    for a in ar:
        if a == i:
            return n
        n += 1
    return -1


def amax(arr):
    m = arr[0]
    for e in arr:
        if e > m:
            m = e
    return m


def is_sorted_asc(a):
    for i in range(a.size - 1):
        if a[i+1] < a[i]:
            return False
    return True


def is_sorted_desc(a):
    for i in range(a.size - 1):
        if a[i+1] > a[i]:
            return False
    return True


def nandiv(a, b):
    zeros = b == 0
    c = np.divide(a, b, where=~zeros)
    c[zeros] = np.nan
    return c


def apply_except(f, ar, exs):
    nar = []
    for n in ar:
        if n not in exs:
            nar.append(f(n))
    return nar


@nb.njit
def derivative(y):
    d = np.empty_like(y)
    d[-1] = 0
    d[:-1] = y[1:] - y[:-1]
    return d


@nb.njit
def first_less_than(arr, val):
    for i in range(len(arr)):
        if arr[i] < val:
            return i
    return -1


@nb.njit
def first_greater_than(arr, val):
    for i in range(len(arr)):
        if arr[i] > val:
            return i
    return -1


@nb.njit
def left_local_extrema(y, x0=-1):
    if x0 < 0:
        x0 = len(y) - x0
    dy = derivative(y)
    # не включая сам элемент
    side = dy[:x0-1][::-1]
    minimum = x0 - first_less_than(side, 0.0) - 1
    maximum = x0 - first_greater_than(side, 0.0) - 1
    return minimum, maximum


def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count


def bincount3d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[arr.shape[0], arr.shape[1], bins], dtype=np.int64)
    index2d = np.ones_like(arr) * np.reshape(np.arange(arr.shape[1]), newshape=[1, arr.shape[1], 1])
    index3d = np.ones_like(arr) * np.reshape(np.arange(arr.shape[0]), newshape=[arr.shape[0], 1, 1])
    np.add.at(count, (index3d, index2d, arr), 1)

    return count


def squash(arr, n):
    sl = arr.shape[0] * n
    new_shape = [sl]
    new_shape.extend(arr.shape[1:])
    res = np.empty(shape=new_shape, dtype=arr.dtype)
    for i in range(n):
        res[i:sl:n] = arr
    return res


def maxgrad(f, x, y, stepx, stepy, D):
    z = f(x, y)

    # east
    # west
    # north
    # south

    Z = [
        (f(x + stepx, y), x + stepx, y),
        (f(x - stepx, y), x - stepx, y),
        (f(x, y + stepy), x, y + stepy),
        (f(x, y - stepy), x, y - stepy),
        (f(x + stepx, y + stepy), x + stepx, y + stepy),
        (f(x + stepx, y - stepy), x + stepx, y - stepy),
        (f(x - stepx, y + stepy), x - stepx, y + stepy),
        (f(x - stepx, y - stepy), x - stepx, y - stepy)
    ]

    m = np.asarray(Z).transpose()

    Z = m[0]
    X = m[1]
    Y = m[2]

    z_max = np.max(Z)
    zn_max = np.where(Z == z_max)[0][0]
    if z - z_max < D:
        return X[zn_max], Y[zn_max], Z[zn_max]
    else:
        return maxgrad(f, X[zn_max], Y[zn_max], stepx, stepy, D)
