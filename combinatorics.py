import itertools
import numpy as np
from math import factorial


def pair_wr(a, w=2):
    N = a.shape[0]
    b = np.empty(shape=(c(N, 2), w), dtype=a.dtype)
    start = 0
    end = N - 1
    for n in range(N - 1):
        b[start:end, 0] = a[n]
        b[start:end, 1] = a[n+1:]
        start = start + (N - n - 1)
        end = start + (N - n - 2)
    return b


def pair(a, b=None, d=0):
    """
    :param a: array 1
    :param b: array 2
    :param d: если этот параметр 1, то будут все пары, кроме пар в стандартном порядке (a = [1,2,3], b = [4,5,6] ->
                [[1,4],[2,5],[3,6]] - НЕ БУДЕТ!!!
    :return: все возможные пары элементов из a и b БЕЗ ПОВТОРЕНИЙ
    """
    if b is None:
        b = a
    if a.shape != b.shape:
        raise ValueError("a and b shapes are not equal. " + str(a.shape) + " != " + str(b.shape))
    N = a.shape[0]
    c = np.empty(shape=(N * (N - d), 2), dtype=a.dtype)

    for n in range(1, N + 1 - d):
        start = N * (n - 1)
        end = N * n
        c[start:end, 0] = a
        c[start:end-n, 1] = b[n:]
        c[end-n:end, 1] = b[:n]
    return c


def c(n, k):
    if n < 0 or k < 0:
        raise ValueError(str(n) + ", " + str(k) + " - element(s) negative.")
    elif n == 0 or k == 0:
        return 0
    elif n < k:
        raise ValueError("Can't choose " + str(k) + " from " + str(n))
    elif n == k:
        return 1
    else:
        start = n - k + 1
        stop = n + 1
        numerator = 1
        for i in range(start, stop):
            numerator *= i
        return int(numerator / factorial(k))


def c_r(n, k):
    return c(n - 1 + k, k)


def c_rr(n, k, r):
    """
    Combinations with restricted repetition.
    :param n:
    :param k:
    :param r: max number of repetitions
    :return:
    """
    if r < 0:
        raise ValueError("Number of repetitions can't be a negative number.")
    elif r >= k:
        return c_r(n, k)
    elif r == 1:
        return c(n, k)

    combs = c_r(n, k)

    def possible_combs(arr, acc, ind, min):
        if ind > len(arr):
            return acc
        elif arr[ind] <= min:
            return possible_combs(arr, acc, ind + 1, min)
        else:
            pass

    for i in range(r + 1, k + 1):
        pass


def combinations(a, r):
    a = np.asarray(a)
    dt = np.dtype([('', a.dtype)]*r)
    b = np.fromiter(itertools.combinations(a, r), dt)
    return b.view(a.dtype).reshape(-1, r)


def random_probe(pn=6):
    deck = np.arange(52, dtype=np.int8)
    np.random.shuffle(deck)
    table = deck[:5]
    hands = deck[5:5+pn*2]
    hands = hands.reshape(pn, 2)
    probe = np.empty(shape=[pn, 7], dtype=np.int8)
    probe[:, :2] = hands
    probe[:, 2:] = table
    return probe

"""
combs = combinations(np.arange(13), 5)
bitmap = np.zeros(shape=(len(combs), 13), dtype=np.bool)
index = np.arange(combs.shape[0])
for n in range(combs.shape[1]):
    bitmap[index, combs[:, n]] = True
straights = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ], dtype=np.bool)
hc = bitmap[~np.isin(bitmap, straights)]


print(c(40, 4))
print(c(40, 3))
print(c(37, 2))
# 91390 * 36
# 9880 * 666


cs = combinations(np.arange(1, 53), 7)
print('combined')
sort = np.sort(cs, axis=1)
print('sorted')
normalized = sort - np.arange(0, 7)
print('normalized')
# del sort

N = [100, 93, 95, 96, 97, 99]
M = [2, 3, 5, 6, 7, 9, 1]

diff = sort.copy()
diff[1:] -= diff[:-1]
pow100 = np.power(100, np.arange(7))

A = np.sum(sort * pow100, axis=1)
B = np.sum(diff * pow100, axis=1)

res = (A + B) % 133784560
print('hashed')
print('|---------------------------|')
print('|        ~|results|~        |')
print('|---------------------------|')
print('|uniq|', np.unique(res).size)
print('|----|----------------------|')
print('|max |', np.max(res))
print('|----|----------------------|')
print('|min |', np.min(res))
print('|----|----------------------|')
print('|size|', res.size)
print('|____|______________________|')


# c | n  | r | return
# ----------------------
#   | 47 | 2 | 1081
#   | 48 | 3 | 17296
#   | 49 | 4 | 211876
#   | 50 | 5 | 2118760
#   | 51 | 6 | 18009460
#   | 52 | 7 | 133784560

@nb.njit
def func():
    offset1 = [0,
               1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276,
               300, 325, 351, 378, 406, 435, 465, 496, 528, 561, 595, 630, 666, 703, 741, 780, 820, 861, 903, 946, 990,
               1035]
    offset2 = [0,
               46, 137, 272, 450, 670, 931, 1232, 1572, 1950, 2365, 2816, 3302, 3822, 4375, 4960, 5576, 6222, 6897,
               7600, 8330, 9086, 9867, 10672, 11500, 12350, 13221, 14112, 15022, 15950, 16895, 17856, 18832, 19822,
               20825, 21840, 22866, 23902, 24947, 26000, 27060, 28126, 29197, 30272, 31350, 32430]
    offset3 = [0,
               1080, 3195, 6300, 10351, 15305, 21120, 27755, 35170, 43326, 52185, 61710, 71865, 82615, 93926, 105765,
               118100, 130900, 144135, 157776, 171795, 186165, 200860, 215855, 231126, 246650, 262405, 278370, 294525,
               310851, 327330, 343945, 360680, 377520, 394451, 411460, 428535, 445665, 462840, 480051, 497290, 514550,
               531825, 549110, 566401, 583695, 211876]
    offset4 = [0,
               17295, 50805, 99495, 162375, 238499, 326964, 426909, 537514, 657999, 787623, 925683, 1071513, 1224483,
               1383998, 1549497, 1720452, 1896367, 2076777, 2261247, 2449371, 2640771, 2835096, 3032021, 3231246,
               3432495, 3635515, 3840075, 4045965, 4252995, 4460994, 4669809, 4879304, 5089359, 5299869, 5510743,
               5721903, 5933283, 6144828, 6356493, 6568242, 6780047, 6991887, 7203747, 7415617, 7627491, 2118760]
    offset5 = [0,
               211875, 618330, 1203150, 1951155, 2848155, 3880906, 5037067, 6305158, 7674519, 9135270, 10678272,
               12295089, 13977951, 15719718, 17513845, 19354348, 21235771, 23153154, 25102002, 27078255, 29078259,
               31098738, 33136767, 35189746, 37255375, 39331630, 41416740, 43509165, 45607575, 47710830, 49817961,
               51928152, 54040723, 56155114, 58270870, 60387627, 62505099, 64623066, 66741363, 68859870, 70978503,
               73097206, 75215944, 77334697, 79453455, 18009460]
    offset6 = [0,
               2118759, 6144402, 11882349, 19154235, 27796875, 37661274, 48611681, 60524686, 73288359, 86801430,
               100972509, 115719345, 130968123, 146652798, 162714465, 179100764, 195765319, 212667210, 229770477,
               247043655, 264459339, 281993778, 299626497, 317339946, 335119175, 352951534, 370826397, 388734909,
               406669755, 424624950, 442595649, 460577976, 478568871, 496565954, 514567405, 532571859, 550578315,
               568586058, 586594593, 604603590, 622612839, 640622214, 658631645, 676641097, 694650555, 133784560]

    A = [0 for i in range(0)]
    for a in range(46):
        # 46
        for b in range(a, 46):
            # 1081 + 1034 = 2115
            for c in range(b, 46):
                #  46  92 136 179 221 262
                #  46  45  44  43  42  41
                for d in range(c, 46):
                    for e in range(d, 46):
                        for f in range(e, 46):
                            # 18009460
                            for g in range(f, 46):
                                ind = (
                                        g +
                                        (f * 46 - offset1[f]) +
                                        (e * 1081 - offset2[e]) +
                                        (d * 17295 - offset3[d]) +
                                        (c * 211875 - offset4[c]) +
                                        (b * 2118759 - offset5[b]) +
                                        (a * 18009459 - offset6[a])
                                )
                                A.append(ind)

    B = np.array(A, dtype=np.int64)
    diffs = np.diff(B)
    tears = np.where(diffs != 1)[0]
    offsets = diffs[tears]
    offsets = offsets - 1
    offsets = np.cumsum(offsets)
    print(len(offsets))

    equal = B == np.arange(B.size)
    print(len(B))
    print(np.unique(B).size)
    print(np.all(equal))


func()



@nb.njit
def func():
    offset1 = np.cumsum(np.arange(46))
    offset2 = np.cumsum(np.ones(46, dtype=np.int64) * 46 - np.arange(46)) - offset1

    A = [0 for i in range(0)]
    for a in range(46):
        # 46
        for b in range(a, 46):
            # 1081 + 1034 = 2115
            for c in range(b, 46):
                #  46  92 136 179 221 262
                #  46  45  44  43  42  41
                for d in range(c, 46):
                    for e in range(d, 46):
                        for f in range(e, 46):
                            for g in range(f, 46):
                                ind = a + b * 46 - offset1[b] + c * 1081 - offset2[c] + d * 17296 + e * 211876 + f * 2118760 + g * 18009460
                                A.append(ind)

    B = np.array(A)
    B.sort()
    equal = B == np.arange(B.size)
    print(len(B))
    print(np.unique(B).size)
    print(np.all(equal))
"""