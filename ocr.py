from scr.image import *
from PIL import Image
import numpy as np
from functools import partial
from os import listdir
import cv2
from skimage.segmentation import flood
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scr.image import shift_eq


class LineReader:
    def __init__(self, delta=0, pre=0, post=-2, x_train=None, y_train=None, dc=None):
        self.dc = dc
        self.delta = delta
        self.pre = pre
        self.post = post
        if self.dc is None:
            self.dc = DigitClassifier(features='l', z=6, ccx=8, ccy=8)
            self.dc.train(x_train, y_train)

    def readline(self, img):
        s = ""
        digits = waterfall_box(img, self.delta)[0][self.pre:self.post]
        for digit in digits:
            digit = trim(digit)[0]
            if np.sum(digit) / digit.size > 0.7:
                s += "."
            else:
                s += str(self.dc.classify(digit)[0])
        return s

    def set_delta(self, d):
        self.delta = d


class DigitClassifier:
    SYM_FTR_LEN = 2
    ZONE_FTR = 36
    LBF_FTR = 36

    def __init__(self, features='all', clsfier='nn',
                 digit_size=(25, 15), z=2, ccx=6, ccy=6,
                 xray=3, xweights=(0.25, 1, 0.25),
                 yray=3, yweights=(0.25, 1, 0.25),
                 div=(5, 5)):
        self.div = div
        self.digit_size = digit_size
        self.xray = xray
        self.xweights = np.array(xweights, dtype=np.float64)
        self.yray = yray
        self.yweights = np.array(yweights, dtype=np.float64)
        if features == 'all':
            self.ftrs = [
                partial(DigitClassifier.zigzag_ftr, z=z),
                partial(DigitClassifier.transpose_zig_zag, z=z),
                partial(DigitClassifier.zone_ftr, ccx=ccx, ccy=ccy),
                DigitClassifier.fill_ftr,
                DigitClassifier.sym_ftr,
                DigitClassifier.lbf_ftr,
            ]
            self.FEATURE_VECTOR_LEN = 78 + (1 + z * 2) * 4
        else:
            self.FEATURE_VECTOR_LEN = 0
            self.ftrs = []
            if features.find('z') > -1:
                self.ftrs.append(partial(DigitClassifier.zigzag_ftr, z=z))
                self.FEATURE_VECTOR_LEN += (1 + z * 2) * 2
            if features.find('t') > -1:
                self.ftrs.append(partial(DigitClassifier.transpose_zig_zag, z=z))
                self.FEATURE_VECTOR_LEN += (1 + z * 2) * 2
            if features.find('f') > -1:
                self.ftrs.append(DigitClassifier.fill_ftr)
                self.FEATURE_VECTOR_LEN += 4
            if features.find('s') > -1:
                self.ftrs.append(DigitClassifier.sym_ftr)
                self.FEATURE_VECTOR_LEN += 2
            if features.find('l') > -1:
                self.ftrs.append(DigitClassifier.lbf_ftr)
                self.FEATURE_VECTOR_LEN += 36
            if features.find('o') > -1:
                self.ftrs.append(partial(DigitClassifier.zone_ftr, ccx=ccx, ccy=ccy))
                self.FEATURE_VECTOR_LEN += 36
            if features.find('x') > -1:
                self.ftrs.append(partial(DigitClassifier.max_x_dist_reduct, ray=self.xray, weights=self.xweights))
                self.FEATURE_VECTOR_LEN += self.digit_size[1] * 2
            if features.find('y') > -1:
                self.ftrs.append(partial(DigitClassifier.max_y_dist_reduct, ray=self.yray, weights=self.yweights))
                self.FEATURE_VECTOR_LEN += self.digit_size[0] * 2
            if features.find('r') > -1:
                self.ftrs.append(partial(DigitClassifier.zone_reduct, div=self.div))
                self.FEATURE_VECTOR_LEN += self.div[0] * self.div[1]
        self.z = z

        if clsfier == 'nn':
            classifier = MLPClassifier(
                hidden_layer_sizes=(self.FEATURE_VECTOR_LEN, self.FEATURE_VECTOR_LEN * 2),
                max_iter=2000,
                batch_size=10,
                activation='relu',
                solver='adam',
                tol=0.0000001,
                random_state=228
            )
        elif clsfier == 'svc':
            classifier = SVC(C=2, kernel='poly', degree=2, gamma=0.2)
        else:
            raise ValueError("Not supported.")

        self.classifier = classifier

    def train(self, x_train, y_train):
        train_set_len = len(x_train)
        x_train_feature_set = np.empty(shape=[train_set_len, self.FEATURE_VECTOR_LEN], dtype=np.float64)

        for n in range(train_set_len):
            x = x_train[n]
            x_train_feature_set[n] = self.ftr(x)

        self.classifier.fit(x_train_feature_set, y_train)

    def classify(self, img):
        if img.shape[0] == 0 or img.shape[1] == 1:
            return "N"
        img = cv2.resize(img, self.digit_size, interpolation=cv2.INTER_CUBIC)
        x = self.ftr(img)
        return self.classifier.predict(x.reshape(1, -1))

    @staticmethod
    def transpose_zig_zag(img, z=2):
        return DigitClassifier.zigzag_ftr(np.transpose(img), z)

    @staticmethod
    def max_y_dist_reduct(img, weights, ray=3):
        y, x = img.shape
        y_half = y // 2
        A = img[:y_half]
        B = img[y_half:]
        A_rng = np.arange(1, A.shape[0] + 1)[::-1]
        B_rng = np.arange(1, B.shape[0] + 1)
        A_dist = (A.T * A_rng).T
        A_dist = np.amax(A_dist, axis=0)
        B_dist = (B.T * B_rng).T
        B_dist = np.amax(B_dist, axis=0)
        dists = np.zeros(shape=(2, x), dtype=np.float64)
        for xn in range(x - ray - 1):
            rvx = x - xn - ray
            dists[0, xn] += np.mean(A_dist[xn:xn + ray] * weights)
            dists[0, rvx + ray - 1] += np.mean(A_dist[rvx:rvx + ray] * weights)
            dists[1, xn] += np.mean(B_dist[xn:xn + ray] * weights)
            dists[1, rvx + ray - 1] += np.mean(B_dist[rvx:rvx + ray] * weights)
        return dists.flatten()

    @staticmethod
    def max_x_dist_reduct(img, weights, ray=3):
        y, x = img.shape
        x_half = x // 2
        A = img[:, :x_half]
        B = img[:, x_half:]
        A_rng = np.arange(1, A.shape[1] + 1)[::-1]
        B_rng = np.arange(1, B.shape[1] + 1)
        A_dist = A * A_rng
        A_dist = np.amax(A_dist, axis=1)
        B_dist = B * B_rng
        B_dist = np.amax(B_dist, axis=1)
        dists = np.zeros(shape=(y, 2), dtype=np.float64)
        for yn in range(y - ray - 1):
            rvy = y - yn - ray
            dists[yn, 0] += np.mean(A_dist[yn:yn + ray] * weights)
            dists[rvy + ray - 1, 0] += np.mean(A_dist[rvy:rvy + ray] * weights)
            dists[yn, 1] += np.mean(B_dist[yn:yn + ray] * weights)
            dists[rvy + ray - 1, 1] += np.mean(B_dist[rvy:rvy + ray] * weights)
        return dists.flatten()

    @staticmethod
    def zone_reduct(img, div):
        ys, xs = img.shape
        yz, xz = div
        sizey, sizex = ys // yz, xs // xz
        reduct = np.zeros(shape=(yz, xz), dtype=np.float64)
        for x in range(xz):
            for y in range(yz):
                reduct[y, x] = np.sum(img[y * sizey:(y + 1) * sizey, x * sizex:(x + 1) * sizex])
        return reduct.flatten()

    @staticmethod
    def zigzag_ftr(img, z=2):
        """
        алгоритм становится в середине изображения и оттуда ищет наиболее удаленную точку
        затем изображение делится на z * 2 равных сегмента вдоль оси х
        из вышеописанной точки он снова переходит в наиболее удаленную точку в первом сегмент
        и последнее действие повторяется, пока не обойдет все сегменты
        :param img: изображение
        :param z: коэффициент зиг-загов.
        :return: приметы изображения
        """

        h, w = img.shape
        cy, cx = cntr(img)
        # наиболее удаленный от центра пункт
        dst = dst_map(img, cx, cy)

        zz = 2 * z

        segment_h = h / zz

        ssh = int(segment_h)
        topmost = dst[:ssh]
        bottommost = dst[h - ssh:]

        coords = np.empty(shape=(zz + 1, 2), dtype=np.int32)

        rev = 0
        if np.size(bottommost) == 0 and np.size(topmost) == 0:
            return -np.ones(shape=(zz + 1)*2)
        elif np.size(topmost) == 0 or np.max(bottommost) > np.max(topmost):
            bottommost = np.rot90(bottommost, 2)
            cy, cx = np.unravel_index(np.argmax(bottommost), shape=bottommost.shape)
            img = np.rot90(img, 2)
            rev = 1
            coords[0] = h - 1 - cy, w - 1 - cx
        else:
            cy, cx = np.unravel_index(np.argmax(topmost), shape=topmost.shape)
            coords[0] = cy, cx

        for n in range(zz):
            i0 = int(n * segment_h)
            m = n + 1
            if m == zz:
                segment = img
            else:
                i1 = int(m * segment_h)
                segment = img[:i1]
            segment = dst_map(segment, cx, cy)
            segment = segment[i0:]
            cy, cx = np.unravel_index(np.argmax(segment), shape=segment.shape)

            coords[n + 1] = abs((h - 1) * rev - (cy + i0)), abs((w - 1) * rev - cx)
            cy += i0

        coords = np.array(coords, dtype=np.float64)
        # относительность
        # coords[:, 0] /= h
        # coords[:, 1] /= w

        return coords.flatten()

    @staticmethod
    def lbf_ftr(img):
        cy, cx = cntr(img)
        area = (cy + 1) * (cx + 1)
        f = lbf(img)

        lt = f[:cy, :cx].flatten()
        rt = f[:cy, cx:].flatten()
        lb = f[cy:, :cx].flatten()
        rb = f[cy:, cx:].flatten()
        r = np.array([
            np.bincount(lt, minlength=9),
            np.bincount(rt, minlength=9),
            np.bincount(lb, minlength=9),
            np.bincount(rb, minlength=9)
        ], dtype=np.float64)
        # относительность
        # r = r / area
        return r.flatten()

    @staticmethod
    def fill_ftr(img):
        img = np.array(img, dtype=np.bool)
        cy, cx = cntr(img)

        emptys = np.sum(~img)

        center = img[cy, cx]
        if not center:
            d = dst_map(img, cx, cy)
            d = d + (d == 0) * (d.max() + 1)
            # Чтобы метод был честнее, начало работы
            # алгоритма будет в самой близкой от центра ненулевой точке на изображении.
            cy, cx = np.unravel_index(np.argmin(d), shape=d.shape)

        linesx = [img[cy, cx:],
                  np.flip(img[cy, :cx])]
        linesy = [img[cy:, cx].flatten(),
                  np.flip(img[:cy, cx].flatten())]

        fill_points = []
        if linesx[0].size:
            fill_points.append([cy, np.argmin(linesx[0]) + cx])
        else:
            fill_points.append([-1, -1])

        if linesx[1].size:
            fill_points.append([cy, cx - 1 - np.argmin(linesx[1])])
        else:
            fill_points.append([-1, -1])

        if linesy[0].size:
            fill_points.append([np.argmin(linesy[0]) + cy, cx])
        else:
            fill_points.append([-1, -1])

        if linesy[1].size:
            fill_points.append([cy - 1 - np.argmin(linesy[1]), cx])
        else:
            fill_points.append([-1, -1])

        fill_points = np.array(fill_points, dtype=np.int64)

        flood_stats = np.zeros(shape=4, dtype=np.float64)
        intimg = np.array(img, dtype=np.int16)
        n = 0
        for fill_point in fill_points:
            if np.all(fill_point != (-1, -1)):
                seed = fill_point[0], fill_point[1]
                flooded = flood(intimg, seed)
                s = np.sum(flooded)
                if emptys != 0:
                    flood_stats[n] = s / emptys
                else:
                    print("Fill featuring: no empty pixels found. Seed point: (" +
                          str(fill_point[0]) + ", " + str(fill_point[1]) + ").")
                    flood_stats[n] = 2.
            else:
                flood_stats[n] = 2.
            n += 1

        return flood_stats

    @staticmethod
    def thickness_ftr(img):
        # находит наименее плотную часть изображения по х и у и рассчитывает их плотности
        y, x = img.shape
        s = np.sum(img)
        sy, sx = np.sum(img, 0), np.sum(img, 1)
        tx, ty = np.argmin(sy), np.argmin(sx)
        return [tx / x, np.sum(img[:, tx]) / s, ty / y, np.sum(img[ty, :]) / s]

    @staticmethod
    def sym_ftr(img):
        s = np.sum(img)

        xhalved = cut_in_half(img)
        yhalved = cut_in_half(img.transpose())

        xmerge = xhalved[0] * xmirror(xhalved[1])
        ymerge = yhalved[0] * ymirror(yhalved[1])

        # относительная симметричность изображения в двух осях
        # чем ближе к 1, тем объект симметричнее
        xms_prop = (np.sum(xmerge) / s * 2) ** 2
        yms_prop = (np.sum(ymerge) / s * 2) ** 2

        return xms_prop, yms_prop

    @staticmethod
    def zone_ftr(img, ccx=6, ccy=6):
        # north east south west
        """
        :param ccx: coefficient which decides the amount of pixels
                    central area is going to take up in a letter along the x axis
        :param ccy: <...> along the y axis
        :param img: source image
        :return: features of the source image
        """

        def feature_out(zone):
            if zone.size == 0:
                return 0, 0, 0, 0
            else:
                z_h, z_w = zone.shape

                zx, zy = np.unravel_index(np.argmax(zone), zone.shape)

                fval = zone[zx][zy] / (x * y)
                fzx = zx / z_w
                fzy = zy / z_h
                nonzeros = (zone != 0)
                # sum of all ones
                o = np.sum(nonzeros)
                sx, sy = zone.shape
                # all elements
                a = sx * sy
                # all zeros
                z = a - o

                return fval, fzx, fzy, o / (z + 1)

        x, y = img.shape

        cx = int(x / 2)
        cy = int(y / 2)
        map = dst_map(img, cx, cy)

        cdx = int(x / ccx)
        cdy = int(y / ccy)
        # четыре центра по две координаты

        # зоны
        features = np.empty(shape=[9, 4])

        n = map[(cx - cdx):(cx + cdx), 0:(cy - cdy)]
        s = map[(cx - cdx):(cx + cdx), (cy + cdy):y]
        w = map[0:(cx - cdx), (cy - cdy):(cy + cdy)]
        e = map[(cx + cdx):x, (cy - cdy):(cy + cdy)]

        features[0] = feature_out(n)
        features[1] = feature_out(s)
        features[2] = feature_out(w)
        features[3] = feature_out(e)

        nw = map[0:(cx - cdx), 0:(cy - cdy)]
        ne = map[(cx + cdx):x, 0:(cy - cdy)]
        sw = map[0:(cx - cdx), (cy - cdy):y]
        se = map[(cx + cdx):x, (cy - cdy):y]

        features[4] = feature_out(nw)
        features[5] = feature_out(ne)
        features[6] = feature_out(sw)
        features[7] = feature_out(se)

        cc = map[(cx - cdx):(cx + cdx), (cy - cdy):(cy + cdy)]
        features[8] = feature_out(cc)

        return features.ravel()

    def ftr(self, img):
        ftr_vector = np.empty(shape=self.FEATURE_VECTOR_LEN)
        s = 0
        for f in self.ftrs:
            res = f(img)
            e = len(res) + s
            ftr_vector[s:e] = res
            s = e
        return ftr_vector


class CardRecognizer:
    def __init__(self, ideal, empty,
                 delta=3, noise=0.03, ymargin=0.1, xmargin=0.2, empty_ratio=0.75,
                 x_train=None, ry_train=None, sy_train=None, sc=None, rc=None):
        """
        :param ideal: эталонная карта, которая будет служить примером для сравнения всех следующих
        :param empty: пустая карта
        :param info: каким маркером обозначена информация на картинке
        :param delta: параметр для функции waterfall_box
        :param xtrain: карты для тренировки распознавания достоинства карты и ее масти
        :param ytrain: достоинства и масти карт для тренировки
        """
        self.ideal = ideal
        self.empty = empty
        ideal_h, ideal_w = ideal.shape
        # ideal height / width proportion
        ideal_hw_prop = ideal_h / ideal_w

        symbols, xinds = waterfall_box(ideal, delta)
        symbols, yinds = waterfall_box(symbols[0].transpose(), delta)

        suit_bbox = (
            xinds[0][0],
            yinds[1][0],
            xinds[0][1],
            yinds[1][1]
        )
        rank_bbox = (
            xinds[0][0],
            yinds[0][0],
            xinds[0][1],
            yinds[0][1]
        )
        self.snr_sum = np.sum(symbols[0]) + np.sum(symbols[1])

        # ideal suit and rank proportional bounding boxes
        suit_pbbox = (suit_bbox[0] / ideal_w,
                      suit_bbox[1] / ideal_h,
                      suit_bbox[2] / ideal_w,
                      suit_bbox[3] / ideal_h)
        rank_pbbox = (rank_bbox[0] / ideal_w,
                      rank_bbox[1] / ideal_h,
                      rank_bbox[2] / ideal_w,
                      rank_bbox[3] / ideal_h)

        # suit and rank bounding box
        self.snr_bbox = (min(suit_bbox[0], rank_bbox[0]),
                         min(suit_bbox[1], rank_bbox[1]),
                         max(suit_bbox[2], rank_bbox[2]),
                         max(suit_bbox[3], rank_bbox[3]))

        # suit and rank box width and height
        self.snr_width = self.snr_bbox[2] - self.snr_bbox[0]
        self.snr_height = self.snr_bbox[3] - self.snr_bbox[1]

        # part of the card which must contain suit and rank
        self.container_w = int((self.snr_bbox[2] + self.snr_bbox[0]) * (xmargin + 1))
        self.container_h = int((self.snr_bbox[3] + self.snr_bbox[1]) * (ymargin + 1))
        self.container_pw = self.container_w / ideal_w
        self.container_ph = self.container_h / ideal_h

        ideal_trim, ideal_trim_bbox = trim(ideal)

        self.empty_ratio = empty_ratio
        self.w_trim, self.h_trim = ideal_trim_bbox[2] - ideal_trim_bbox[0], ideal_trim_bbox[3] - ideal_trim_bbox[1]
        self.noise = noise
        self.delta = delta
        self.h = ideal_h
        self.w = ideal_w
        self.hw_prop = ideal_hw_prop
        self.suit_bbox = suit_bbox
        self.rank_bbox = rank_bbox
        self.rank_pbbox = rank_pbbox
        self.suit_pbbox = suit_pbbox

        if rc is not None:
            self.rc = rc
        else:
            train_set_len = len(ry_train)
            rank_x_train = []
            suit_x_train = []
            for n in range(train_set_len):
                r = self.ranknsuit(x_train[n])
                rank, suit = r[0], r[1]
                rank_x_train.append(rank)
                suit_x_train.append(suit)
            self.sxt = suit_x_train
            self.syt = sy_train
            self.rc = DigitClassifier(rank_x_train, ry_train, features='all', z=3)

        if sc is not None:
            self.sc = sc
        else:
            if hasattr(self, 'sxt') and hasattr(self, 'syt'):
                self.sc = DigitClassifier(self.sxt, self.syt, features='sl')
            else:
                train_set_len = len(sy_train)
                suit_x_train = []
                for n in range(train_set_len):
                    r = self.ranknsuit(x_train[n])
                    rank, suit = r[0], r[1]
                    suit_x_train.append(suit)
                self.sc = DigitClassifier(suit_x_train, sy_train, 'sl')

    def recognize(self, card, return_rns=False):
        if self.is_empty(card)[0]:
            return "Nn"
        r = self.ranknsuit(card)
        rank, suit = r[0], r[1]
        if return_rns:
            return str(self.rc.classify(rank)[0]) + str(self.sc.classify(suit)[0]), rank, suit
        return str(self.rc.classify(rank)[0]) + str(self.sc.classify(suit)[0])

    def is_empty(self, card):
        ratio = shift_eq(self.empty, card)
        return ratio > self.empty_ratio, ratio

    def ranknsuit(self, card, only_top=True):
        ch, cw = card.shape
        ih, iw = self.ideal.shape
        ph, pw = ch / ih, cw / iw

        card_supposed_shape = keep_dims(self.ideal.shape, card.shape)
        container_x = int(card_supposed_shape[1] * self.container_pw)
        container_y = int(card_supposed_shape[0] * self.container_ph)

        # проверка максимального порога
        ptw = self.w_trim / self.w
        pth = self.h_trim / self.h
        if ph < pth or pw < ptw or only_top:
            # нижняя часть карты не проверяется, если порог пересечен
            rns = card[:container_y, :container_x]
            rns_bbox = (0, 0, container_x, container_y)
            label = 'a'
        else:
            # проверяются нижняя и верхняя части карт
            # выбирается основной та часть, которая наиболее похожа на произведение обеих (оператор "и")
            # и одновременно наиболее похожа на идеальный пример
            a, b = cut_in_half(card)
            a_rns = a[:container_y, :container_x]
            b_rns = b[:container_y, :container_x]

            def more_like_it(left, right):
                snr_btw_ratio = self.snr_sum / (self.container_h * self.container_w)
                h, w = right.shape
                area = h * w
                s_right = np.sum(right)
                s_left = np.sum(left)
                if s_left == 0:
                    return False
                elif s_right == 0:
                    return True

                left_btw_ratio = s_left / area
                right_btw_ratio = s_right / area

                left_btw_diff = 1 - abs(snr_btw_ratio - left_btw_ratio)
                right_btw_diff = 1 - abs(snr_btw_ratio - right_btw_ratio)

                if left_btw_diff > right_btw_diff:
                    return True
                return False

            if more_like_it(a_rns, b_rns):
                rns_bbox = (0, 0, container_x, container_y)
                rns = a_rns
                label = 'a'
            else:
                rns_bbox = (cw - container_x - 1, ch - container_y - 1, cw - 1, ch - 1)
                rns = b_rns
                label = 'b'
        # область изучения выбрана
        s = np.sum(rns, axis=1, dtype=np.uint32)
        ideal_ys = int((self.rank_bbox[3] + self.suit_bbox[1]) / 2)
        ys = int(ideal_ys / self.h * ch)
        s[:ys] += np.arange(ys - 1, -1, -1, dtype=np.uint32)
        s[ys:] += np.arange(0, len(s) - ys, 1, dtype=np.uint32)
        ys = np.argmin(s)
        rank = rns[:ys]
        suit = rns[ys:]

        # в последствии достоинство и масть очищаются от информационного мусора
        suit_h, suit_w = suit.shape
        smidp = int(suit_h / 2), int(suit_w / 2)
        # очистка масти карты
        if suit[smidp]:
            # объект в середине
            suit, sbbox = extract_obj(suit, smidp)
        else:
            # достать объект из самого длинного постоянного промежутка черной полосы
            gaps_y, gaps_x = gaps(suit)
            ysum = np.sum(suit, 0) * gaps_y
            xsum = np.sum(suit, 1) * gaps_x
            smaxp = np.argmax(xsum), np.argmax(ysum)
            suit, sbbox = extract_obj(suit, smaxp)

        def recalc_bbox(old, new):
            new = old[0] + new[0], old[1] + new[1], old[0] + new[2], old[1] + new[3]
            return new

        # очистка достоинства карты
        rank, rank_bbox = trim(~rank)
        rank = ~rank
        rank = clear_edges(rank, target_loss=0.2, b=2)
        rank, rb = trim(rank)
        rank_bbox = recalc_bbox(rank_bbox, rb)
        if rank.shape[0] != 0 and rank.shape[1] != 0:
            rank = clear_borders(rank)
            rank, rb = trim(rank)
            rank_bbox = recalc_bbox(rank_bbox, rb)
            rank, rb = percent_trim(rank, noise=self.noise)
            rank_bbox = recalc_bbox(rank_bbox, rb)

        region_black_pxs = np.sum(rank)

        if region_black_pxs > 0:
            max_ratio = 0
            rank_n = -1
            rank_img = None
            rank_xs = None

            n = 0
            rank_candidates = waterfall_box(rank, self.delta)
            for candidate_img, candidate_x_coords in zip(rank_candidates[0], rank_candidates[1]):
                candidate_black_pxs = np.sum(candidate_img)
                bl_pxs_ratio = candidate_black_pxs / region_black_pxs
                if bl_pxs_ratio > max_ratio:
                    rank_img = candidate_img
                    rank_xs = candidate_x_coords
                    max_ratio = bl_pxs_ratio
                    rank_n = n
                n += 1
            if rank_n == 0:
                rank_img, rank_trim_bbox = trim(rank_img, (False, False, True, True))
                new_bbox = rank_xs[0], rank_trim_bbox[1], rank_xs[1], rank_trim_bbox[3]
                rank_bbox = recalc_bbox(rank_bbox, new_bbox)
                rank = rank_img
            else:
                rank = rank[:, :rank_xs[1]]
                rank, rank_trim_bbox = trim(rank, (True, False, True, True))
                new_bbox = rank_trim_bbox[0], rank_trim_bbox[1], rank_xs[1], rank_trim_bbox[3]
                rank_bbox = recalc_bbox(rank_bbox, new_bbox)

        if label == 'a':
            suit_bbox = sbbox[0], sbbox[1] + ys, sbbox[2], sbbox[3] + ys
        else:
            rank_bbox = cw - rank_bbox[2] - 1, ch - rank_bbox[3] - 1, cw - rank_bbox[0] - 1, ch - rank_bbox[1] - 1
            suit_bbox = cw - sbbox[2] - 1, ch - sbbox[3] - ys, cw - sbbox[0] - 1, ch - sbbox[1] - ys

        return np.array(rank, dtype=np.bool), np.array(suit, dtype=np.bool), rank_bbox, suit_bbox, rns_bbox


def create_training_set(folder):
    files = listdir(folder)
    files = filter(lambda x: x.endswith(".png"), files)
    x = []
    y = []
    for img in files:
        print(img[0])
        y.append(img[0])
        img = np.array(Image.open(folder + img), dtype=np.bool)
        x.append(img)
    return x, y
