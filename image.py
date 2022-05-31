from PIL import Image
from PIL import ImageGrab
import numpy as np
from skimage.segmentation import flood_fill, flood
from skimage.transform import resize
from skimage.segmentation import clear_border
import platform
from itertools import combinations, chain
from scr.basicfunc import maxgrad
import cv2
from skimage.morphology import thin, skeletonize
from matplotlib import pyplot as plt
import mss


def cntr(a):
    y, x = a.shape
    return int(y / 2), int(x / 2)


def keep_dims(shape1, shape2):
    if shape1 != shape2:
        # height to width ratio
        h1, w1 = shape1
        h2, w2 = shape2

        hwr1 = h1 / w1
        hwr2 = h2 / w2
        if hwr1 != hwr2:
            # второе изображение измеряется относительно первого, поэтому сравнивается
            # схожесть hwr2 к hwr1, а не наоборот
            d = hwr1 - hwr2

            if d < 0:
                # если отношение длины к ширине второго изображения больше, чем первого,
                # то соотношение сторон должно быть поправлено увеличением ширины второго
                # изображения
                # принимаем длину за 100%
                # height1 / width1 = height2 / x
                # x = width1 * height2 / height1
                new_width = int(w1 * h2 / h1)
                shape = h2, new_width
            else:
                # принимаем ширину за 100%
                # height1 / width1 = x / width2
                # x = width1 * height2 / height1
                new_height = int(h1 * w2 / w1)
                shape = new_height, w2
            return shape
    return shape2


def bin_resize(img, newshape):
    return np.array(resize(
        img,
        output_shape=newshape,
        mode='edge',
        anti_aliasing=False,
        anti_aliasing_sigma=None,
        preserve_range=True,
        order=0
    ), dtype=np.bool)


def imshift(img, dx, dy):
    Y, X = img.shape
    dx, dy = dx % X, dy % Y
    new_img = img.copy()
    if dx > 0:
        new_img[:, dx:] = img[:, :X - dx]
        new_img[:, :dx] = img[:, X - dx:]
    elif dx < 0:
        dx = -dx
        new_img[:, -dx:] = img[:, :dx]
        new_img[:, :-dx] = img[:, dx:]
    if dy > 0:
        new_img[dy:, :] = img[:Y - dy, :]
        new_img[:dy, :] = img[Y - dy:, :]
    elif dy < 0:
        dy = -dy
        new_img[:, -dy:] = img[:, :dy]
        new_img[:, :-dy] = img[:, dy:]
    return new_img


def s_eq(im1, im2):
    a = im1 & im2
    return np.sum(a) / np.sum(im1)


def shift_eq(im1, im2, empty_fill_val=True, target=0.1):
    if im1.shape[0] * im1.shape[1] == 0 or im2.shape[0] * im2.shape[1] == 0:
        raise ValueError("One of the passed images is 0-shaped.\n" +
                         str(im1.shape) + ", " + str(im2.shape))
    if im1.shape != im2.shape:
        # height to width ratio
        h1, w1 = im1.shape
        h2, w2 = im2.shape

        hwr1 = im1.shape[0] / im1.shape[1]
        hwr2 = im2.shape[0] / im2.shape[1]
        if hwr1 != hwr2:
            # второе изображение измеряется относительно первого, поэтому сравнивается
            # схожесть hwr2 к hwr1, а не наоборот
            d = hwr1 - hwr2

            if d < 0:
                # если отношение длины к ширине второго изображения больше, чем первого,
                # то соотношение сторон должно быть поправлено увеличением ширины второго
                # изображения
                # принимаем длину за 100%
                # height1 / width1 = height2 / x
                # x = width1 * height2 / height1
                new_width = int(w1 * h2 / h1)
                if empty_fill_val:
                    t = np.ones(shape=(h2, new_width), dtype=np.bool)
                else:
                    t = np.zeros(shape=(h2, new_width), dtype=np.bool)
            else:
                # принимаем ширину за 100%
                # height1 / width1 = x / width2
                # x = width1 * height2 / height1
                new_height = int(h1 * w2 / w1)
                if empty_fill_val:
                    t = np.ones(shape=(new_height, w2), dtype=np.bool)
                else:
                    t = np.zeros(shape=(new_height, w2), dtype=np.bool)
            t[:h2, :w2] = im2
            im2 = t

    area1 = im1.shape[0] * im1.shape[1]
    area2 = im2.shape[0] * im2.shape[1]
    if area1 != area2:
        im2 = bin_resize(im2, im1.shape)

    def s_eq_xy(x, y):
        return s_eq(im1, imshift(im2, x, y))

    return maxgrad(s_eq_xy, 0, 0, 1, 1, target)[2]


def default_trim(img):
    img, bbox = trim(img)
    img = clear_edges(img)
    img = clear_borders(img)
    img, bbox = trim(img)
    return img


def multi_scale_template_match(template, img, from_=0.2, to=2, iters=20,
                               interpolation=cv2.INTER_CUBIC, comparison_method=cv2.TM_CCOEFF):
    img_height, img_width = img.shape
    height, width = template.shape

    resized_templates = []
    match_percents = np.zeros(shape=iters, dtype=np.float64)
    match_loc = np.zeros(shape=(iters, 4), dtype=np.int64)
    factors = np.linspace(from_, to, iters)

    n = 0
    for rescale_factor in factors:
        new_size = round(rescale_factor * width), round(rescale_factor * height)
        if new_size[1] > img_height or new_size[0] > img_width:
            break
        resized_template = cv2.resize(template, new_size, interpolation=interpolation)
        resized_templates.append(resized_template)
        res = cv2.matchTemplate(img, resized_template, method=comparison_method)

        best_match = np.unravel_index(np.argmax(res), res)
        match_loc[n, :2] = best_match
        match_loc[n, 2:] = np.add(best_match, new_size)

        match_percent = res[best_match]
        match_percents[n] = match_percent
        n += 1

    best_fit = np.argmax(match_percents)
    return match_loc[best_fit], resized_templates[best_fit], match_percents[best_fit]


def clear_edges(img, fval=False, target_loss=0.1, b=2):
    fval = int(fval)
    bin_img = img
    img = np.array(img, dtype=np.int8)
    s = np.sum(img)
    if s == 0:
        return img
    y, x = img.shape
    corners = [(0, 0, b, b), (y - b, 0, y, b), (0, x - b, b, x), (y - b, x - b, y, x)]
    filled = np.empty(shape=(4, y, x), dtype=np.bool)
    n = 0
    for c in corners:
        prep = np.copy(bin_img)
        prep[c[0]:c[2], c[1]:c[3]] = not fval
        yx = (c[0], c[1])
        filled[n] = flood_fill(img, yx, fval)
        n += 1

    closest_loss = 0
    n_closest = -1
    n = 0

    for st in POWERSET_0123:
        st = list(st)

        if len(st) < 2:
            loss_st = filled[st]
        else:
            loss_st = bin_img
            for layer in filled[st]:
                loss_st = loss_st & layer
        loss_st = 1 - np.sum(loss_st) / s

        if closest_loss < loss_st <= target_loss:
            closest_loss = loss_st
            n_closest = n
        n += 1
    if n_closest < 0:
        return img
    chosen_layers = filled[list(POWERSET_0123[n_closest])]
    if len(chosen_layers) > 1:
        result = bin_img
        for layer in chosen_layers:
            result = layer & result
        return result
    return chosen_layers[0]


def fill_corners(img, fval=False):
    fval = int(fval)
    img = np.array(img, dtype=np.uint8)
    y, x = img.shape
    corners = [(0, 0), (y - 1, 0), (0, x - 1), (y - 1, x - 1)]
    for corner in corners:
        img = flood_fill(img, corner, fval)
    return np.array(img, dtype=np.bool)


def cut_in_half(img):
    rows, columns = img.shape
    center = int(rows / 2)
    fhalf = img[:center]
    shalf = img[center:]

    if fhalf.shape != shalf.shape:
        fr = fhalf.shape[0]
        sr = shalf.shape[0]
        if fr < sr:
            shalf = shalf[:fr]
        else:
            fhalf = fhalf[:sr]
    return fhalf, np.rot90(shalf, k=2)


def mirror_map(img1, img2, N):
    if img1.shape != img2.shape:
        return None
    full = img1 & img2
    for n in range(1, N):
        cut1 = img1[n:]
        cut2 = img2[:-n]
        mirror = cut1 & cut2
        full[:-n] += mirror
    return full


def clear_borders(img, loss_threshold=0.2):
    if np.sum(img) == 0:
        return img

    y, x = img.shape
    masks = np.ones(shape=(4, y, x), dtype=np.bool)
    masks[0][:y, 0] = 0
    masks[1][:y, x-1] = 0
    masks[2][0, :x] = 0
    masks[3][y-1, :x] = 0
    i = np.array(img, dtype=np.int8)

    z = len(POWERSET_0123)
    products = np.zeros(shape=(z, y, x), dtype=np.bool)
    losses = np.zeros(shape=z, dtype=np.float64)
    max_loss = 0
    mln = -1
    n = 0

    for st in POWERSET_0123:
        st = list(st)
        chosen_masks = masks[st]
        for mask in chosen_masks:
            cleared = clear_border(i, bgval=0, mask=mask).astype(np.bool)
            products[n] = products[n] | (img & ~cleared)
        # invalid value??
        loss = np.sum(products[n]) / np.sum(img)
        losses[n] = loss
        if max_loss < loss < loss_threshold:
            max_loss = loss
            mln = n
        n += 1

    if mln > 0:
        return img & ~products[mln]
    return img


def extract_obj(img, point):
    """
    найти контуры объекта, внутри которого находится точка под координатами coords
    :param img: изображение
    :param point: точка, от которой начинается поиск
    :return:
    """
    elem = flood(np.array(img, np.uint16), point)
    return trim(elem)


def percent_trim(img, noise=0.05):
    h, w = img.shape
    ysum = np.sum(img, 1)
    xsum = np.sum(img, 0)
    cy, cx = cntr(img)

    y0 = cy - 1
    x0 = cx - 1
    y1 = cy + 1
    x1 = cx + 1
    for y in np.arange(y0, -1, -1):
        area = w * (y + 1)
        info = np.sum(ysum[:y])
        if info / area < noise:
            y0 = y
            break
    for y in np.arange(y1, h + 1, 1):
        area = w * (h - y)
        info = np.sum(ysum[y:])
        if y == h:
            y1 = h - 1
            break
        elif info / area < noise:
            y1 = y
            break
    for x in np.arange(x0, -1, -1):
        area = h * (x + 1)
        info = np.sum(xsum[:x])
        if info / area < noise:
            x0 = x
            break
    for x in np.arange(x1, w + 1, 1):
        area = h * (w - x)
        info = np.sum(xsum[x:])
        if x == w:
            x1 = w - 1
            break
        elif info / area < noise:
            x1 = x
            break

    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def gaps(a):
    a = np.array(a, dtype=np.int8)
    X = (a.transpose()[1:] - a.transpose()[:-1]) == 1
    Y = (a[1:] - a[:-1]) == 1
    return np.sum(Y, 0) < 2, np.sum(X, 0)[::-1] < 2


def matrix_filter(src, f, fin):
    filtered = np.copy(src)
    X, Y = src.shape
    fin_x, fin_y = fin
    for x in range(X - fin_x + 1):
        for y in range(Y - fin_y + 1):
            part = f(filtered[x:(x + fin_x), y:(y + fin_y)])
            filtered[x:(x + fin_x), y:(y + fin_y)] = part
    return filtered


def double_lbm(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 != h2 or w2 != w1:
        raise ValueError("Images don't have the same dimensions.")
    else:
        h = h1
        w = w2
    fl_shape = 8, h, w
    filter_layers = np.empty(fl_shape, dtype=np.int32)
    filter_layers[:] = img1

    layer = img2[1:, 1:]
    filter_layers[0, :h - 1, :w - 1] += layer

    layer = img2[1:, :]
    filter_layers[1, :h - 1, :] += layer

    layer = img2[1:, :-1]
    filter_layers[2, :h - 1, 1:] += layer

    layer = img2[:, :-1]
    filter_layers[3, :, 1:] += layer

    layer = img2[:-1, :-1]
    filter_layers[4, 1:, 1:] += layer

    layer = img2[:-1, :]
    filter_layers[5, 1:, :] += layer

    layer = img2[:-1, 1:]
    filter_layers[6, 1:, :-1] += layer

    layer = img2[:, 1:]
    filter_layers[7, :, :-1] += layer

    return np.sum(filter_layers, 0)


def local_binary_mask(img):
    h, w = img.shape
    fl_shape = 8, h, w
    filter_layers = np.zeros(fl_shape, dtype=np.bool)

    layer = img[1:, 1:]
    filter_layers[0, :h - 1, :w - 1] = layer

    layer = img[1:, :]
    filter_layers[1, :h - 1, :] = layer

    layer = img[1:, :-1]
    filter_layers[2, :h - 1, 1:] = layer

    layer = img[:, :-1]
    filter_layers[3, :, 1:] = layer

    layer = img[:-1, :-1]
    filter_layers[4, 1:, 1:] = layer

    layer = img[:-1, :]
    filter_layers[5, 1:, :] = layer

    layer = img[:-1, 1:]
    filter_layers[6, 1:, :-1] = layer

    layer = img[:, 1:]
    filter_layers[7, :, :-1] = layer

    return filter_layers


def local_binary_filter(img):
    masks = np.array(local_binary_mask(img), dtype=np.int16)
    for n in range(1, 8):
        masks[n] *= 2 ** n
    return np.sum(masks, 0)


def lbf(img):
    masks = np.array(local_binary_mask(img), dtype=np.int32)
    return np.sum(masks, 0)


def xmirror(img):
    return np.flipud(np.flip(img))


def ymirror(img):
    return np.flip(np.flipud(img))


def coldiv(img):
    y, x = img.shape
    had_ones = False
    conseq = False
    divs = []
    nonzero = np.count_nonzero(img, axis=0)
    for l in range(x):
        if nonzero[l] > 0:
            if had_ones and not conseq:
                divs.append(l - 1)
                conseq = True
            had_ones = True
        else:
            if had_ones and conseq:
                divs.append(l)
                had_ones = False
                conseq = False
    if had_ones:
        divs.append(x)
    cols = []
    for n in range(0, len(divs)-1, 2):
        cols.append(img[:, divs[n]:divs[n + 1]])
    return cols


def linediv(img):
    y, x = img.shape
    had_ones = False
    conseq = False
    divs = []
    nonzero = np.count_nonzero(img, axis=1)
    for l in range(y):
        if nonzero[l] > 0:
            if had_ones and not conseq:
                divs.append(l - 1)
                conseq = True
            had_ones = True
        else:
            if had_ones and conseq:
                divs.append(l)
                had_ones = False
                conseq = False
    if had_ones:
        divs.append(y)
    lines = []
    for n in range(0, len(divs)-1, 2):
        lines.append(img[divs[n]:divs[n+1]])
    return lines


def linediv_at(img):
    y, x = img.shape
    had_ones = False
    conseq = False
    divs = []
    nonzero = np.count_nonzero(img, axis=1)
    for l in range(y):
        if nonzero[l] > 0:
            if had_ones and not conseq:
                divs.append(l - 1)
                conseq = True
            had_ones = True
        else:
            if had_ones and conseq:
                divs.append(l)
                had_ones = False
                conseq = False
    if had_ones:
        divs.append(y)
    lines = []
    for n in range(0, len(divs)-1):
        if np.any(img[divs[n]:divs[n+1]] > 0):
            lines.append((divs[n], divs[n+1]))
    return lines


def waterfall_box(img, delta):
    """
    :param img: left to right written text as a binarized pillow image
    :param delta: maximum difference between pixel values in a set to be treated as one letter
    :return: letters separated
    """
    if np.amax(img) < 1:
        return [], []

    letters = []

    # rows instead of columns
    wtf = max_waterfall(img.transpose())

    groups = []

    for col in wtf:
        f = np.unique(col)
        groups.append(f)

    def prognose(a):
        if a == 0:
            return 0
        else:
            return a + 1

    letters_ind = [0]

    # prognosed group max
    pgmax = prognose(np.amax(groups[0]))

    # end-start index
    esi = 1
    for g in groups[1:]:
        gmax = g[-1]
        if gmax != pgmax:
            # прогнозы не сбылись и максимум исчез
            letters_ind.append(esi)
        pgmax = prognose(gmax)
        esi += 1
    letters_ind.append(wtf.shape[0] - 1)

    xinds = []

    for l in range(len(letters_ind) - 1):
        r = letters_ind[l]
        e = letters_ind[l + 1]

        rmax = np.amax(wtf[r])
        if rmax > 1:
            bp = rmax - 1
            m = r - 1
            # кусок буквы, который алгоритм откусил
            pieces = []
            # проверка правильности прогноза
            pcheck = wtf[m] == bp
            # пока прогноз верен, буква есть
            while np.any(pcheck) and m > 0 and bp > 0:
                pieces.append(pcheck * bp)
                bp -= 1
                m -= 1
                pcheck = wtf[m] == bp
            # обычный порядок
            pieces = pieces[::-1]
            # прикрепить оставшуюся часть буквы
            pieces.append(wtf[r:e])
            letter = np.vstack(pieces)
            letters.append(letter)
            if m != r - 1:
                m += 1
            xinds.append((m, e))
        else:
            letter = wtf[r:e]
            if np.amax(letter) > 0:
                letters.append(letter)
                xinds.append((r, e))

    i = 0
    for letter in letters:
        N, M = letter.shape
        for n in range(N - 1, -1, -1):
            piece = letter[n]
            lmax = np.amax(letter[n])

            if lmax <= delta:
                outscope = False
            else:
                outscope = (piece < (lmax - delta)) & (piece > 0)
            if np.any(outscope):
                piece = piece * ~outscope
                letter[n] = piece
            else:
                break
        letters[i] = letter.transpose() > 0
        i += 1
    return letters, xinds


def dst_map(img, cx, cy):
    # не охота переписывать метод из-за путаницы
    cx, cy = cy, cx

    X, Y = img.shape

    x_coords = np.zeros(shape=img.shape, dtype=np.int32)
    y_coords = np.zeros(shape=img.shape, dtype=np.int32)

    x_coords = (x_coords.transpose() + np.arange(X)).transpose()
    y_coords = y_coords + np.arange(Y)

    x_coords = ((cx - x_coords) / X) ** 2
    y_coords = ((cy - y_coords) / Y) ** 2

    return np.sqrt(x_coords + y_coords) * img


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


POWERSET_0123 = list(powerset([0, 1, 2, 3]))


def screenshot(bbox=None):
    # bbox - {left, top, right, bottom}
    if bbox is None:
        return np.array(mss.mss().grab(mss.mss().monitors[0]))
    else:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        monitor = {"top": bbox[1], "left": bbox[0], "width": width, "height": height}
        return np.array(mss.mss().grab(monitor))


def binarize_rgb(img, rt, gt, bt):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    Rt = R >= rt
    Gt = G >= gt
    Bt = B >= bt

    return Rt & Gt & Bt


def convert(img, r, g, b, inv):
    region = np.asarray(img)
    region = binarize_rgb(region, r, g, b)
    if inv:
        region = ~region
    return region


def binarize(img):
    return np.array(img.convert('1', dither=Image.NONE))


def trim(img, sides=(True, True, True, True)):
    # if meaningful information is True it means that
    # if maximum is False, then row/column is empty
    # and can be concatenated

    # first non-empty row and column counting from start
    srow, scol = 0, 0
    # last non-empty row and column counting from end
    erow, ecol = np.shape(img)

    left, right, up, down = sides

    if up or down:
        sumofr = np.sum(img, 1)
    if left or right:
        sumofc = np.sum(img, 0)

    if up:
        for rsum in sumofr:
            if rsum == 0:
                srow += 1
            else:
                break
    if left:
        for csum in sumofc:
            if csum == 0:
                scol += 1
            else:
                break
    if down:
        for rsum in sumofr[::-1]:
            if rsum == 0:
                erow -= 1
            else:
                break
    if right:
        for csum in sumofc[::-1]:
            if csum == 0:
                ecol -= 1
            else:
                break

    return img[srow:erow, scol:ecol], (scol, srow, ecol, erow)


def shift(xs, n, val):
    """
    :param xs: array to shift
    :param n: shift by n (positive right, negative left)
    :param val: value of new values spawned by shifting
    :return: shifted array
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = val
        e[n:] = xs[:-n]
    else:
        e[n:] = val
        e[:n] = xs[-n:]
    return e


def max_waterfall(img):
    filtered = np.empty_like(img, dtype=np.uint64)
    N, M = img.shape
    filtered[0] = img[0]

    def flood(lvl, at, d):
        val = lvl[at]
        if 0 <= at + d < lvl.shape[0]:
            if 0 < lvl[at + d] < val:
                lvl[at + d] = val
                return flood(lvl, at + d, d)
        return lvl

    for n in range(N - 1):
        n_lvl = filtered[n] * 1
        next_lvl = img[n + 1] * 1
        s = n_lvl + next_lvl
        s = s * next_lvl
        leaks = np.where(s > 1)[0]
        for leak in leaks:
            s = flood(s, leak, 1)
            s = flood(s, leak, -1)

        filtered[n + 1] = s
    return filtered


"""
img = np.array(Image.open("training-source/ranks/8-020200406185547.png"), dtype=np.bool)
img = skeletonize(img)
plt.imshow(max_waterfall(img))
plt.show()
"""