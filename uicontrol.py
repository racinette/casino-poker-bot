from scr.ocr import binarize_rgb, trim, clear_edges, clear_borders
import pyautogui
from random import randrange
import numpy as np
from scr.image import screenshot
from scr.game import date_time
from PIL import Image


CARD_PREFIX = "card:."
BUTTON_PREFIX = "btn:."
CHAR_LINE_PREFIX = "charli:."
DIGITS_LINE_PREFIX = "digitli:."

PLAYER_STACK_LINE = "digitli:.Player_stack"

OPPONENT_CARD_2 = "card:.Opponent_2"
OPPONENT_CARD_1 = "card:.Opponent_1"
HOLE_CARD_1 = "card:.Hole_1"
HOLE_CARD_2 = "card:.Hole_2"

BET_BUTTON = "btn:.Bet"
RAISE_BUTTON = "btn:.Raise"
CHECK_BUTTON = "btn:.Check"
FOLD_BUTTON = "btn:.Fold"
DEAL_HAND_BUTTON = "btn:.Deal_hand"
FOR_SURE_BUTTON = "btn:.For_sure"
CLEAR_STAKES_BUTTON = "btn:.Clear_stakes"
DEAL_STAKES_BUTTON = "btn:.Deal_stakes"
STAKES_SPINBOX_BUTTON = "btn:.Stakes_spinbox"
POINT01_CHIP_BUTTON = "btn:0.01_chip"
POINT10_CHIP_BUTTON = "btn:.0.10_chip"
STAKE_CHANGE_MENU_BUTTON = "btn:.Stake_change_menu"
PUT_CHIP_BUTTON = "btn:.Put_chip"

TABLE_CARD_1 = "card:.Table_1"
TABLE_CARD_2 = "card:.Table_2"
TABLE_CARD_3 = "card:.Table_3"
TABLE_CARD_4 = "card:.Table_4"
TABLE_CARD_5 = "card:.Table_5"


# Screen Element List
HEADS_UP_SEL = [HOLE_CARD_1, HOLE_CARD_2,
                TABLE_CARD_1, TABLE_CARD_2, TABLE_CARD_3, TABLE_CARD_4, TABLE_CARD_5,
                OPPONENT_CARD_1, OPPONENT_CARD_2,
                BET_BUTTON, RAISE_BUTTON, CHECK_BUTTON, FOLD_BUTTON]

CASINO_POKER_SEL = [HOLE_CARD_1, HOLE_CARD_2,
                    TABLE_CARD_1, TABLE_CARD_2, TABLE_CARD_3, TABLE_CARD_4, TABLE_CARD_5,
                    OPPONENT_CARD_1, OPPONENT_CARD_2,
                    PLAYER_STACK_LINE,
                    BET_BUTTON, FOLD_BUTTON, FOR_SURE_BUTTON, DEAL_HAND_BUTTON,
                    STAKE_CHANGE_MENU_BUTTON, STAKES_SPINBOX_BUTTON, DEAL_STAKES_BUTTON, CLEAR_STAKES_BUTTON,
                    PUT_CHIP_BUTTON, POINT01_CHIP_BUTTON, POINT10_CHIP_BUTTON
                    ]

SELs = {
    "totalcasino_pl": HEADS_UP_SEL,
    "casino-poker(0.01-0.10)": CASINO_POKER_SEL
}

def config_filter(img, config):
    r = config[0]
    g = config[1]
    b = config[2]
    inv = config[3]
    delta = config[4]
    ft = config[5]
    cc = config[6]
    cb = config[7]
    st = config[8]

    img = binarize_rgb(img, r, g, b)
    if inv:
        img = ~img
    if ft:
        img, b = trim(img)
    if cc:
        img = clear_edges(img)
    if cb:
        img = clear_borders(img)
    if st:
        img, b = trim(img)

    return img


def config_to_img(bbox, config, s=2):
    sbbox = [bbox[0] * s, bbox[1] * s, bbox[2] * s, bbox[3] * s]

    img = screenshot(sbbox)
    img = np.asarray(img)

    return config_filter(img, config)


class ViewTree:
    ROOT = 'root'

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def empty(root_bbox, keys, N=9):
        root = View(root_bbox, key='root')
        tree = ViewTree(root, N)
        for key in keys:
            tree.put(key)
        return tree

    def __init__(self, root, N=9):
        self.keys = []
        self.N = N
        self.root = root
        self.elems = {self.ROOT: root}

    def put(self, key, bbox=None, features=None, parent=None):
        if key != self.ROOT and key not in self.keys:
            self.keys.append(key)
            if parent is None:
                parent = self.root
            val = parent.add_child(bbox, features, key, self.N)
            self.elems[key] = val
            return val
        return None

    def __getitem__(self, item):
        return self.elems[self.keys[item]]

    def get(self, key):
        return self.elems[key]

    def remove(self, key):
        if key != self.ROOT and key in self.keys:
            self.keys.remove(key)
            r = self.elems.pop(key)
            if r.parent is not None:
                r.parent.children.remove(r)
            for child in r.children:
                child.parent = None
            return r
        return None

    def edit(self, key, bbox=None, features=None):
        if bbox is None and features is None:
            return
        e = self.elems[key]
        if features is not None:
            e.features = features
        if bbox is not None:
            e.set_bbox(bbox)

    def _copy(self, tree, new_elem, copied_elem):
        for child in copied_elem.children:
            n = tree.put(child.key, child.get_absolute_bbox(), child.get_features(), new_elem)
            self._copy(tree, n, child)

    def prefix_elements(self, prefix):
        keys_only = list(filter(lambda s: s.startswith(prefix), self.keys))
        nodes = []
        for key in keys_only:
            nodes.append(self.get(key))
        return nodes

    def copy(self):
        copy_root = self.root.essence()
        tree = ViewTree(copy_root, self.N)
        self._copy(tree, copy_root, self.root)
        return tree

    def _family_shot(self, rsc, parent, s=2):
        cbbox = parent.get_absolute_bbox()
        cbbox = [cbbox[0] - self.root.x1, cbbox[1] - self.root.y1, cbbox[2] - self.root.x1, cbbox[3] - self.root.y1]
        cbbox = [cbbox[0] * s, cbbox[1] * s, cbbox[2] * s, cbbox[3] * s]
        region = rsc[cbbox[1]:cbbox[3], cbbox[0]:cbbox[2]]
        photoshot = config_filter(region, parent.get_features())
        shots = [photoshot]
        for child in parent.children:
            if not child.is_zero_dim():
                shots.extend(self._family_shot(rsc, child))
        return shots

    def family_shot(self, s=2):
        shots = []
        cbbox = self.root.get_absolute_bbox()
        cbbox = [cbbox[0] * s, cbbox[1] * s, cbbox[2] * s, cbbox[3] * s]
        img = screenshot(cbbox)
        img.save("screeen.png")
        img = np.asarray(img)
        rsc = img
        for child in self.root.children:
            if not child.is_zero_dim():
                shots.extend(self._family_shot(rsc, child))
        return shots


class View:
    def __init__(self, bbox=None, features=None, key=None, N=9):
        if bbox is None:
            self.x1, self.y1, self.x2, self.y2 = [0, 0, 0, 0]
        else:
            self.x1, self.y1, self.x2, self.y2 = bbox
        if features is None:
            self.features = np.zeros(shape=N, dtype=np.int)
        else:
            self.features = features
        self.key = key
        self.children = []
        self.parent = None

    def get_features(self):
        return self.features

    def get_bbox(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def get_absolute_bbox(self):
        if self.parent is None:
            return [self.x1, self.y1, self.x2, self.y2]
        else:
            b = self.parent.get_absolute_bbox()
            return [self.x1 + b[0], self.y1 + b[1], self.x2 + b[0], self.y2 + b[1]]

    def get_height(self):
        return self.y2 - self.y1

    def get_width(self):
        return self.x2 - self.x1

    def add_child(self, bbox=None, features=None, key=None, N=9):
        pabs_bbox = self.get_absolute_bbox()
        if bbox is None:
            bbox = 0 - pabs_bbox[0], 0 - pabs_bbox[1], 0 - pabs_bbox[0], 0 - pabs_bbox[1]
        else:
            bbox = bbox[0] - pabs_bbox[0], bbox[1] - pabs_bbox[1], bbox[2] - pabs_bbox[0], bbox[3] - pabs_bbox[1]
        child = View(bbox, features, key, N)
        child.parent = self
        self.children.append(child)
        return child

    def matrix(self):
        N = len(self.children)
        m = np.empty(shape=(N + 1, 4), dtype=np.int32)
        m[0] = self.get_absolute_bbox()
        for n in range(1, N):
            m[n] = self.children[n].get_absolute_bbox()
        return m

    def shape(self):
        return self.get_height(), self.get_width()

    def set_features(self, f):
        self.features = f

    def _shift(self):
        if self.parent is None:
            return self.x1, self.y1
        else:
            x1, y1 = self.parent._shift()
            return self.x1 + x1, self.y1 + y1

    def _get_shift(self):
        if self.parent is None:
            return 0, 0
        else:
            return self.parent._shift()

    def set_bbox(self, new_bbox):
        shape = self.shape()
        new_shape = new_bbox[3] - new_bbox[1], new_bbox[2] - new_bbox[0]
        if new_shape[0] < 1 or new_shape[1] < 1:
            raise ValueError("New bounding box's shape is invalid (one of dimensions is less or equal to zero).\n" +
                             str(new_shape))
        if self.parent is None:
            self.x1, self.y1, self.x2, self.y2 = new_bbox
        else:
            dx, dy = self._get_shift()
            pshape = self.parent.shape()
            relative = new_bbox[0] - dx, new_bbox[1] - dy, new_bbox[2] - dx, new_bbox[3] - dy
            if relative[2] > pshape[1] or relative[3] > pshape[0] or relative[0] < 0 or relative[1] < 0:
                raise ValueError("New bounding box of the view is out of bounds of its parent's shape.\n" +
                                 "Parent's abs bounding box: " + str(self.parent.get_absolute_bbox()) + "\n" +
                                 "View's abs bounding box: " + str(new_bbox) + "\n" +
                                 "View's calculated relative bounding box: " + str(relative))

            self.x1, self.y1, self.x2, self.y2 = relative

        if shape[0] * shape[1] > 0:
            hr = new_shape[0] / shape[0]
            wr = new_shape[1] / shape[1]
            for child in self.children:
                child._resize(hr, wr)

    def _resize(self, hr, wr):
        self.x1 = int(self.x1 * wr)
        self.y1 = int(self.y1 * hr)
        self.x2 = int(self.x2 * wr)
        self.y2 = int(self.y2 * hr)
        for child in self.children:
            child._resize(hr, wr)

    def resize_bbox(self, newshape):
        shape = self.shape()
        if newshape != shape:
            self.x2 = self.x1 + newshape[1]
            self.y2 = self.y1 + newshape[0]
            if shape[0] * shape[1] > 0:
                hr = newshape[0] / shape[0]
                wr = newshape[1] / shape[1]
                for child in self.children:
                    child._resize(hr, wr)

    def _reloc(self, dx, dy):
        bbox = self.get_bbox()
        self.x1 = bbox[0] + dx
        self.x2 = bbox[1] + dx
        self.y1 = bbox[2] + dy
        self.y2 = bbox[3] + dy
        for child in self.children:
            child._reloc(dx, dy)

    def relocate(self, x, y):
        self.x2 = self.x2 + x - self.x1
        self.y2 = self.y2 + y - self.y1
        self.x1 = x
        self.y1 = y

    def get_key(self):
        return self.key

    def __str__(self):
        return self.key + " = (" + str(self.x1) + ", " + str(self.y1) + ", " + str(self.x2) + ", " + str(self.y2) + ")"

    def essence(self):
        e = View(self.get_absolute_bbox(), self.features, self.key)
        return e

    def is_default(self):
        return self.get_absolute_bbox() == [0, 0, 0, 0]

    def is_zero_dim(self):
        shape = self.shape()
        return shape[0] == 0 or shape[1] == 0

    def scrshot(self):
        if self.is_zero_dim():
            raise ValueError("Screenshot not taken: zero-dim bbox.")
        sshot = config_to_img(self.get_absolute_bbox(), self.get_features())
        return sshot


class UiProjection:
    """
    Этот класс нужен, чтобы соотнести дерево элементов экрана и запросы бота. Например, боту требуется получить карты
    игрока - бот делает запрос на возвращение функции hole().
    """
    def __init__(self, cr, slr, vt, savedir=None):
        self.tree = vt
        self.cr = cr
        self.slr = slr
        self.save = savedir is not None
        self.savedir = savedir

    def set_savedir(self, savedir=None):
        if savedir is None:
            self.savedir = None
            self.save = False
        else:
            self.savedir = savedir
            self.save = True

    def stack(self):
        s = self.view(PLAYER_STACK_LINE).scrshot()
        return float(self.slr.readline(s))

    def all(self, c=9):
        screens = self.tree.family_shot()[:c]

        cards = []
        for e in screens:
            if self.save:
                r, s = self.cr.ranknsuit(e)[:2]
                rank_class = str(self.cr.rc.classify(r)[0])
                suit_class = str(self.cr.sc.classify(s)[0])
                dt = date_time()
                rank = Image.fromarray(r)
                suit = Image.fromarray(s)
                rank_filename = rank_class + dt + ".png"
                suit_filename = suit_class + dt + ".png"
                suit.save(self.savedir + "suits/" + suit_filename)
                rank.save(self.savedir + "ranks/" + rank_filename)
                cards.append(rank_class + suit_class)
            else:
                c = self.cr.recognize(e)
                cards.append(c)
        return cards

    def _get_view_props(self, key):
        e = self.tree.get(key)
        return e.get_absolute_bbox(), e.get_features()

    def view(self, key):
        return self.tree.get(key)

    def recognize_elem(self, key):
        ss = self.view(key).scrshot()
        if self.save:
            r, s = self.cr.ranknsuit(ss)[:2]
            rank_class = str(self.cr.rc.classify(r)[0])
            suit_class = str(self.cr.sc.classify(s)[0])
            dt = date_time()
            rank = Image.fromarray(r)
            suit = Image.fromarray(s)
            rank_filename = rank_class + dt + ".png"
            suit_filename = suit_class + dt + ".png"
            suit.save(self.savedir + "suits/" + suit_filename)
            rank.save(self.savedir + "ranks/" + rank_filename)
            return rank_class + suit_class
        return self.cr.recognize(ss)

    def hole(self):
        return self.recognize_elem(HOLE_CARD_1), self.recognize_elem(HOLE_CARD_2)

    def flop(self):
        return self.recognize_elem(TABLE_CARD_1), self.recognize_elem(TABLE_CARD_2), self.recognize_elem(TABLE_CARD_3)

    def turn(self):
        return (self.recognize_elem(TABLE_CARD_1),
                self.recognize_elem(TABLE_CARD_2),
                self.recognize_elem(TABLE_CARD_3),
                self.recognize_elem(TABLE_CARD_4))

    def river(self):
        return (self.recognize_elem(TABLE_CARD_1),
                self.recognize_elem(TABLE_CARD_2),
                self.recognize_elem(TABLE_CARD_3),
                self.recognize_elem(TABLE_CARD_4),
                self.recognize_elem(TABLE_CARD_5))

    def only_river(self):
        return (self.recognize_elem(TABLE_CARD_4),
                self.recognize_elem(TABLE_CARD_5))

    def is_preflop(self):
        c1, c2 = self.hole()
        return c1[0] + c1[1] != "Nn" and c2[0] + c2[1] != "Nn", c1[0] + c1[1], c2[0] + c2[1]

    def is_flop(self):
        c1, c2, c3 = self.flop()
        return c1[0] + c1[1] != "Nn" and c2[0] + c2[1] != "Nn" and c3[0] + c3[1] != "Nn", \
               c1[0] + c1[1], c2[0] + c2[1], c3[0] + c3[1]

    def is_turn(self):
        c4 = self.recognize_elem(TABLE_CARD_4)
        c1, c2, c3 = self.is_flop()[1:]
        return self.is_flop() and c4[0] + c4[1] != "Nn", c1, c2, c3, c4[0] + c4[1]

    def is_river(self):
        c5 = self.recognize_elem(TABLE_CARD_5)
        c1, c2, c3, c4 = self.is_turn()[1:]
        return self.is_turn() and c5[0] + c5[1] != "Nn", c1, c2, c3, c4, c5[0] + c5[1]

    def is_opponent(self):
        c1, c2 = self.opponent_cards()
        return c1 != "Nn" and c2 != "Nn", c1, c2

    def opponent_cards(self):
        return self.recognize_elem(OPPONENT_CARD_1), self.recognize_elem(OPPONENT_CARD_2)

    def _press(self, key):
        bbox = self._get_view_props(key)[0]
        x, y = randrange(bbox[0], bbox[2]), randrange(bbox[1], bbox[3])
        print("click " + str(key))
        pyautogui.click(x, y)

    def bet(self):
        self._press(BET_BUTTON)

    def raise_(self):
        self._press(RAISE_BUTTON)

    def check(self):
        self._press(CHECK_BUTTON)

    def fold(self):
        self._press(FOLD_BUTTON)

    def deal(self):
        self._press(DEAL_HAND_BUTTON)

    def sure(self):
        self._press(FOR_SURE_BUTTON)

    def clear_stakes(self):
        self._press(CLEAR_STAKES_BUTTON)

    def open_stake_change_menu(self):
        self._press(STAKE_CHANGE_MENU_BUTTON)

    def choose_001_chip(self):
        self._press(POINT01_CHIP_BUTTON)

    def choose_010_chip(self):
        self._press(POINT10_CHIP_BUTTON)

    def deal_stakes(self):
        self._press(DEAL_STAKES_BUTTON)

    def put_chip(self):
        self._press(PUT_CHIP_BUTTON)

    def open_stakes_spinbox(self):
        self._press(STAKES_SPINBOX_BUTTON)




