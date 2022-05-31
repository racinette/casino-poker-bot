import numpy as np
from scr.ocr import screenshot
import scr.ocr as ocr
import matplotlib.pyplot as plt


class TableAnalyzer:
    def __init__(self, cconf, cr, s=2, belonging_mask=None):
        """
        :param cards_bboxes: bounding box'ы карт
        :param cr: CardRecognizer - распознавание карт
        :param belonging_mask: 0 - карты игрока, 1 - карты на столе, остальные - карты соперника(ов)
        """
        if belonging_mask is None:
            self.mask = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=np.int32)
        else:
            self.mask = np.array(belonging_mask, dtype=np.int32)
        self.conv_vals = cconf[:, :4]
        self.cards_bboxes = cconf[:, -4:]
        self.cr = cr
        self.s = s

    def get_groups(self):
        return np.unique(self.mask)

    def get_state(self, cmask=None):
        """
        :param cmask: какие карты проверить
        :return:
        """
        if cmask is None:
            cmask = np.ones(shape=7, dtype=np.bool)
        groups = self.get_groups()
        group_cards = {}
        for group in groups:
            group_cards[group] = ''

        cm_cards_bboxes, cm_conv_vals, cm_mask = self.cards_bboxes[cmask], self.conv_vals[cmask], self.mask[cmask]

        scrshot = screenshot()
        N = np.count_nonzero(cmask)
        for n in range(N):
            bbox, belong = cm_cards_bboxes[n], cm_mask[n]
            r, g, b, inv = cm_conv_vals[n]
            area = scrshot.crop(bbox * self.s)
            card = ocr.convert(area, r, g, b, inv)
            rank, suit = self.cr.recognize(card)
            card = rank[0] + suit[0]
            group_cards[belong] += card + " "
        return group_cards