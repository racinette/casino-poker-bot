import pickle
from scr.ocr import DigitClassifier, CardRecognizer, LineReader, create_training_set
import numpy as np
from PIL import Image
from scr.image import fill_corners
import tkinter as tk
from scr.gui import TableObserver
from scr.statistics.data.collection import StatsAggregator


def main(test_ocr=False):
    try:
        f = open("nn/rank.csf", mode="rb")
        rc = pickle.load(f)
        f.close()
    except FileNotFoundError:
        rX, rY = create_training_set("ytrain/ranks/")
        rc = DigitClassifier(rX, rY, features='all', pre_thinning=False, z=3)
        f = open("nn/rank.csf", mode="wb")
        pickle.dump(rc, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    try:
        f = open("nn/suit.csf", mode="rb")
        sc = pickle.load(f)
        f.close()
    except FileNotFoundError:
        sX, sY = create_training_set("ytrain/suits/")
        sc = DigitClassifier(sX, sY, features='sl', pre_thinning=False)
        f = open("nn/suit.csf", mode="wb")
        pickle.dump(sc, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    try:
        f = open("nn/stack.lrd", mode="rb")
        slr = pickle.load(f)
        f.close()
    except FileNotFoundError:
        stX, stY = create_training_set("digits/")
        slr = LineReader(delta=0, x_train=stX, y_train=stY)
        f = open("nn/stack.lrd", mode="wb")
        pickle.dump(slr, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    empty = np.array(Image.open("nn/empty.png"), dtype=np.bool)
    example = np.array(Image.open("nn/example.png"), dtype=np.bool)
    example = fill_corners(example, False)
    cr = CardRecognizer(example, empty, sc=sc, rc=rc)

    if test_ocr:
        x_test, y_test = create_training_set("training-source/ranks/")
        all = 0
        hit = 0
        miss = 0
        for x, y in zip(x_test, y_test):
            c = str(rc.classify(x)[0])
            print(c + " - " + y)
            if y == c:
                hit += 1
            else:
                miss += 1
            all += 1
        print(str(hit / all))
        print(str(miss / all))

    root = tk.Tk()
    sa = StatsAggregator()
    to = TableObserver(root, cr, slr, sa)
    to.pack()

    root.mainloop()


if __name__ == '__main__':
    main()