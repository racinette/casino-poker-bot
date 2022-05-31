from pynput import keyboard
from PIL import ImageGrab
from PIL import Image
import matplotlib.pyplot as plt
import pyautogui
from matplotlib.widgets import RectangleSelector
from functools import partial


class Zone:
    def __init__(self, x1, y1, x2, y2):
        x_left = x1
        y_up = y1
        x_right = x2
        y_down = y2
        if x2 < x_left:
            x_left = x1
            x_right = x2
        if y2 < y_up:
            y_up = y2
            y_down = y1

        self.xl = x_left
        self.yu = y_up
        self.xr = x_right
        self.yd = y_down

    def left_up(self):
        return self.xl, self.yu

    def right_down(self):
        return self.xr, self.yd

    def shift(self, x, y):
        self.xr += x
        self.xl += x
        self.yu += y
        self.yd += y

    def gshift(self, x, y):
        return Zone(self.xl + x, self.yu + y, self.xr + x, self.yd + y)

    # relative shift
    # get coordinates of this zone relative to z
    def rshift(self, z):
        return Zone(self.xl + z.xl, self.yu + z.yu, self.xr + z.xl, self.yd + z.yu)


class GUIElement:
    def __init__(self, zone=None, parent=None):
        self.parent = parent
        self.zone = zone

    def hook(self, image):
        region_alt(image, partial(hook_gui_element, gui_element=self, image=image, close='all'))

    def abszone(self):
        if self.parent is None:
            return self.zone
        else:
            return self.zone.rshift(self.parent.abszone())

    def state(self):
        screen = ImageGrab.grab()
        z = self.abszone()
        return screen.crop((z.xl, z.yu, z.xr, z.yd))


class PlayerWrp():
    def __init__(self, place, nick=None, bank=None):
        self.nick = nick
        self.bank = bank
        self.place = place


class PlaceGUI(GUIElement):
    def __init__(self, table, zone=None, player=None):
        super().__init__(zone, table)
        self.nick = GUIElement(parent=self)
        self.bank = GUIElement(parent=self)
        self.turn = GUIElement(parent=self)
        self.player = player


class TableGUI(GUIElement):
    def __init__(self, key, zone, np):
        super().__init__(zone)
        self.key = key
        self.np = np
        self.configured = False
        self.places = []


def region_alt(img, altf):
    plt.imshow(img)
    ax = plt.gca()
    rs = RectangleSelector(ax, altf, 'box')
    plt.show()


# click release crop
# crops by click and release points
def crcrop(click, release, image):
    z = Zone(click.xdata, click.ydata, release.xdata, release.ydata)
    img = image.crop((z.xl, z.yu, z.xr, z.yd))
    return z


# crcrop special for any object subclassing GUIElement
def hook_gui_element(click, release, image, gui_element, close):
    gui_element.zone = crcrop(click, release, image)
    plt.close(close)


def toint(s):
    try:
        i = int(s)
        return i
    except ValueError:
        return None


tables = []
command = ""


def crop(click, release, image):
    z = Zone(click.xdata, click.ydata, release.xdata, release.ydata)
    cropped_img = image.crop((z.xl, z.yu, z.xr, z.yd))
    plt.close('all')
    plt.imshow(cropped_img)
    plt.show()
    # ask for confirmation of result
    command = ""
    while command != "y" and command != "n":
        command = input("save? y/n \n")
        if command == "y":
            key = ""
            while key == "":
                key = input("key for the room: ")
            np = None
            while np is None:
                np = input("number of players: ")
                np = toint(np)
            table = TableGUI(key, z, np)
            tables.append(table)
        elif command == "n":
            plt.close('all')


print("'n' to add a room \n"
      "'c [key]' to configure a room \n"
      "'exit' to exit ")

while command != "exit":
    command = input("")
    # start by taking a screenshot of a room
    if command == "n":
        screen = ImageGrab.grab()
        plot = plt.imshow(screen)
        ax = plt.gca()
        rs = RectangleSelector(ax, partial(crop, image=screen), 'box')
        plt.show()
    elif command.startswith("c "):
        tname = command[2:]
        found = False
        # find table with the key
        for table in tables:
            if table.key == tname:
                found = True
                for i in range(table.np):
                    # selecting a place
                    print("place #" + str(i))
                    place = PlaceGUI(table)
                    place.hook(table.state())
                    # selecting player's nick
                    print("nick?")
                    place.nick.hook(place.state())
                    print("bank?")
                    place.bank.hook(place.state())
                    # print("turn indicator?")
                    # place.turn.hook(place.state())

                    table.places.append(place)
                print("configured")
                break
        if not found:
            print("table key " + tname + " not found")
    elif command == "test":
        for place in tables[0].places:
            n = place.nick.state()

    elif command == "1":
        i = Image.open("testimg.png")
        i = i.convert('1', dither=Image.NONE)
        plt.imshow(i)
        plt.show()
    elif command == "2":
        i = Image.open(r"pic/subj.png")
        i = i.convert('1', dither=Image.NONE)
        i.save("ttt.png")