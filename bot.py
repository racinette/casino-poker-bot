import pyautogui
import time
from scr.game import Room
import PIL.ImageFilter
import numpy as np


class Config:
    def __init__(self, bb, room_size, money_type, game):
        self.bb = bb
        self.room_size = room_size
        self.money_type = money_type
        self.game = game


class ResolutionException(Exception):
    pass


def threshold(img, t):
    img = img.convert("L")

    def black_n_white(p):
        if p < t:
            return 0
        else:
            return 255

    return img.point(black_n_white)


def read_chat(img):
    width, height = img.size
    bnw = threshold(img, 170)

    # locate scroll bar
    ar = np.array(bnw)
    flip = ar[:, ::-1]

    black = 0
    white = 255

    prev = white
    crop_up_to = 0

    for column in flip.transpose():
        if column.min() == black:
            prev = black
        else:
            if prev == black:
                break
            prev = white
        crop_up_to += 1

    crop_width = width - crop_up_to
    
    # crop scroll bar
    bnw = bnw.crop((0, 0, crop_width, height))
    bnw.show()


# resolution check
resolution = pyautogui.size()
full_hd = (1920, 1080)

holdem = "No Limit Hold'em"

play_money = "Play Money"
real_money = "USD"

waiting_time = 7

latest_config = None

scroll_bar_width = 18

play_money_stakes = {
                    100: "50/100", 200: "100/200", 500: "250/500", 2000: "1000/2000",
                    10000: "5000/10000", 50000: "25000/50000", 200000: "100000/200000",
                    500000: "250000/500000", 1000000: "500000/1000000", 2000000: "1000000/2000000"
                    }
real_money_stakes = {
                    0.02: "$0.01/$0.02", 0.05: "$0.02/$0.05",
                    0.10: "$0.05/$0.10", 0.25: "$0.10/$0.25",
                    0.50: "$0.25/$0.50"
                    }


def locate_buttons(resolution):
    if resolution == full_hd:
        poker = (352, 42)
        quick_seat = (68, 107)
        cash_games = (197, 147)
        stakes = (1070, 558)
        play_money = (1688, 145)
        real_money = (1558, 145)
        play_money_stakes = {100: (1100, 613), 200: (1100, 639), 500: (1100, 665), 2000: (1100, 691),
                             10000: (1100, 717), 50000: (1100, 743), 200000: (1100, 769), 500000: (1100, 795),
                             1000000: (1100, 821), 2000000: (1100, 847)}
        real_money_stakes = {0.02: (1100, 613), 0.05: (1100, 639), 0.10: (1100, 665), 0.25: (1100, 691),
                             0.50: (1100, 717)}
        max_players = (1065, 602)
        max_players_list = {9: (1076, 625), 6: (1076, 652), 2: (1076, 679)}
        play_now = (1077, 669)
        observe = (1077, 737)

        return (
            poker, quick_seat, cash_games, stakes, play_money,
            real_money, play_money_stakes, real_money_stakes,
            max_players, max_players_list, play_now, observe
        )

    raise ResolutionException("The program doesn't support your screen resolution.")


def click_button(location, movement_dur=0.5):
    x, y = location
    pyautogui.click(x, y, duration=movement_dur)


try:
    buttons = locate_buttons(resolution)
    (
        poker_btn, quick_seat_btn, cash_games_btn, stakes_btn,
        play_money_btn, real_money_btn,
        play_money_stakes_btn, real_money_stakes_btn, max_players_btn,
        max_players, play_now_btn, observe_btn
    ) = buttons

    def configure(bb, players, money_type, game):
        click_button(poker_btn)
        click_button(quick_seat_btn)
        click_button(cash_games_btn)
        if money_type == real_money:
            click_button(real_money_btn)
        else:
            click_button(play_money_btn)

        click_button(stakes_btn)

        if money_type == real_money:
            click_button(real_money_stakes_btn[bb])
        else:
            click_button(play_money_stakes_btn[bb])
        click_button(max_players_btn)
        click_button(max_players[players])

        return Config(bb, players, money_type, game)

    process_id = int(input("Enter PokerStars.exe process id: "))

    app = pywinauto.Application(backend="uia").connect(process=process_id)
    lobby = app['PokerStars Lobby']

    lobby.set_focus()
    lobby.maximize()
    latest_config = configure(bb=100, players=6, money_type=play_money, game=holdem)
    click_button(observe_btn)


    def find_rooms(config):
        bb = config.bb
        money_type = config.money_type
        game = config.game
        room_size = config.room_size

        rooms = []
        stakes = play_money_stakes[bb] if money_type == play_money else real_money_stakes[bb]
        regex = " - " + stakes + " " + money_type + " - " + game
        windows = app.windows()
        for window in windows:
            title = window.element_info.name
            if regex in title:
                name = title
                room = Room(window, name, room_size, money_type, bb, game)
                rooms.append(room)
        return rooms


    # 12956
    # wait till a room loads
    time.sleep(waiting_time)
    rooms = find_rooms(latest_config)
    print(rooms)

    for room in rooms:
        room.window.set_focus()

        room.chat_editor.type_keys("NIGGER!")

        read_chat(room.chat.capture_as_image())

        time.sleep(1)


except ResolutionException:
    pass
except pywinauto.application.ProcessNotFoundError:
    pass



# 13988
# chips area: (1669, 96); size: (103, 20)
# poker btn : (352, 42)
# quick seat: (68, 107)
# cash games: (197, 147)
# stakes    : (952, 548), size: (234, 25)
# play money: (1688, 145)
# real money: (1558, 145)
# drop-down : (950, 601)
# 26 pixels/item
# 50/100    : (1100, 613)
# 100/200   : (1100, 639)
# 250/500   : (1100, 665)
#          ...