import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scr.ocr import binarize_rgb
from scr.ocr import waterfall_box
import numpy as np
from datetime import datetime
from os import listdir
from scr.image import trim
from scr.uicontrol import ViewTree
import pickle
from scr.uicontrol import SELs, CARD_PREFIX, DIGITS_LINE_PREFIX
from scr.uicontrol import UiProjection
from scr.logic.sequential.casinopoker import CasinoPokerLogic
from scr.logic.communication import Messages, Outcomes, MessageListener, TaskExecutor
from scr.poker.equity import comb2str

EV_PREFIX = "EV = "
EQUITY_PLACEHOLDER = "        "
CALC_EQUITY = "Calc Equity"
TEST = "Table State"
CLEAR_CONSOLE = "Очистить\nконсоль"
NOT_CONFIGURED = "Не выбрана конфигурация стола.\n"
CHOOSE_ELEMENT = "Выберите элемент \nдля редактирования"
INVERSE = "Инверсия цвета"
REFRESH = "Обновлений в минуту"
DELTA = "Дельта"
SAVE_OCR = "Сохр.\nOCR"
SHOW_DIV = "Границы раздела"
SAVE_SETTINGS = "Сохранить"
DISCARD_CHANGES = "Сбросить"
CHOOSE_REGION = "Выбрать регион"
TAKE_SCREENSHOT = "Снимок текущего\nсостояния"
SHOOT_ALL_CARDS = "Снять все карты"
SHOOT_ALL_DIGITS = "Снять все цифровые\nполя"
CHANGES_SAVED = "Сохранено!"
CREATE_PRESET = "Создать новый"
DELETE_PRESET = "Удалить выбранный"
START = "Запуск"
STOP = "Стоп"
CONFIGURE = "Настройки"
CHOOSE_PRESET = "Загрузить готовые\nнастройки"
PRESET_NAME = "Название настройки"
SAVE_AND_EXIT = "Сохранить и выйти"
ALREADY_RUNNING = "Бот уже запущен.\n"
SETTINGS_SUCCESSFULLY_APPLIED = "Настройки успешно применены. \n"
PROBLEM_OCCURED = "Возникла проблема с настройками. \n"
STARTED = "Включение. \n"
STOPPED = "Выключение... \n"
EMPTY = "<Empty>"
SHOW_OCR_RESULT = "Распознать"
SET_EXAMPLE = "Пример"
REVERSED_SIDE = "Рубашка"
MAIN_BOUNDING_REGION = "Обозначить игровое\nокно"
PLAYER = "Игрок"
OPPONENT = "Против"
TABLE = "Стол"

LOSS_PREC = "L"
CLOSE_LOSS_PREC = "CL"
DRAW_PREC = "D"
CLOSE_WIN_PREC = "CW"
WIN_PREC = "W"
DNQUAL_PREC = "NQ"

LOSS_COLOR = '#ff0000'
COMB_LOSS_COLOR = '#ffb401'
DRAW_COLOR = '#fdff01'
COMB_WIN_COLOR = '#a3ff01'
WIN_COLOR = '#01ff1f'
DEALER_DOESNT_QUALIFY_COLOR = '#0000ff'

EQUITY_COLORS = [LOSS_COLOR, COMB_LOSS_COLOR, DRAW_COLOR, COMB_WIN_COLOR, WIN_COLOR, DEALER_DOESNT_QUALIFY_COLOR]

output = "elementshots"

LOG_FILE_NAME = "logs/log"

PRESET_FOLDER = "presets"
PATH_TO_PRESET_USED = PRESET_FOLDER + "/used.uip"
BOT_GAME_SETTINGS_EXT = ".bgs"


class FigureOptions:
    DONT_SHOW = 'Не показывать'
    OUTCOME_IN_CASE_OF_COMB = 'P(исход)|комбинация'
    ALL = [DONT_SHOW, OUTCOME_IN_CASE_OF_COMB]


def equity_to_bar(ax, equity, y, h=1):
    if len(equity) == 5:
        ax.broken_barh([(0, equity[0]),
                        (equity[0], equity[1]),
                        (equity[0] + equity[1], equity[2]),
                        (equity[0] + equity[1] + equity[2], equity[3]),
                        (equity[0] + equity[1] + equity[2] + equity[3], equity[4])
                        ],
                       [y, h],
                       facecolors=EQUITY_COLORS[:5])
    elif len(equity) == 6:
        ax.broken_barh([(0, equity[0]),
                        (equity[0], equity[1]),
                        (equity[0] + equity[1], equity[2]),
                        (equity[0] + equity[1] + equity[2], equity[3]),
                        (equity[0] + equity[1] + equity[2] + equity[3], equity[4]),
                        (equity[0] + equity[1] + equity[2] + equity[3] + equity[4], equity[5])
                        ],
                       [y, h],
                       facecolors=EQUITY_COLORS)
    else:
        raise ValueError("Equity data doesn't fit. len(equity) == " + str(len(equity)))


def get_preset_used():
    """
    :return: возвращает настройку, используемую в данный момент
    """
    try:
        cp_file = open(PATH_TO_PRESET_USED, mode='rb')
        cp = pickle.load(cp_file)
        cp_file.close()
        return cp
    except FileNotFoundError:
        print("No file under name 'used.uip' was found. Try choosing a preset before opening it.")
        return None


class TableObserver(tk.Frame, MessageListener, TaskExecutor):
    def __init__(self, root, cr, slr, sa, cconf=None):
        """
        :param root:
        :param cr: Card Recognizer
        :param slr: Stack Line Reader
        :param cconf: Current Config
        """
        tk.Frame.__init__(self, root)
        MessageListener.__init__(self)
        TaskExecutor.__init__(self)

        if cconf is None:
            self.cconf = get_preset_used()

        self.sa = sa
        self.slr = slr
        self.cr = cr
        self.cconf = cconf
        self.root = root

        self.bot_comb = None
        self.opp_comb = None

        self.fold_count = 0
        self.lose_count = 0
        self.draw_count = 0
        self.win_count = 0
        self.nq_count = 0

        self.bot_running = False

        games = SELs.keys()
        self.game_var = tk.StringVar()
        self.game_var.set(" ")
        game_option_menu = tk.OptionMenu(root, self.game_var, *games)
        game_option_menu.pack(fill=tk.X)

        self.configured = False

        cards_state_frame = tk.Frame(root)

        bot_cards_frame = tk.Frame(cards_state_frame)
        bot_cards_label = tk.Label(bot_cards_frame)
        bot_cards_label.configure(text=PLAYER)
        self.bot_cards_text = tk.Text(bot_cards_frame, height=1, width=7)
        bot_cards_label.pack(fill=tk.X, pady=1)
        self.bot_cards_text.pack(fill=tk.X, pady=1)
        bot_cards_frame.pack(fill=tk.X, side=tk.LEFT, padx=2)

        table_cards_frame = tk.Frame(cards_state_frame)
        table_cards_label = tk.Label(table_cards_frame)
        table_cards_label.configure(text=TABLE)
        self.table_cards_text = tk.Text(table_cards_frame, height=1, width=23)
        table_cards_label.pack(fill=tk.X, pady=1)
        self.table_cards_text.pack(fill=tk.X, pady=1)
        table_cards_frame.pack(fill=tk.X, side=tk.LEFT, padx=2)

        opp_cards_frame = tk.Frame(cards_state_frame)
        opp_cards_label = tk.Label(opp_cards_frame)
        opp_cards_label.configure(text=OPPONENT)
        self.opp_cards_text = tk.Text(opp_cards_frame, height=1, width=7)
        opp_cards_label.pack(fill=tk.X, pady=1)
        self.opp_cards_text.pack(fill=tk.X, pady=1)
        opp_cards_frame.pack(fill=tk.X, side=tk.LEFT,  padx=2)

        cards_state_frame.pack(fill=tk.X, pady=5)

        matplotlib.use("TkAgg")
        equity_frame = tk.Frame(root)

        equity_array_frame = tk.Frame(equity_frame)

        eqt_loss_frame = tk.Frame(equity_array_frame)
        loss_text = tk.Label(eqt_loss_frame)
        loss_text.configure(text=LOSS_PREC)
        loss_text.pack()
        self.loss_num_text = tk.Label(eqt_loss_frame, background=LOSS_COLOR, text=EQUITY_PLACEHOLDER, foreground='white')
        self.loss_num_text.pack()
        eqt_loss_frame.pack(side=tk.LEFT)

        eqt_close_loss_frame = tk.Frame(equity_array_frame)
        close_loss_text = tk.Label(eqt_close_loss_frame, text=CLOSE_LOSS_PREC)
        close_loss_text.pack()
        self.close_loss_num_text = tk.Label(eqt_close_loss_frame, background=COMB_LOSS_COLOR, text=EQUITY_PLACEHOLDER, foreground='white')
        self.close_loss_num_text.pack()
        eqt_close_loss_frame.pack(side=tk.LEFT)

        eqt_draw_frame = tk.Frame(equity_array_frame)
        draw_text = tk.Label(eqt_draw_frame, text=DRAW_PREC)
        draw_text.pack()
        self.draw_num_text = tk.Label(eqt_draw_frame, background=DRAW_COLOR, text=EQUITY_PLACEHOLDER, foreground='gray50')
        self.draw_num_text.pack()
        eqt_draw_frame.pack(side=tk.LEFT)

        eqt_close_win_frame = tk.Frame(equity_array_frame)
        close_win_text = tk.Label(eqt_close_win_frame, text=CLOSE_WIN_PREC)
        close_win_text.pack()
        self.close_win_num_text = tk.Label(eqt_close_win_frame, background=COMB_WIN_COLOR, text=EQUITY_PLACEHOLDER, foreground='black')
        self.close_win_num_text.pack()
        eqt_close_win_frame.pack(side=tk.LEFT)

        eqt_win_frame = tk.Frame(equity_array_frame)
        win_text = tk.Label(eqt_win_frame, text=WIN_PREC)
        win_text.pack()
        self.win_num_text = tk.Label(eqt_win_frame, background=WIN_COLOR, text=EQUITY_PLACEHOLDER, foreground='black')
        self.win_num_text.pack()
        eqt_win_frame.pack(side=tk.LEFT)

        eqt_dnqual_frame = tk.Frame(equity_array_frame)
        dnqual_text = tk.Label(eqt_dnqual_frame, text=DNQUAL_PREC)
        dnqual_text.pack()
        self.dnqual_num_text = tk.Label(eqt_dnqual_frame, background=DEALER_DOESNT_QUALIFY_COLOR, text=EQUITY_PLACEHOLDER,
                                   foreground='white')
        self.dnqual_num_text.pack()
        eqt_dnqual_frame.pack(side=tk.LEFT)

        self.ev_label = tk.Label(equity_array_frame, text=EV_PREFIX)
        self.ev_label.pack(side=tk.LEFT)

        equity_array_frame.pack(fill=tk.BOTH, expand=1)

        # FIGURE!!
        self.equity_figure = plt.figure(figsize=[3, 0.3])
        self.equity_canvas = FigureCanvasTkAgg(self.equity_figure, equity_frame)
        self.equity_ax = self.equity_figure.add_subplot(111)

        self.equity_canvas.get_tk_widget().pack(fill=tk.BOTH, pady=3, expand=1)
        self.configure_equity_figure([0.2, 0.2, 0.2, 0.2, 0.2])

        equity_frame.pack(fill=tk.BOTH)

        # рамка графика статистики бота
        self.stats_frame = tk.Frame(equity_frame)

        # выбор графика статистики бота
        self.figure_option_var = tk.StringVar()
        self.figure_option_menu = tk.OptionMenu(equity_frame,
                                                self.figure_option_var,
                                                *FigureOptions.ALL,
                                                command=self.figure_change)
        self.figure_option_var.set(FigureOptions.OUTCOME_IN_CASE_OF_COMB)
        self.figure_option_menu.pack(fill=tk.X, pady=3)

        # возможные графики
        self.comb_equity_figure = plt.figure(figsize=[3, 3])
        self.comb_equity_canvas = FigureCanvasTkAgg(self.comb_equity_figure, self.stats_frame)
        self.comb_equity_canvas.get_tk_widget().pack()
        self.stats_frame.pack()

        self.comb_equity_ax = self.comb_equity_figure.add_subplot(111)
        self.configure_comb_equity_figure(np.ones(shape=[10, 5], dtype=np.float) * 0.2, np.zeros(shape=10, dtype=np.float))

        comb_text_frame = tk.Frame(root)
        self.my_comb_text = tk.Text(comb_text_frame, height=3, width=15)
        self.opp_comb_text = tk.Text(comb_text_frame, height=3, width=15)
        self.my_comb_text.pack(side=tk.LEFT)
        self.opp_comb_text.pack(side=tk.LEFT)
        comb_text_frame.pack()

        self.console = tk.Text(root, height=5, width=20)
        self.console.pack(fill=tk.BOTH)

        button_frame = tk.Frame(root)
        self.start_button = tk.Button(button_frame, text=START, command=self.start)
        self.configure_button = tk.Button(button_frame, text=CONFIGURE, command=self.uip_config)
        self.clear_console_button = tk.Button(button_frame, text=CLEAR_CONSOLE, command=self.clear_console)

        integral_logic_buttons_frame = tk.Frame(button_frame)
        self.calc_equity_button = tk.Button(integral_logic_buttons_frame, text=CALC_EQUITY, command=self.calc_equity)
        self.calc_equity_button.pack(fill=tk.X)

        self.test_button = tk.Button(integral_logic_buttons_frame, text=TEST, command=self.test)
        self.test_button.pack(fill=tk.X)

        self.save_ocr_var = tk.BooleanVar()
        self.ocr_log_checkbtn = tk.Checkbutton(integral_logic_buttons_frame,
                                          text=SAVE_OCR, variable=self.save_ocr_var, command=self.save_ocr_logs)
        self.ocr_log_checkbtn.pack(fill=tk.X)

        integral_logic_buttons_frame.pack(side=tk.RIGHT, fill=tk.X, padx=5)

        self.stop_button = tk.Button(button_frame, text=STOP, command=self.stop)

        self.smt_button = tk.Button(button_frame, text='0', command=self.some_test)
        self.smt_button.pack(side=tk.LEFT, fill=tk.X, padx=0)

        self.start_button.pack(side=tk.LEFT, fill=tk.X, padx=5)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, padx=5)
        self.clear_console_button.pack(side=tk.RIGHT, padx=5)

        self.configure_button.pack(side=tk.RIGHT, fill=tk.X, padx=5)

        button_frame.pack(fill=tk.X)

    def figure_change(self, a):
        print(a)
        if self.figure_option_var.get() == FigureOptions.DONT_SHOW:
            self.stats_frame.pack_forget()
        else:
            self.stats_frame.pack()

    def configure_equity_figure(self, equity):
        if self.equity_figure is not None:
            equity_to_bar(self.equity_ax, equity, 0, 1)
            self.equity_ax.axes.yaxis.set_visible(False)
            self.equity_ax.axes.xaxis.set_visible(True)

            self.equity_ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            self.equity_ax.tick_params('x', direction='inout', labelsize='xx-small')

            self.equity_ax.xaxis.grid(True)

    def configure_comb_equity_figure(self, equity, histogram):
        if self.comb_equity_figure is not None:
            self.comb_equity_ax.axes.yaxis.set_visible(True)
            self.comb_equity_ax.axes.xaxis.set_visible(True)

            self.comb_equity_ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            self.comb_equity_ax.set_yticklabels(["",
                                                 "\n\nHC \n(" + '{:.3f}'.format(histogram[0]) + ")",
                                                 "\n\nOP \n(" + '{:.3f}'.format(histogram[1]) + ")",
                                                 "\n\nTP \n(" + '{:.3f}'.format(histogram[2]) + ")",
                                                 "\n\nSE \n(" + '{:.3f}'.format(histogram[3]) + ")",
                                                 "\n\nST \n(" + '{:.3f}'.format(histogram[4]) + ")",
                                                 "\n\nFL \n(" + '{:.3f}'.format(histogram[5]) + ")",
                                                 "\n\nFH \n(" + '{:.3f}'.format(histogram[6]) + ")",
                                                 "\n\nQU \n(" + '{:.3f}'.format(histogram[7]) + ")",
                                                 "\n\nSF \n(" + '{:.3f}'.format(histogram[8]) + ")",
                                                 "\n\nRF \n(" + '{:.3f}'.format(histogram[9]) + ")"
                                                 ]
                                                )
            self.comb_equity_ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            self.comb_equity_ax.tick_params('both', direction='inout', labelsize='xx-small')

            self.comb_equity_ax.xaxis.grid(True)
            self.comb_equity_ax.yaxis.grid(True, color='black')

            for n in range(10):
                equity_to_bar(self.comb_equity_ax, equity[n], n, 1)

    def clear_ui(self):
        self.bot_cards_text.delete(1.0, tk.END)
        self.table_cards_text.delete(1.0, tk.END)
        self.opp_cards_text.delete(1.0, tk.END)

        self.loss_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.close_loss_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.draw_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.close_win_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.win_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.dnqual_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.ev_label.configure(text=EV_PREFIX)

        self.my_comb_text.delete(1.0, tk.END)
        self.opp_comb_text.delete(1.0, tk.END)
        self.bot_comb = None
        self.opp_comb = None

        self.equity_ax.clear()
        self.equity_canvas.draw()
        self.comb_equity_ax.clear()
        self.comb_equity_canvas.draw()

    def save_ocr_logs(self):
        b = self.save_ocr_var.get()
        self.logic.set_save_ocr_log(b)

    def clear_console(self):
        self.console.delete(1.0, tk.END)

    def console_insert(self, t):
        self.console.insert(tk.END, t + "\n")
        self.console.see(tk.END)

    def countdown_callback(self, n, f):
        if n > 0:
            self.console_insert(str(n))
            n -= 1
            self.root.after(1000, lambda: self.countdown_callback(n, f))
            return
        else:
            self.root.after(1000, f)
            return

    def run_bot(self, logic):
        self.console_insert("Через 3 секунды бот начнет работать.")
        self.countdown_callback(3, logic.start)

    def start(self):
        if self.configured and not self.logic.is_running():
            # open logfile
            datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
            self.log_file = open(LOG_FILE_NAME + datetime_str, 'w')
            self.console_insert(STARTED)
            self.run_bot(self.logic)
        elif not self.configured:
            self.console_insert(NOT_CONFIGURED)
        elif self.logic.is_running():
            self.console_insert(ALREADY_RUNNING)

    def test(self):
        if self.configured:
            self.collect()
        else:
            self.console.insert(tk.END, NOT_CONFIGURED)

    def calc_equity(self):
        hand_equity, comb_equity, hand_histogram, ev = self.logic.get_flop_equity()

        self.equity_ax.clear()
        self.configure_equity_figure(hand_equity)
        self.loss_num_text.configure(text='{:.2f}'.format(hand_equity[0]))
        self.close_loss_num_text.configure(text='{:.2f}'.format(hand_equity[1]))
        self.draw_num_text.configure(text='{:.2f}'.format(hand_equity[2]))
        self.close_win_num_text.configure(text='{:.2f}'.format(hand_equity[3]))
        self.win_num_text.configure(text='{:.2f}'.format(hand_equity[4]))
        self.ev_label.configure(text=EV_PREFIX + '{:.2f}'.format(ev))
        if len(hand_equity) > 5:
            self.dnqual_num_text.configure(text='{:.2f}'.format(hand_equity[5]))
        else:
            self.dnqual_num_text.configure(text=EQUITY_PLACEHOLDER)
        self.equity_canvas.draw()

        self.comb_equity_ax.clear()
        self.configure_comb_equity_figure(comb_equity, hand_histogram)
        self.comb_equity_canvas.draw()

    def stop(self):
        if self.logic.is_running():
            self.console_insert(STOPPED)
            self.log_file.close()
            self.logic.stop()

    def collect(self):
        def print_all():
            cards = self.uip.all()
            s = ""
            for card in cards:
                s += card + " "
            self.console_insert(s)

        if self.configured:
            print_all()

    def some_test(self):
        self.uip.fold()

    def uip_config(self):
        name = self.game_var.get()
        sel = SELs[name]
        filename = PRESET_FOLDER + "/" + name + BOT_GAME_SETTINGS_EXT

        self.root.withdraw()
        toplev = tk.Toplevel()
        ts = TableSettings(toplev, name, sel, filename, cr=self.cr, slr=self.slr, s=2)
        ts.pack()
        toplev.mainloop()
        self.root.deiconify()

        self.cconf = get_preset_used()
        if self.cconf is not None:
            self.configured = True
            self.uip = UiProjection(self.cr, self.slr, self.cconf)
            self.logic = CasinoPokerLogic(uip=self.uip, executor=self, sa=self.sa)
            self.logic.subscribe_listener(self)
            self.console_insert(SETTINGS_SUCCESSFULLY_APPLIED)
        else:
            self.console_insert(PROBLEM_OCCURED)

    def exec(self, task, delay):
        self.after(delay * 1000, task)

    def handle_message(self, msg, val):
        if msg == Messages.HAND_DEALT_MSG:
            self.clear_ui()
            self.console_insert("-~*~-")
        elif msg == Messages.HOLE_CARDS_MSG:
            for card in val:
                self.bot_cards_text.insert(tk.END, card + " ")
        elif msg == Messages.FLOP_CARDS_MSG:
            for card in val:
                self.table_cards_text.insert(tk.END, card + " ")
        elif msg == Messages.HAND_EQUITY_MSG:
            self.equity_ax.clear()
            self.configure_equity_figure(val)
            self.loss_num_text.configure(text='{:.2f}'.format(val[0]))
            self.close_loss_num_text.configure(text='{:.2f}'.format(val[1]))
            self.draw_num_text.configure(text='{:.2f}'.format(val[2]))
            self.close_win_num_text.configure(text='{:.2f}'.format(val[3]))
            self.win_num_text.configure(text='{:.2f}'.format(val[4]))
            if len(val) > 5:
                self.dnqual_num_text.configure(text='{:.2f}'.format(val[5]))
            else:
                self.dnqual_num_text.configure(text=EQUITY_PLACEHOLDER)
            self.equity_canvas.draw()
        elif msg == Messages.COMB_EQUITY_AND_HISTOGRAM_MSG:
            self.comb_equity_ax.clear()
            self.configure_comb_equity_figure(val[0], val[1])
            self.comb_equity_canvas.draw()
        elif msg == Messages.RIVER_CARDS_MSG:
            for card in val:
                self.table_cards_text.insert(tk.END, card + " ")
        elif msg == Messages.OPPONENT_CARDS_MSG:
            for card in val:
                self.opp_cards_text.insert(tk.END, card + " ")
        elif msg == Messages.BOT_RIVER_COMB_MSG:
            self.my_comb_text.delete(1.0, tk.END)
            self.my_comb_text.insert(tk.END, comb2str(val))
        elif msg == Messages.OPP_RIVER_COMB_MSG:
            self.opp_comb_text.delete(1.0, tk.END)
            self.opp_comb_text.insert(tk.END, comb2str(val))
            self.opp_comb = val
        elif msg == Messages.OUTCOME_MSG:
            self.console_insert(Outcomes.to_str(val))
            if val == Outcomes.FOLD:
                self.fold_count += 1
            elif val == Outcomes.LOST:
                self.lose_count += 1
            elif val == Outcomes.WON:
                self.win_count += 1
            elif val == Outcomes.DRAW:
                self.draw_count += 1
            elif val == Outcomes.NQ:
                self.nq_count += 1
            self.console_insert("F" + str(self.fold_count) +
                                " L" + str(self.lose_count) +
                                " D" + str(self.draw_count) +
                                " W" + str(self.win_count) +
                                " NQ" + str(self.nq_count))
            self.log_file.write(self.logic.get_log())
        elif msg == Messages.EV_MSG:
            self.ev_label.configure(text=EV_PREFIX + '{:.5f}'.format(val))
        elif msg == Messages.DECISION_MADE_MSG:
            self.console_insert(val)
        elif msg == Messages.EVAL_TIME_MSG:
            self.console_insert(str(val))
        elif msg == Messages.OUTCOME_STACK_MSG:
            self.console_insert(str(val))
        elif msg == Messages.HAND_BALANCE_MSG:
            self.console_insert("Баланс = " + str(val))
        elif msg == Messages.CURRENT_STAKE_MSG:
            self.console_insert("Ставка = " + str(val))


def build_screen_element_tree(pseudoroot, conf, sel):
    for n in conf.shape[0]:
        pseudoroot.add_child(conf[n, :4])
    return pseudoroot, dict(zip(sel, pseudoroot.children))


class TableSettings(tk.Frame):
    def __init__(self, root,
                 cur, sel, filename, cr, slr, presets=None, current=EMPTY, s=1, dt=1000):
        """
        :param root: root of the Frame
        :param conf: configurations of screen elements (red, green, blue, inverse, delta, x1, y1, x2, y2)
        :param sel: Screen Elements List - list of the elements whose regions need to be determined to be edited
                    and filtered by the bot
        :param dt: each dt milliseconds a screenshot is made to refresh the canvas
        """
        super().__init__(root)

        self.cr = cr
        self.slr = slr

        if presets is None:
            try:
                f = open(PRESET_FOLDER + "/" + cur + BOT_GAME_SETTINGS_EXT, mode='rb')
                self.presets = pickle.load(f)
                f.close()
            except FileNotFoundError:
                self.presets = {EMPTY: ViewTree.empty([0, 0, root.winfo_screenwidth(), root.winfo_screenheight()], sel)}
        else:
            self.presets = presets

        self.view_tree = self.presets[current]

        stop = False

        def r_change(*args):
            selected_element().features[0] = r_scroll_var.get()

        def g_change(*args):
            selected_element().features[1] = g_scroll_var.get()

        def b_change(*args):
            selected_element().features[2] = b_scroll_var.get()

        def inverse_change(*args):
            selected_element().features[3] = inv_var.get()

        def delta_change(*args):
            selected_element().features[4] = delta_var.get()

        def fst_trim_change(*args):
            selected_element().features[5] = fst_trim_var.get()

        def clear_corners_change(*args):
            selected_element().features[6] = clear_corners_var.get()

        def clear_borders_change(*args):
            selected_element().features[7] = clear_borders_var.get()

        def snd_trim_change(*args):
            selected_element().features[8] = snd_trim_var.get()

        delta_var = tk.IntVar()
        delta_var.trace_add('write', delta_change)
        r_scroll_var = tk.IntVar()
        r_scroll_var.trace_add('write', r_change)
        g_scroll_var = tk.IntVar()
        g_scroll_var.trace_add('write', g_change)
        b_scroll_var = tk.IntVar()
        b_scroll_var.trace_add('write', b_change)

        inv_var = tk.BooleanVar()
        inv_var.trace_add('write', inverse_change)

        fst_trim_var = tk.BooleanVar()
        fst_trim_var.trace_add('write', fst_trim_change)
        clear_corners_var = tk.BooleanVar()
        clear_corners_var.trace_add('write', clear_corners_change)
        clear_borders_var = tk.BooleanVar()
        clear_borders_var.trace_add('write', clear_borders_change)
        snd_trim_var = tk.BooleanVar()
        snd_trim_var.trace_add('write', snd_trim_change)

        div_var = tk.BooleanVar()

        last_selection = 0

        spawnbox = np.array((100, 100, 0, 0), dtype=np.uint16)

        def selected_element():
            return self.view_tree[last_selection]

        def create_preset():
            def ok():
                new_name = name_input.get()
                if new_name != EMPTY:
                    new_config = self.presets[current_preset_var.get()].copy()
                    self.presets[new_name] = new_config
                    m = presets_option_menu.children['menu']
                    m.add_command(label=new_name, command=tk._setit(current_preset_var, new_name))
                    name_win.destroy()
            name_win = tk.Toplevel(root)
            name_win.title(PRESET_NAME)
            name_input = tk.Entry(name_win)
            name_input.pack()
            ok_btn = tk.Button(name_win, text="OK", command=ok)
            ok_btn.pack(fill=tk.X)

        def delete_preset():
            pass

        def shoot_screen():
            def shoot_cards():
                shoot_cards_button.configure(state='disabled')
                shoot_digits_button.configure(state='disabled')
                card_nodes = self.view_tree.prefix_elements(CARD_PREFIX)
                screenshots = []
                for node in card_nodes:
                    screenshots.append(node.scrshot())
                for c in range(len(screenshots)):
                    shot = screenshots[c]
                    if shot.size:
                        rns = self.cr.ranknsuit(shot)
                        rank, suit = Image.fromarray(rns[0]), Image.fromarray(rns[1])
                        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
                        rank_filename = str(self.cr.rc.classify(rns[0])[0]) + "-" + str(c) + datetime_str + ".png"
                        suit_filename = str(self.cr.sc.classify(rns[1])[0]) + "-" + str(c) + datetime_str + ".png"
                        suit.save("training-source/suits/" + suit_filename)
                        rank.save("training-source/ranks/" + rank_filename)
                shoot_cards_button.configure(state='normal')
                button_holder.quit()

            def shoot_dilines():
                shoot_cards_button.configure(state='disabled')
                shoot_digits_button.configure(state='disabled')
                digit_nodes = self.view_tree.prefix_elements(DIGITS_LINE_PREFIX)

                for c in range(len(digit_nodes)):
                    shot = digit_nodes[c].scrshot()
                    delta = digit_nodes[c].get_features()[4]
                    digits = waterfall_box(shot, delta)[0]
                    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
                    n = 0
                    for digit in digits:
                        d = Image.fromarray(trim(digit)[0])
                        digit_filename = datetime_str + "-" + str(n) + ".png"
                        d.save("digits/" + digit_filename)
                        n += 1
                shoot_cards_button.configure(state='normal')
                shoot_digits_button.configure(state='normal')
                button_holder.quit()

            root.withdraw()

            button_holder = tk.Toplevel()
            shoot_cards_button = tk.Button(button_holder, text=SHOOT_ALL_CARDS, command=shoot_cards)
            shoot_cards_button.pack()
            shoot_digits_button = tk.Button(button_holder, text=SHOOT_ALL_DIGITS, command=shoot_dilines)
            shoot_digits_button.pack()
            button_holder.mainloop()
            button_holder.destroy()

            root.deiconify()

        def change_config(n):
            nonlocal spawnbox

            config = self.view_tree[n].get_features()
            bbox = self.view_tree[n].get_bbox()

            r = config[0]
            g = config[1]
            b = config[2]
            inv = bool(config[3])
            delta = config[4]
            ft = bool(config[5])
            cc = bool(config[6])
            cb = bool(config[7])
            st = bool(config[8])
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            x1_var.set(x1)
            x2_var.set(x2)
            y1_var.set(y1)
            y2_var.set(y2)

            delta_var.set(delta)
            inv_var.set(inv)

            fst_trim_var.set(ft)
            clear_corners_var.set(cc)
            clear_borders_var.set(cb)
            snd_trim_var.set(st)

            r_scroll_var.set(r)
            g_scroll_var.set(g)
            b_scroll_var.set(b)

        def save():
            if current_preset_var.get() != EMPTY:
                nonlocal last_selection
                n = listbox.curselection()
                print(n)
                if n != ():
                    n = n[0]
                else:
                    n = last_selection

                c = np.empty(shape=9, dtype=np.uint16)
                x1 = x1_var.get()
                y1 = y1_var.get()
                x2 = x2_var.get()
                y2 = y2_var.get()

                nbbox = x1, y1, x2, y2

                c[0] = r_scroll_var.get()
                c[1] = g_scroll_var.get()
                c[2] = b_scroll_var.get()

                c[3] = int(inv_var.get())
                c[4] = delta_var.get()

                c[5] = fst_trim_var.get()
                c[6] = clear_corners_var.get()
                c[7] = clear_borders_var.get()
                c[8] = snd_trim_var.get()

                self.view_tree[n].set_features(c)
                self.view_tree[n].set_bbox(nbbox)

                save_button.configure(text=CHANGES_SAVED)

        def x1_change():
            x1 = x1_var.get()
            x2 = x2_var.get()
            if x1 < x2:
                selected_element().x1 = x1

        def y1_change():
            y1 = y1_var.get()
            y2 = y2_var.get()
            if y1 < y2:
                selected_element().y1 = y1

        def x2_change():
            x1 = x1_var.get()
            x2 = x2_var.get()
            if x1 < x2:
                selected_element().x2 = x2

        def y2_change():
            y1 = y1_var.get()
            y2 = y2_var.get()
            if y1 < y2:
                selected_element().y2 = y2

        def listbox_select(t):
            nonlocal last_selection
            n = listbox.curselection()
            if n != ():
                save_button.configure(text=SAVE_SETTINGS)
                n = n[0]
                last_selection = n
                change_config(n)

        def choose_main_bounding_region():
            r = self.view_tree.root.get_absolute_bbox()
            spawn = [r[2] - r[0], r[3] - r[1], r[0], r[1]]
            root.withdraw()
            main_bbox = tpwin(spawn, bg='red')[0]
            self.view_tree.root.set_bbox(main_bbox)
            x1, y1, x2, y2 = selected_element().get_bbox()
            x1_var.set(x1)
            x2_var.set(x2)
            y1_var.set(y1)
            y2_var.set(y2)
            print(str(main_bbox))
            root.deiconify()

        def choose_region():
            nonlocal spawnbox
            nonlocal x1_var, x2_var, y1_var, y2_var

            root.withdraw()

            bbox, spawnbox = tpwin(spawnbox=spawnbox)

            selected_element().set_bbox(bbox)
            bbox = selected_element().get_bbox()
            x1_var.set(bbox[0])
            x2_var.set(bbox[2])
            y1_var.set(bbox[1])
            y2_var.set(bbox[3])

            root.deiconify()

        choiceframe = tk.Frame(self)
        lbframe = tk.Frame(choiceframe)

        def preset_change(*args):
            create_preset_button['state'] = tk.NORMAL
            print(current_preset_var.get())
            self.view_tree = self.presets[current_preset_var.get()]
            if current_preset_var.get() == EMPTY:
                save_button['state'] = tk.DISABLED
                delete_preset_button['state'] = tk.DISABLED
            else:
                save_button['state'] = tk.NORMAL
                delete_preset_button['state'] = tk.NORMAL
            change_config(last_selection)

        presets_frame = tk.Frame(choiceframe)
        choose_preset_label = tk.Label(presets_frame, text=CHOOSE_PRESET)
        choose_preset_label.pack(fill=tk.BOTH, padx=5)
        current_preset_var = tk.StringVar()
        current_preset_var.set(" ")
        current_preset_var.trace("w", preset_change)
        presets_option_menu = tk.OptionMenu(presets_frame, current_preset_var, *self.presets.keys())
        presets_option_menu.pack(fill=tk.BOTH, pady=5)
        preset_buttons_frame = tk.Frame(presets_frame)

        create_preset_button = tk.Button(preset_buttons_frame, text=CREATE_PRESET, command=create_preset)
        create_preset_button.pack(fill=tk.BOTH)
        create_preset_button['state'] = tk.DISABLED

        delete_preset_button = tk.Button(preset_buttons_frame, text=DELETE_PRESET, command=delete_preset)
        delete_preset_button.pack(fill=tk.BOTH)
        delete_preset_button['state'] = tk.DISABLED

        preset_buttons_frame.pack(fill=tk.BOTH)
        presets_frame.pack(fill=tk.BOTH, side=tk.TOP, pady=5, padx=5)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        coord_frame = tk.Frame(choiceframe)
        xy_frame = tk.Frame(coord_frame)
        x_frame = tk.Frame(xy_frame)
        y_frame = tk.Frame(xy_frame)

        x_label = tk.Label(x_frame, text="X")
        y_label = tk.Label(y_frame, text="Y")

        x1_var = tk.IntVar()
        x1_sb = tk.Spinbox(x_frame, from_=0, to=screen_w, increment=1, textvariable=x1_var, command=x1_change, width=6)
        y1_var = tk.IntVar()
        y1_sb = tk.Spinbox(y_frame, from_=0, to=screen_h, increment=1, textvariable=y1_var, command=y1_change, width=6)
        x2_var = tk.IntVar()
        x2_sb = tk.Spinbox(x_frame, from_=0, to=screen_w, increment=1, textvariable=x2_var, command=x2_change, width=6)
        y2_var = tk.IntVar()
        y2_sb = tk.Spinbox(y_frame, from_=0, to=screen_h, increment=1, textvariable=y2_var, command=y2_change, width=6)

        x_label.pack(fill=tk.X, pady=3)
        y_label.pack(fill=tk.X, pady=3)
        x1_sb.pack(fill=tk.X, pady=1)
        y1_sb.pack(fill=tk.X, pady=1)
        x2_sb.pack(fill=tk.X, pady=1)
        y2_sb.pack(fill=tk.X, pady=1)

        x_frame.pack(side=tk.LEFT, padx=2)
        y_frame.pack(side=tk.LEFT, padx=2)
        xy_frame.pack(fill=tk.BOTH)
        choose_region_btn = tk.Button(coord_frame, text=CHOOSE_REGION, command=choose_region)
        choose_region_btn.pack(fill=tk.X)

        sb = tk.Scrollbar(lbframe, orient=tk.VERTICAL)
        listbox = tk.Listbox(lbframe, yscrollcommand=sb.set, height=10, selectmode=tk.SINGLE)
        listbox.bind("<<ListboxSelect>>", listbox_select)

        for e in sel:
            listbox.insert(tk.END, e)

        lab = tk.Label(choiceframe, text=CHOOSE_ELEMENT)
        lab.pack(fill=tk.X, side=tk.TOP, pady=4)

        sb.config(command=listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        listbox.pack(fill=tk.Y, side=tk.TOP)
        lbframe.pack()

        coord_frame.pack(fill=tk.BOTH, padx=5)

        show_div_frame = tk.Frame(choiceframe)

        delta_frame = tk.Frame(show_div_frame)
        delta_label = tk.Label(delta_frame)
        delta_label.configure(text=DELTA)
        delta_label.pack(side=tk.LEFT)
        delta_frame.pack()
        delta_spinbox = tk.Spinbox(delta_frame, from_=0, to=100, increment=1, textvariable=delta_var, width=4)
        delta_spinbox.pack(side=tk.LEFT)

        showdivcb = tk.Checkbutton(show_div_frame, text=SHOW_DIV, variable=div_var)
        showdivcb.pack()
        show_div_frame.pack()

        main_btns_frame = tk.Frame(choiceframe)

        def save_and_exit():
            nonlocal stop
            stop = True
            file = open(filename, mode='wb')
            pickle.dump(self.presets, file, pickle.HIGHEST_PROTOCOL)
            file.close()
            cp = self.presets[current_preset_var.get()]
            cp_file = open(PATH_TO_PRESET_USED, mode='wb')
            pickle.dump(cp, cp_file, pickle.HIGHEST_PROTOCOL)
            cp_file.close()
            root.destroy()
            root.quit()

        main_bbox_button = tk.Button(main_btns_frame, text=MAIN_BOUNDING_REGION, command=choose_main_bounding_region, height=2)
        main_bbox_button.pack(fill=tk.X, padx=5, pady=5)
        save_button = tk.Button(main_btns_frame, text=SAVE_SETTINGS, command=save, height=2)
        save_button.pack(fill=tk.X, padx=5, pady=5)
        save_button['state'] = tk.DISABLED
        screenshot_button = tk.Button(main_btns_frame, text=TAKE_SCREENSHOT, command=shoot_screen, height=2)
        screenshot_button.pack(fill=tk.X, padx=5, pady=5)
        savenexit_btn = tk.Button(main_btns_frame, text=SAVE_AND_EXIT, command=save_and_exit, height=2)
        savenexit_btn.pack(fill=tk.X, padx=5, pady=5)

        main_btns_frame.pack(fill=tk.X)

        """
        def discard():
            pass
        discard_button = tk.Button(choiceframe, text=discard_changes_txt, command=discard)
        discard_button.pack(side=tk.BOTTOM, fill=tk.Y)
        """

        choiceframe.pack(side=tk.LEFT, fill=tk.BOTH)

        cframe = tk.Frame(self)
        mplframe = tk.Frame(cframe)
        fframe = tk.Frame(cframe)

        matplotlib.use("TkAgg")

        figure = plt.figure()
        canvas = FigureCanvasTkAgg(figure, mplframe)
        ax = figure.add_subplot(111)
        canvas.get_tk_widget().pack(side=tk.TOP)
        toolbar = NavigationToolbar2Tk(canvas, mplframe)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=30)

        def canvas_update():
            nonlocal ax

            figure.delaxes(ax)
            ax = figure.add_subplot(111)
            e = selected_element()
            # print(str(e.get_absolute_bbox()) + ", " + str(e.get_bbox()) + ", " + str(e.features))
            shape = e.shape()

            if shape[0] * shape[1] != 0 and bbox_check(e.get_bbox()):
                region = e.scrshot()
                ax.imshow(region)
                if div_var.get():
                    c = 0
                    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange']
                    delta = delta_var.get()
                    letters, boxindx = waterfall_box(img=region, delta=delta)
                    a = 0
                    for xx in boxindx:
                        if c > len(colors) - 1:
                            c = 0
                        region[:, xx[0]:xx[1]] = region[:, xx[0]:xx[1]] & letters[a]
                        if xx[0] > 0:
                            ax.axvspan(xx[0] - 1, xx[1], facecolor=colors[c], alpha=0.2)
                        else:
                            ax.axvspan(xx[0], xx[1], facecolor=colors[c], alpha=0.2)
                        c += 1
                        a += 1
                if check_ocr_res_var.get() and np.sum(region) > 0:
                    element_key = self.view_tree[last_selection].get_key()

                    if element_key.startswith(CARD_PREFIX):
                        empty, ratio = self.cr.is_empty(region)
                        empty_ratio_label.configure(text='{:.2f}'.format(ratio))
                        if not empty:
                            r = self.cr.ranknsuit(region)
                            rank, suit, rank_bbox, suit_bbox, rns_bbox = r

                            rns_xy = rns_bbox[0], rns_bbox[1]
                            rns_wh = rns_bbox[2] - rns_xy[0], rns_bbox[3] - rns_xy[1]
                            rns_rect = Rectangle(rns_xy, rns_wh[0], rns_wh[1], linewidth=1, edgecolor='g', fill=False)

                            rank_xy = rank_bbox[0], rank_bbox[1]
                            rank_wh = rank_bbox[2] - rank_xy[0], rank_bbox[3] - rank_xy[1]
                            rank_rect = Rectangle(rank_xy, rank_wh[0], rank_wh[1], linewidth=1, edgecolor='r',
                                                  fill=False)

                            suit_xy = suit_bbox[0], suit_bbox[1]
                            suit_wh = suit_bbox[2] - suit_xy[0], suit_bbox[3] - suit_xy[1]
                            suit_rect = Rectangle(suit_xy, suit_wh[0], suit_wh[1], linewidth=1, edgecolor='b',
                                                  fill=False)

                            ax.add_patch(rns_rect)
                            ax.add_patch(rank_rect)
                            ax.add_patch(suit_rect)

                            if np.size(rank) != 0 and np.size(suit) != 0:
                                ocr_res_label.configure(
                                    text=str(self.cr.rc.classify(rank)[0]) + str(self.cr.sc.classify(suit)[0]))
                        else:
                            ocr_res_label.configure(text="Nn")
                    elif element_key.startswith(DIGITS_LINE_PREFIX):
                        s = self.slr.readline(region)
                        ocr_res_label.configure(text=s)
                        empty_ratio_label.configure(text="")
                    else:
                        ocr_res_label.configure(text="")
                        empty_ratio_label.configure(text="")
            canvas.draw()
            if not stop:
                root.after(dt, canvas_update)

        scaleR = tk.Scale(fframe, from_=0, to=255, orient=tk.HORIZONTAL, variable=r_scroll_var)
        scaleR.pack(fill=tk.BOTH)
        scaleG = tk.Scale(fframe, from_=0, to=255, orient=tk.HORIZONTAL, variable=g_scroll_var)
        scaleG.pack(fill=tk.BOTH)
        scaleB = tk.Scale(fframe, from_=0, to=255, orient=tk.HORIZONTAL, variable=b_scroll_var)
        scaleB.pack(fill=tk.BOTH)

        textframe = tk.Frame(mplframe, pady=2)
        ocr_text = tk.Label(textframe, text="")
        ocr_text.pack()

        cframe.pack(side=tk.RIGHT)

        bframe = tk.Frame(mplframe)
        inversecb = tk.Checkbutton(bframe, text=INVERSE, variable=inv_var)
        inversecb.pack(side=tk.LEFT)

        check_ocr_res_var = tk.BooleanVar()
        fst_trim_checkbtn = tk.Checkbutton(bframe, text="FT", variable=fst_trim_var)
        fst_trim_checkbtn.pack(side=tk.LEFT)
        clear_corners_checkbtn = tk.Checkbutton(bframe, text="CC", variable=clear_corners_var)
        clear_corners_checkbtn.pack(side=tk.LEFT)
        clear_borders_checkbtn = tk.Checkbutton(bframe, text="CB", variable=clear_borders_var)
        clear_borders_checkbtn.pack(side=tk.LEFT)
        snd_trim_checkbtn = tk.Checkbutton(bframe, text="ST", variable=snd_trim_var)
        snd_trim_checkbtn.pack(side=tk.LEFT)

        ocr_res_checkbtn = tk.Checkbutton(bframe, text=SHOW_OCR_RESULT, variable=check_ocr_res_var)
        ocr_res_checkbtn.pack(side=tk.LEFT, padx=5)

        ocr_res_label = tk.Label(bframe)
        ocr_res_label.configure(text="")
        ocr_res_label.pack(side=tk.LEFT, padx=5)
        empty_ratio_label = tk.Label(bframe)
        empty_ratio_label.configure(text="")
        empty_ratio_label.pack(side=tk.LEFT, padx=5)

        mplframe.pack()
        textframe.pack()
        fframe.pack(fill=tk.X, padx=30, pady=10)
        bframe.pack(fill=tk.Y)

        canvas_update()


def tpwin(spawnbox=(100, 100, 0, 0), t=0.4, bg='green'):
    w, h, x, y = spawnbox

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    import tkinter as tk
    root = tk.Toplevel()
    root.title("<Space> to select")
    root.attributes('-alpha', t)
    root.configure(background=bg)

    root.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def space_pressed(event):
        nonlocal x, y, h, w
        x = root.winfo_x()
        y = root.winfo_y()
        h = root.winfo_height()
        w = root.winfo_width()
        xin = root.winfo_rootx()
        yin = root.winfo_rooty()

        nonlocal x1
        nonlocal x2
        nonlocal y1
        nonlocal y2
        x1 = xin
        x2 = x + w
        y1 = yin
        y2 = yin + h

        root.quit()

    root.bind("<space>", space_pressed)
    root.mainloop()
    root.destroy()
    print(str([x1, y1, x2, y2]))
    return [x1, y1, x2, y2], [w, h, x, y]


def bbox_check(bbox):
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]


def create_training_set(folder, output):
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    entries = listdir(folder)
    images = list(filter(lambda s: s.endswith(".png"), entries))

    char_sets = []
    # заполнить пустыми строками
    for i in range(len(images)):
        char_sets.append("")

    filter_config = np.zeros(shape=[len(images), 5], dtype=np.uint8)

    for j in range(len(images)):
        images[j] = Image.open(folder + "/" + images[j], "r")

    matplotlib.use("TkAgg")

    root = tk.Tk()
    root.title("training set init")
    root.resizable(False, False)

    frame = tk.Frame(root)
    frame.pack()

    figure = plt.figure()
    canvas = FigureCanvasTkAgg(figure, frame)
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.pack()

    sb_val = False
    original = True

    delta_var = tk.StringVar()
    text_var = tk.StringVar()

    def delta_change(x, y, z):
        try:
            delta = int(delta_var.get())
            filter_config[i, 4] = delta
        except ValueError:
            print("Delta must be a non-negative integer number.")
            if len(delta_var.get()) == 0:
                filter_config[i, 4] = 0
                print("delta: " + delta_var.get())

    delta_var.trace_add('write', delta_change)

    i = 0

    def show():
        nonlocal ax
        figure.delaxes(ax)
        ax = figure.add_subplot(111)
        if original:
            ax.imshow(images[i])
            canvas.draw()
        else:
            red = filter_config[i, 0]
            green = filter_config[i, 1]
            blue = filter_config[i, 2]
            filtered = binarize_rgb(np.asarray(images[i]), red, green, blue)
            # если инверсированная картинка
            if filter_config[i, 3]:
                filtered = ~filtered
            if sb_val:
                c = 0
                colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange']
                delta = filter_config[i, 4]
                letters, boxindx = waterfall_box(img=filtered, delta=delta)
                a = 0
                for xx in boxindx:
                    if c > len(colors) - 1:
                        c = 0
                    filtered[:, xx[0]:xx[1]] = filtered[:, xx[0]:xx[1]] & letters[a]
                    if xx[0] > 0:
                        ax.axvspan(xx[0] - 1, xx[1], facecolor=colors[c], alpha=0.2)
                    else:
                        ax.axvspan(xx[0], xx[1], facecolor=colors[c], alpha=0.2)
                    c += 1
                    a += 1
            ax.imshow(filtered)
            canvas.draw()

    ax = figure.add_subplot(111)
    show()

    filterFrame = tk.Frame(root)

    checkbtnFrame = tk.Frame(root)

    def orig():
        nonlocal original
        original = not original
        show()

    def inverse():
        filter_config[i, 3] = not filter_config[i, 3]
        show()

    originalCbtn = tk.Checkbutton(checkbtnFrame, text="Show Original", command=orig)
    originalCbtn.pack(side=tk.LEFT)
    originalCbtn.select()
    inverseCbtn = tk.Checkbutton(checkbtnFrame, text="Invert", command=inverse)
    inverseCbtn.pack(side=tk.LEFT)

    checkbtnFrame.pack(fill=tk.X)

    def sc(v):
        red = scaleR.get()
        green = scaleG.get()
        blue = scaleB.get()
        filter_config[i, 0] = red
        filter_config[i, 1] = green
        filter_config[i, 2] = blue
        show()

    scaleR = tk.Scale(filterFrame, from_=0, to=255, orient=tk.HORIZONTAL, command=sc)
    scaleR.pack(side=tk.TOP, fill=tk.BOTH)
    scaleG = tk.Scale(filterFrame, from_=0, to=255, orient=tk.HORIZONTAL, command=sc)
    scaleG.pack(side=tk.BOTTOM, fill=tk.BOTH)
    scaleB = tk.Scale(filterFrame, from_=0, to=255, orient=tk.HORIZONTAL, command=sc)
    scaleB.pack(side=tk.BOTTOM, fill=tk.BOTH)

    filterFrame.pack(fill=tk.X)

    countLabel = tk.Label(checkbtnFrame, text=(str(i + 1) + "/" + str(len(images))))

    def pic_forward():
        nonlocal i

        i += 1
        if i < len(images):
            scaleR.set(filter_config[i, 0])
            scaleG.set(filter_config[i, 1])
            scaleB.set(filter_config[i, 2])
            delta_var.set(filter_config[i, 4])
            text_var.set(char_sets[i])
            if filter_config[i, 3]:
                inverseCbtn.select()
            else:
                inverseCbtn.deselect()
            show()
        else:
            i = len(images) - 1
        countLabel.config(text=(str(i + 1) + "/" + str(len(images))))

    def pic_backward():
        nonlocal i

        i -= 1
        if i > -1:
            scaleR.set(filter_config[i, 0])
            scaleG.set(filter_config[i, 1])
            scaleB.set(filter_config[i, 2])
            delta_var.set(filter_config[i, 4])
            text_var.set(char_sets[i])
            if filter_config[i, 3]:
                inverseCbtn.select()
            else:
                inverseCbtn.deselect()
            show()
        else:
            i = 0
        countLabel.config(text=(str(i + 1) + "/" + str(len(images))))

    backwardButton = tk.Button(checkbtnFrame, text="<", command=pic_backward)
    forwardButton = tk.Button(checkbtnFrame, text=">", command=pic_forward)

    forwardButton.pack(side=tk.RIGHT)
    countLabel.pack(side=tk.RIGHT)
    backwardButton.pack(side=tk.RIGHT)

    doneFrame = tk.Frame(root)
    deltaEntry = tk.Entry(doneFrame, textvariable=delta_var)
    deltaLabel = tk.Label(doneFrame, text="Delta Val")

    def done():
        nonlocal root
        root.quit()
        root.destroy()
        try:
            n = 0
            for img in images:

                red = filter_config[i, 0]
                green = filter_config[i, 1]
                blue = filter_config[i, 2]
                filtered = binarize_rgb(np.asarray(img), red, green, blue)
                # если инверсированная картинка
                if filter_config[i, 3]:
                    filtered = ~filtered
                delta = filter_config[i, 4]
                letters, boxes = waterfall_box(filtered, delta)

                if len(letters) == len(char_sets[n]):
                    m = 0
                    for char in char_sets[n]:
                        li = Image.fromarray(trim(letters[m])[0])
                        filename = char + "-" + str(n) + "-" + str(m) + ".png"
                        li.save(output + "/" + filename)
                        m += 1
                else:
                    print("number of letters is not equal to the number of characters in sequence " + char_sets[n])
                n += 1
        except NotADirectoryError:
            print("Output path is not a directory")
        except FileNotFoundError:
            print("Cannot write to given directory")

    def show_div():
        nonlocal sb_val
        sb_val = True
        show()
        sb_val = False

    def save_char_set(x, y, z):
        char_sets[i] = text_var.get()

    text_var.trace_add(mode='write', callback=save_char_set)

    textEntry = tk.Entry(doneFrame, textvariable=text_var)
    textLabel = tk.Label(doneFrame, text="Char Set")

    doneButton = tk.Button(doneFrame, text="Done", command=done)
    showBoxButton = tk.Button(doneFrame, text="Show Division", command=show_div)
    doneFrame.pack(side=tk.BOTTOM, fill=tk.BOTH)
    deltaLabel.pack(side=tk.LEFT)
    deltaEntry.pack(side=tk.LEFT)
    showBoxButton.pack(side=tk.LEFT, fill=tk.BOTH)
    doneButton.pack(side=tk.LEFT, fill=tk.BOTH)
    textEntry.pack(side=tk.LEFT)
    textLabel.pack(side=tk.LEFT)
    root.mainloop()
