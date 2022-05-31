import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scr.games.livestream.parimatch.poker.scrapers import PM
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from time import sleep

# from scr.livestream.parimatch.poker.scrapers import PMPoker

HANDS = "Hands"
NUM1 = "#1"
NUM2 = "#2"
NUM3 = "#3"
NUM4 = "#4"
NUM5 = "#5"
NUM6 = "#6"

REAL = "Real"
CASINO = "Casino"
VALUE_PERCENT = "Value(%)"
STAKE = "Stake"
CARDS = "Cards"
TABLE = "Table"
FLOP = "Flop"
TURN = "Turn"
RIVER = "River"
COMBINATIONS = "Combinations"
COMB_LABELS = ['HC', 'OP', 'TP', 'SE', 'ST', 'FL', 'FH', 'QU', 'SF', 'RF']


def main():
    # miliseconds per frame
    mspf = 1000

    offset = 1

    root = tk.Tk()

    hand_label = tk.Label(root, text=HANDS)

    hand1_label = tk.Label(root, text=NUM1)
    hand2_label = tk.Label(root, text=NUM2)
    hand3_label = tk.Label(root, text=NUM3)
    hand4_label = tk.Label(root, text=NUM4)
    hand5_label = tk.Label(root, text=NUM5)
    hand6_label = tk.Label(root, text=NUM6)

    hand_number_labels = [
        hand1_label,
        hand2_label,
        hand3_label,
        hand4_label,
        hand5_label,
        hand6_label
    ]

    hand_label.grid(row=0, column=offset+0, columnspan=12)
    hand1_label.grid(row=1, column=offset+0, columnspan=2)
    hand2_label.grid(row=1, column=offset+2, columnspan=2)
    hand3_label.grid(row=1, column=offset+4, columnspan=2)
    hand4_label.grid(row=1, column=offset+6, columnspan=2)
    hand5_label.grid(row=1, column=offset+8, columnspan=2)
    hand6_label.grid(row=1, column=offset+10, columnspan=2)

    default_bg_color = hand_label['bg']

    hand11_str = tk.StringVar()
    hand12_str = tk.StringVar()
    hand21_str = tk.StringVar()
    hand22_str = tk.StringVar()
    hand31_str = tk.StringVar()
    hand32_str = tk.StringVar()
    hand41_str = tk.StringVar()
    hand42_str = tk.StringVar()
    hand51_str = tk.StringVar()
    hand52_str = tk.StringVar()
    hand61_str = tk.StringVar()
    hand62_str = tk.StringVar()

    hand_card_vars = [
        hand11_str,
        hand12_str,
        hand21_str,
        hand22_str,
        hand31_str,
        hand32_str,
        hand41_str,
        hand42_str,
        hand51_str,
        hand52_str,
        hand61_str,
        hand62_str
    ]

    hand11_entry = tk.Entry(root, width=2, textvariable=hand11_str)
    hand12_entry = tk.Entry(root, width=2, textvariable=hand12_str)
    hand21_entry = tk.Entry(root, width=2, textvariable=hand21_str)
    hand22_entry = tk.Entry(root, width=2, textvariable=hand22_str)
    hand31_entry = tk.Entry(root, width=2, textvariable=hand31_str)
    hand32_entry = tk.Entry(root, width=2, textvariable=hand32_str)
    hand41_entry = tk.Entry(root, width=2, textvariable=hand41_str)
    hand42_entry = tk.Entry(root, width=2, textvariable=hand42_str)
    hand51_entry = tk.Entry(root, width=2, textvariable=hand51_str)
    hand52_entry = tk.Entry(root, width=2, textvariable=hand52_str)
    hand61_entry = tk.Entry(root, width=2, textvariable=hand61_str)
    hand62_entry = tk.Entry(root, width=2, textvariable=hand62_str)

    hand11_entry.grid(row=2, column=offset+0)
    hand12_entry.grid(row=2, column=offset+1)
    hand21_entry.grid(row=2, column=offset+2)
    hand22_entry.grid(row=2, column=offset+3)
    hand31_entry.grid(row=2, column=offset+4)
    hand32_entry.grid(row=2, column=offset+5)
    hand41_entry.grid(row=2, column=offset+6)
    hand42_entry.grid(row=2, column=offset+7)
    hand51_entry.grid(row=2, column=offset+8)
    hand52_entry.grid(row=2, column=offset+9)
    hand61_entry.grid(row=2, column=offset+10)
    hand62_entry.grid(row=2, column=offset+11)

    tk.Label(root, text=CARDS).grid(row=2, column=0)

    casino_hand_odds_str = [tk.StringVar(root) for i in range(6)]
    casino_hand_odds_entries = [tk.Entry(root, width=5, textvariable=casino_hand_odds_str[i]) for i in range(6)]
    i = 0
    for entry in casino_hand_odds_entries:
        entry.grid(row=3, column=offset+i, columnspan=2)
        i += 2

    tk.Label(root, text=CASINO).grid(row=3, column=0)

    real_hand_odds_str = [tk.StringVar(root) for i in range(6)]
    real_hand_odds_entries = [tk.Entry(root, width=5, textvariable=real_hand_odds_str[i]) for i in range(6)]
    i = 0
    for entry in real_hand_odds_entries:
        entry.grid(row=4, column=offset + i, columnspan=2)
        i += 2

    tk.Label(root, text=REAL).grid(row=4, column=0)

    hand_value_str = [tk.StringVar(root) for i in range(6)]
    hand_value_entries = [tk.Entry(root, width=5, textvariable=hand_value_str[i]) for i in range(6)]
    i = 0
    for entry in hand_value_entries:
        entry.grid(row=5, column=offset + i, columnspan=2)
        i += 2

    tk.Label(root, text=VALUE_PERCENT).grid(row=5, column=0)

    """
    hand_stakes_str = [tk.StringVar(root) for i in range(6)]
    hand_stakes_entries = [tk.Entry(root, width=5, textvariable=hand_stakes_str[i]) for i in range(6)]
    i = 0
    for entry in hand_stakes_entries:
        entry.grid(row=6, column=offset + i, columnspan=2)
        i += 2

    tk.Label(root, text=STAKE).grid(row=6, column=0)
    """

    table_label = tk.Label(root, text=TABLE)
    table_label.grid(row=7, column=offset, columnspan=12)

    flop_label = tk.Label(root, text=FLOP)
    flop_label.grid(row=8, column=offset+0, columnspan=6)
    turn_label = tk.Label(root, text=TURN)
    turn_label.grid(row=8, column=offset+6, columnspan=3)
    river_label = tk.Label(root, text=RIVER)
    river_label.grid(row=8, column=offset+9, columnspan=3)

    flop1_str = tk.StringVar()
    flop2_str = tk.StringVar()
    flop3_str = tk.StringVar()
    flop1_entry = tk.Entry(root, width=2, textvariable=flop1_str)
    flop2_entry = tk.Entry(root, width=2, textvariable=flop2_str)
    flop3_entry = tk.Entry(root, width=2, textvariable=flop3_str)

    turn_str = tk.StringVar()
    turn_entry = tk.Entry(root, width=2, textvariable=turn_str)

    river_str = tk.StringVar()
    river_entry = tk.Entry(root, width=2, textvariable=river_str)

    table_card_vars = [
        flop1_str,
        flop2_str,
        flop3_str,
        turn_str,
        river_str
    ]

    flop1_entry.grid(row=9, column=offset+0, columnspan=2)
    flop2_entry.grid(row=9, column=offset+2, columnspan=2)
    flop3_entry.grid(row=9, column=offset+4, columnspan=2)

    turn_entry.grid(row=9, column=offset+6, columnspan=3)
    river_entry.grid(row=9, column=offset+9, columnspan=3)

    tk.Label(root, text=CARDS).grid(row=9, column=0)

    combinations_label = tk.Label(root, text=COMBINATIONS)
    combinations_label.grid(row=10, column=offset, columnspan=12)

    comb_casino_odds_str = []
    comb_casino_odds_entries = []
    comb_real_odds_str = []
    comb_real_odds_entries = []
    comb_value_str = []
    comb_value_entries = []

    comb_casino_label = tk.Label(root, text=CASINO)
    comb_real_label = tk.Label(root, text=REAL)
    comb_value_label = tk.Label(root, text=VALUE_PERCENT)
    comb_casino_label.grid(row=11, column=offset, columnspan=4)
    comb_real_label.grid(row=11, column=offset+4, columnspan=4)
    comb_value_label.grid(row=11, column=offset+8, columnspan=4)

    comb_labels = []

    comb_start_row = 12
    for n in range(comb_start_row, comb_start_row + len(COMB_LABELS)):
        label = tk.Label(root, text=COMB_LABELS[n-comb_start_row])
        comb_labels.append(label)
        label.grid(row=n, column=0)

        casino_odds_str = tk.StringVar(root)
        comb_casino_odds_str.append(casino_odds_str)
        casino_odds_entry = tk.Entry(root, width=12, textvariable=casino_odds_str)
        comb_casino_odds_entries.append(casino_odds_entry)
        casino_odds_entry.grid(row=n, column=offset, columnspan=4)

        real_odds_str = tk.StringVar(root)
        comb_real_odds_str.append(real_odds_str)
        real_odds_entry = tk.Entry(root, width=12, textvariable=real_odds_str)
        comb_real_odds_entries.append(real_odds_entry)
        real_odds_entry.grid(row=n, column=offset+4, columnspan=4)

        value_str = tk.StringVar(root)
        comb_value_str.append(value_str)
        value_entry = tk.Entry(root, width=12, textvariable=value_str)
        comb_value_entries.append(value_entry)
        value_entry.grid(row=n, column=offset+8, columnspan=4)

    matplotlib.use("TkAgg")

    hand_value_figure = plt.figure(figsize=[3, 3])
    hand_value_canvas = FigureCanvasTkAgg(hand_value_figure, root)
    hand_value_ax = hand_value_figure.add_subplot(111)
    hand_value_canvas.get_tk_widget().grid(row=0, column=offset+13, rowspan=10)

    comb_value_figure = plt.figure(figsize=[3, 3])
    comb_value_canvas = FigureCanvasTkAgg(comb_value_figure, root)
    comb_value_ax = comb_value_figure.add_subplot(111)
    comb_value_canvas.get_tk_widget().grid(row=11, column=offset+13, rowspan=10)

    def update():
        state_changed = pm.collect_state()
        if state_changed:
            sm = pm.sm

            for hnl in hand_number_labels:
                hnl.configure(bg=default_bg_color)
            for cl in comb_labels:
                cl.configure(bg=default_bg_color)

            if sm.hole_cards_str is not None:
                print("Отображение карт игроков...")
                print(sm.hole_cards_str)
                cards = sm.hole_cards_str.split()
                for card, var in zip(cards, hand_card_vars):
                    var.set(card)
                print("Готово!")
            else:
                for var in hand_card_vars:
                    var.set('')
            if sm.table_cards_str is not None:
                print("Отображение карт за столом...")
                print(sm.table_cards_str)
                cards = sm.table_cards_str.split()
                for i in range(len(cards)):
                    table_card_vars[i].set(cards[i])
                print("Готово!")
            else:
                for var in table_card_vars:
                    var.set('')
            if sm.iter < 4:
                x = np.arange(4)
                player_colors = ['r', 'g', 'b', 'c', 'm', 'y']
                player_labels = ['#1', '#2', '#3', '#4', '#5', '#6']
                print("Обновление информации о текущем раунде торгов... - " + str(sm.iter))
                hand_value_ax.clear()
                for i in range(6):
                    casino_hand_odds_str[i].set('{:.2f}'.format(sm.hand_odds[sm.iter, i]))
                    real_hand_odds_str[i].set('{:.2f}'.format(sm.hand_real[sm.iter, i]))
                    hand_value_str[i].set('{:.2f}'.format(sm.hand_vals[sm.iter, i] * 100))
                    if sm.comb_vals[sm.iter, i] > 0:
                        hand_value_entries[i].configure(background='green', foreground='white')
                    elif sm.comb_vals[sm.iter, i] < 0:
                        hand_value_entries[i].configure(background='red', foreground='white')
                    else:
                        hand_value_entries[i].configure(background=default_bg_color, foreground='black')
                    hand_value_ax.plot(x, sm.hand_vals[:, i] * 100, player_colors[i], label=player_labels[i])

                hand_value_ax.axhline(y=0, linestyle='-.', color='black')
                hand_value_ax.legend(loc="best", fontsize='xx-small', ncol=2)

                hand_value_ax.set_ylim(bottom=-25, top=25)
                hand_value_canvas.draw()

                comb_labels_str = ['HC', 'OP', 'TP', 'SE', 'ST', 'FL', 'FH', 'QU', 'SF', 'RF']
                comb_colors = ['silver', 'gold',
                               'orangered', 'crimson',
                               'purple', 'magenta',
                               'mediumpurple', 'mediumblue',
                               'springgreen', 'turquoise']
                comb_value_ax.clear()
                for i in range(10):
                    comb_casino_odds_str[i].set('{:.2f}'.format(sm.comb_odds[sm.iter, i]))
                    comb_real_odds_str[i].set('{:.2f}'.format(sm.comb_real[sm.iter, i]))
                    comb_value_str[i].set('{:.2f}'.format(sm.comb_vals[sm.iter, i] * 100))
                    if sm.comb_vals[sm.iter, i] > 0:
                        comb_value_entries[i].configure(background='green', foreground='white')
                    elif sm.comb_vals[sm.iter, i] < 0:
                        comb_value_entries[i].configure(background='red', foreground='white')
                    else:
                        comb_value_entries[i].configure(background=default_bg_color, foreground='black')
                    comb_value_ax.plot(x, sm.comb_vals[:, i] * 100, comb_colors[i], label=comb_labels_str[i])
                comb_value_ax.legend(loc="best", fontsize='xx-small', ncol=2)
                comb_value_ax.axhline(y=0, linestyle='-.', color='black')
                comb_value_ax.set_ylim(bottom=-25, top=25)
                comb_value_canvas.draw()

                print("Готово!")
            else:
                print("Достигнут ривер.")
                if sm.winners is not None:
                    i = 0
                    for is_winner in sm.winners:
                        if is_winner:
                            hand_number_labels[i].configure(bg="red")
                        i += 1
                if sm.winning_comb is not None:
                    comb_labels[sm.winning_comb].configure(bg="red")
        root.after(mspf, update)

    pm = PM()
    pm.start()
    sleep(10)
    pm.choose_game(3)
    sleep(2)
    pm.switch_off_video()
    sleep(4)
    pm.configure_tabs(1)

    root.after(10, update)

    root.mainloop()


if __name__ == '__main__':
    main()