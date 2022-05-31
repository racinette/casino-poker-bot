from scr.game import str2cards
import numpy as np
from scr.poker.equity import JacksOrBetter


def main():
    ec = JacksOrBetter(np.array([0, 1, 2, 3, 4, 6, 9, 25, 50, 250]))
    np.set_printoptions(suppress=True)
    running = True
    while running:
        line = input()
        if line == 'exit':
            running = False
        else:
            try:
                cards = str2cards(line, joker=False)
                if cards.size == 5:
                    pr = ec.play_ranking(cards)

                    print("top 3 plays by ev: ")
                    print(np.array(pr[0][:3], dtype=np.uint8))
                else:
                    print("Неправильное количество")
            except KeyError:
                print("Неправильные карты")
            except ValueError:
                print("Неправильные карты")


if __name__ == "__main__":
    main()