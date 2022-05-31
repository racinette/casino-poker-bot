import numpy as np
from scr.poker.equity import LiveEquityCalc


class InvalidGameState(Exception):
    pass


class LivePokerSM:
    # live poker state machine
    BLIND = 0
    PREFLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    END = 5

    @staticmethod
    def state2str(n):
        if n == LivePokerSM.BLIND:
            return 'blind'
        elif n == LivePokerSM.PREFLOP:
            return 'preflop'
        elif n == LivePokerSM.FLOP:
            return 'flop'
        elif n == LivePokerSM.TURN:
            return 'turn'
        elif n == LivePokerSM.RIVER:
            return 'river'
        elif n == LivePokerSM.END:
            return 'end'
        else:
            raise InvalidGameState("Invalid state number. "
                                    "Expected: "
                                    "[" +
                                   str(LivePokerSM.BLIND) + ", " +
                                   str(LivePokerSM.PREFLOP) + ", " +
                                   str(LivePokerSM.FLOP) + ", " +
                                   str(LivePokerSM.TURN) + ", " +
                                   str(LivePokerSM.RIVER) + ", " +
                                   str(LivePokerSM.END) +
                                    "]" + " - instead got: " + str(n))

    def __init__(self, pn=6, blind_hand_odds=5.2, blind_comb_odds=None):
        if blind_comb_odds is None:
            self.blind_comb_odds = np.array([100, 5.80, 3.10, 6.80, 5.70, 8.70, 8.70, 80, 100, 100], dtype=np.float64)
        else:
            self.blind_comb_odds = blind_comb_odds

        self.blind_hand_odds = blind_hand_odds

        self.round_hand_stakes = np.zeros(shape=[4, pn], dtype=np.float64)
        self.hand_stakes = np.zeros(shape=[4, pn], dtype=np.float64)

        self.hands_casino_profit = np.zeros(shape=[4, pn], dtype=np.float64)
        self.hand_odds = np.zeros(shape=[4, pn], dtype=np.float64)
        self.comb_odds = np.zeros(shape=[4, 10], dtype=np.float64)
        self.hand_real = np.zeros(shape=[4, pn], dtype=np.float64)
        self.comb_real = np.zeros(shape=[4, 10], dtype=np.float64)
        self.comb_vals = np.zeros(shape=[4, 10], dtype=np.float64)
        self.hand_vals = np.zeros(shape=[4, pn], dtype=np.float64)

        self.hand_odds[0, :] = self.blind_hand_odds
        self.comb_odds[0, :] = self.blind_comb_odds
        self.hand_real[0, :] = pn
        self.comb_real[0, :] = [
            1876.17260788,
            6.20249663,
            3.27811459,
            7.16173343,
            6.02799643,
            9.25113882,
            9.18462567,
            111.79429849,
            637.75510204,
            5580.35714286
        ]
        self.hand_vals[0, :] = (self.hand_odds[0, :] - self.hand_real[0, :]) / self.hand_odds[0, :]
        self.comb_vals[0, :] = (self.comb_odds[0, :] - self.comb_real[0, :]) / self.comb_odds[0, :]

        self.game_state = LivePokerSM.BLIND

        self.casino_hand_profit = np.ones(shape=[4, pn], dtype=np.float64)
        self.casino_hand_ev = np.ones(shape=4, dtype=np.float64)
        self.casino_comb_profit = np.ones(shape=[4, 10], dtype=np.float64)
        self.casino_comb_ev = np.ones(shape=4, dtype=np.float64)

        self.hc_shape = [pn, 2]

        self.hole_cards = None
        self.hole_cards_str = None
        self.table_cards = None
        self.table_cards_str = None

        self.casino_hands_profit = 0
        self.winners = None
        self.winning_comb = None

        self.eqc = None

        self.iter = 0

        self.casino_hand_profit[0, :], self.casino_hand_ev[0] = \
            LiveEquityCalc.casino_blind_hand_profit(self.blind_hand_odds)
        self.casino_comb_profit[0, :], self.casino_comb_ev[0] = \
            LiveEquityCalc.casino_blind_comb_profit(self.blind_comb_odds)

    def is_blind(self):
        return self.game_state == LivePokerSM.BLIND

    def is_preflop(self):
        return self.game_state == LivePokerSM.PREFLOP

    def is_flop(self):
        return self.game_state == LivePokerSM.FLOP

    def is_turn(self):
        return self.game_state == LivePokerSM.TURN

    def is_river(self):
        return self.game_state == LivePokerSM.RIVER

    def blind_betting(self):
        if self.is_blind():
            self.iter = 0

    def blind_over(self):
        if self.is_blind():
            self.game_state = LivePokerSM.PREFLOP

    def preflop_betting(self, hole_cards, hand_odds, comb_odds):
        if self.is_preflop():
            self.iter = 1
            print("Префлоп")
            print(hole_cards[0])
            print(hole_cards[1])
            print(hand_odds)
            print(comb_odds)
            if self.eqc is None and (np.unique(hole_cards[1]).size == hole_cards[1].size):
                self.hole_cards_str = hole_cards[0]
                self.hole_cards = hole_cards[1].reshape(self.hc_shape)
                print(self.hole_cards)
                self.eqc = LiveEquityCalc(self.hole_cards)

            if self.eqc is not None and np.any(self.hand_odds[1, :] != hand_odds):
                self.hand_odds[1, :] = hand_odds
                self.hand_vals[1, :], self.hand_real[1, :] = self.eqc.win_draw_value(hand_odds)

            if self.eqc is not None and np.any(self.comb_odds[1, :] != comb_odds):
                self.comb_odds[1, :] = comb_odds
                self.comb_vals[1, :], self.comb_real[1, :] = self.eqc.win_comb_value(comb_odds)


    def preflop_over(self):
        if self.is_preflop():
            self.game_state = LivePokerSM.FLOP

    def flop_betting(self, table_cards, hand_odds, comb_odds):
        if self.is_flop():
            print("Флоп")
            print(table_cards[0])
            print(table_cards[1])
            print(hand_odds)
            self.iter = 2
            if self.eqc is not None and ((self.table_cards is None) or np.any(self.table_cards != table_cards)) and \
                    (np.unique(table_cards[1]).size == table_cards[1].size):
                self.table_cards_str, self.table_cards = table_cards
                self.eqc.set_table(self.table_cards)

            if self.eqc is not None and np.any(self.hand_odds[2, :] != hand_odds):
                self.hand_odds[2, :] = hand_odds
                self.hand_vals[2, :], self.hand_real[2, :] = self.eqc.win_draw_value(hand_odds)

            if self.eqc is not None and np.any(self.comb_odds[2, :] != comb_odds):
                self.comb_odds[2, :] = comb_odds
                self.comb_vals[2, :], self.comb_real[2, :] = self.eqc.win_comb_value(comb_odds)

    def flop_over(self):
        if self.is_flop():
            self.game_state = LivePokerSM.TURN

    def turn_betting(self, table_cards, hand_odds, comb_odds):
        if self.is_turn():
            print("Терн")
            print(table_cards[0])
            print(table_cards[1])
            print(hand_odds)
            print(comb_odds)
            self.iter = 3
            if self.eqc is not None and ((self.table_cards.size != 4) or np.any(self.table_cards != table_cards)) and \
                    (np.unique(table_cards[1]).size == table_cards[1].size):
                self.table_cards_str, self.table_cards = table_cards
                self.eqc.set_table(self.table_cards)

            if self.eqc is not None and np.any(self.hand_odds[3, :] != hand_odds):
                self.hand_odds[3, :] = hand_odds
                self.hand_vals[3, :], self.hand_real[3, :] = self.eqc.win_draw_value(hand_odds)


            if self.eqc is not None and np.any(self.comb_odds[3, :] != comb_odds):
                self.comb_odds[3, :] = comb_odds
                self.comb_vals[3, :], self.comb_real[3, :] = self.eqc.win_comb_value(comb_odds)

    def turn_over(self):
        if self.is_turn():
            self.game_state = LivePokerSM.RIVER

    def river(self, table_cards):
        if self.is_river():
            if self.eqc is not None and (table_cards[1].size == 5) and (np.unique(table_cards[1]).size == table_cards[1].size):
                print("Ривер")
                print(table_cards[0])
                print(table_cards[1])
                self.iter = 4
                self.table_cards_str, self.table_cards = table_cards
                self.eqc.set_table(table_cards[1])
                winner_hands = self.eqc.win_matrix[:, 0]
                winner_comb = self.eqc.tables_win_comb[0]

                self.winners = winner_hands
                self.winning_comb = winner_comb
                self.game_state = LivePokerSM.END

                del self.eqc
