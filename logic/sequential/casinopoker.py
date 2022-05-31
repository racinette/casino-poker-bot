from scr.logic.communication import Messages, Outcomes
from scr.poker.equity import comp_combs_44, casino_poker_ev
from time import sleep
from time import time
from scr.poker.equity import flop_equity_44, find_comb
import numpy as np
from scr.game import card2int
from scr.statistics.betting.regulation import PolynomialGambling, StashGambling
from numpy import random

SAVE_OCR_LOGS_DIR = "ocr-logs/"
LOG_FILE_NAME = "logs/log"

BLIND = "blind"
PREFLOP = "preflop"
FLOP = "flop"
TURN = "turn"
RIVER = "river"
OPPONENT = "opponent"


class SimpleLogic:
    def __init__(self, uip, executor, actions, sa,
                 available_chips=None, init_stake=0.01, min_stake=0.01, max_stake=0.10,
                 const_risk=0.005, risk_dist=0.06):
        self.i = 0
        self.uip = uip
        self.actions = actions
        self.N = len(actions)
        self.log = ""
        self.listeners = []
        self.running = False
        self.executor = executor
        self.sa = sa
        self.const_risk = const_risk
        self.risk_dist = risk_dist
        if available_chips is None:
            self.available_chips = [0.01, 0.1]
        else:
            self.available_chips = available_chips
        self.chip_choose_funcs = {0.01: uip.choose_001_chip, 0.10: uip.choose_010_chip}
        self.min_stake = min_stake
        self.max_stake = max_stake
        self.current_stake = init_stake

    def is_running(self):
        return self.running

    def pause(self):
        self.running = False

    def stop(self):
        self.running = False
        self.i = 0
        self.sa.end_session()
        for listener in self.listeners:
            listener.handle_message(Messages.END_OF_SESSION_MSG, None)

    def run(self):
        if self.is_running():
            a = self.actions[self.i]
            msg = a()
            keys, values, delay = msg
            for listener in self.listeners:
                for key in keys:
                    listener.handle_message(key, values[key])
            self.i = (self.i + 1) % self.N
            self.executor.exec(self.run, delay)

    def subscribe_listener(self, l):
        self.listeners.append(l)

    def start(self):
        stack = self.uip.stack()
        self.sa.start_session(init_balance=stack, init_stake=self.current_stake)
        for listener in self.listeners:
            listener.handle_message(Messages.STARTING_STACK_MSG, stack)
        self.i = 0
        self.running = True
        self.executor.exec(self.run, 0)

    def get_log(self):
        return self.log

    def set_save_ocr_log(self, b):
        if b:
            self.uip.set_savedir(SAVE_OCR_LOGS_DIR)
        else:
            self.uip.set_savedir()


class CasinoPokerLogic(SimpleLogic):
    def __init__(self, uip, executor, sa, regulator='stash',
                 available_chips=None, init_stake=0.01, min_stake=0.01, max_stake=0.10,
                 const_risk=0.005, risk_dist=0.06,
                 blind_sleep=8, bet_sleep=3, fold_sleep=2):
        """
        :param uip: User Interface Projection - an object used to communicate with the User Interface
        :param executor: object, which is a subclass of TaskExecutor. Triggers execution of the logic.
        :param blind_sleep: amount of seconds to sleep after the deal_blind() function
        :param bet_sleep:
        :param fold_sleep:
        """
        actions = [self.change_stakes,
                   self.deal_blind,
                   self.new_hand_dealt,
                   self.see_holes,
                   self.see_flop,
                   self.calc_eqt,
                   self.decide,
                   self.see_river,
                   self.see_opponent_cards,
                   self.get_final_comb]

        super().__init__(uip, executor, actions, sa,
                         available_chips, init_stake, min_stake, max_stake, const_risk, risk_dist)

        if regulator == 'polynomial':
            self.regulator = PolynomialGambling(sa)
        elif regulator == 'stash':
            self.regulator = StashGambling(sa, self.available_chips[0], self.min_stake, self.max_stake)
        else:
            self.regulator = regulator

        self.stakes_changed = False

        self.hand_preblind_stack = 0
        self.hand_outcome_stack = 0
        self.hand_balance = 0

        self.blind_sleep = blind_sleep
        self.bet_sleep = bet_sleep
        self.fold_sleep = fold_sleep

        self.hand = []
        self.hand_equity = []
        self.comb_equity = []
        self.hand_histogram = []
        self.table = []
        self.hole_cards = []
        self.opponent_cards = []
        self.bet = False
        self.hand_n = 0
        self.eqts = []
        self.ev = 0

    def deal_blind(self):
        self.hand_n += 1
        self.table = []
        self.hand = []
        self.hole_cards = []
        self.opponent_cards = []
        self.ev = 0
        self.hand_balance = 0
        self.hand_outcome_stack = 0

        self.hand_preblind_stack = self.uip.stack()
        self.log = "Hand #" + str(self.hand_n) + " - stack: " + str(self.hand_preblind_stack) + "\n"
        if self.stakes_changed:
            self.sa.stakes_changed(self.current_stake)
            self.uip.deal_stakes()
        else:
            self.uip.deal()
        return [], {}, self.blind_sleep

    def new_hand_dealt(self):
        self.stakes_changed = False
        return ([Messages.HAND_DEALT_MSG],
                {Messages.HAND_DEALT_MSG: None},
                0)

    def see_holes(self):
        hole_cards = self.uip.hole()
        self.log += "hole cards: "
        for c in hole_cards:
            card = card2int(c)
            self.hole_cards.append(card)
            self.log += c + " "
            self.hand.append(card)
        self.log += "\n"
        return [Messages.HOLE_CARDS_MSG], {Messages.HOLE_CARDS_MSG: hole_cards}, 0

    def see_flop(self):
        flop_cards = self.uip.flop()
        self.log += "flop cards: "
        for c in flop_cards:
            self.log += c + " "
            card = card2int(c)
            self.hand.append(card)
            self.table.append(card)
        self.log += "\n"
        return [Messages.FLOP_CARDS_MSG], {Messages.FLOP_CARDS_MSG: flop_cards}, 0

    def calc_eqt(self):
        self.hand = np.array(self.hand, dtype=np.int8)
        t1 = time()
        hand_equity, comb_equity, hand_histogram = flop_equity_44(table=self.hand[2:], hand=self.hand[:2])
        t2 = time()
        eval_time = t2 - t1
        self.hand_equity = hand_equity
        self.log += "  hand eqt: " + np.array2string(hand_equity, prefix="  hand eqt: ") + "\n"
        self.comb_equity = comb_equity
        self.log += "  comb eqt: " + np.array2string(comb_equity, prefix="  comb eqt: ") + "\n"
        self.hand_histogram = hand_histogram
        self.log += " histogram: " + np.array2string(hand_histogram, prefix=" histogram: ") + "\n"
        self.ev = casino_poker_ev(comb_equity, hand_histogram)
        self.log += "        ev: " + str(self.ev) + "\n"
        return (
            [Messages.HAND_EQUITY_MSG, Messages.COMB_EQUITY_AND_HISTOGRAM_MSG,
             Messages.EV_MSG, Messages.EVAL_TIME_MSG],
            {Messages.HAND_EQUITY_MSG: hand_equity,
             Messages.COMB_EQUITY_AND_HISTOGRAM_MSG: (comb_equity, hand_histogram),
             Messages.EV_MSG: self.ev,
             Messages.EVAL_TIME_MSG: eval_time},
            0
        )

    def decide(self):
        c = find_comb(self.hand)
        if self.ev + self.const_risk + random.rand() * self.risk_dist >= 0:
            self.uip.bet()
            self.bet = True
            self.log += "Bet.\n"
            return ([Messages.DECISION_MADE_MSG],
                    {Messages.DECISION_MADE_MSG: "Bet"},
                    self.bet_sleep)
        else:
            self.log += "Fold"
            self.uip.fold()
            if c[0] > 0:
                self.log += " for sure"
                sleep(2)
                self.uip.sure()
            self.log += ".\n"
            self.bet = False
            # возврат на начальную позицию
            self.i = -1
            whole_table = -np.ones(shape=9, dtype=np.int8)
            whole_table[:5] = self.hole_cards + self.table
            self.hand_outcome_stack = self.uip.stack()
            self.hand_balance = self.hand_outcome_stack - self.hand_preblind_stack

            self.sa.hand_outcome(self.hand_outcome_stack,
                                 self.hand_balance,
                                 Outcomes.FOLD,
                                 whole_table,
                                 self.ev,
                                 -1, -1)

            return ([Messages.OUTCOME_MSG,
                     Messages.WHOLE_TABLE_MSG,
                     Messages.OUTCOME_STACK_MSG,
                     Messages.HAND_BALANCE_MSG],
                    {Messages.OUTCOME_MSG: Outcomes.FOLD,
                     Messages.OUTCOME_STACK_MSG: self.hand_outcome_stack,
                     Messages.WHOLE_TABLE_MSG: whole_table,
                     Messages.HAND_BALANCE_MSG: self.hand_balance},
                    self.fold_sleep)

    def see_river(self):
        if self.bet:
            river_cards = self.uip.only_river()
            self.log += "     river: "
            for r in river_cards:
                self.table.append(card2int(r))
                self.log += r + " "
            self.log += "\n"
            return [Messages.RIVER_CARDS_MSG], {Messages.RIVER_CARDS_MSG: river_cards}, 0
        return

    def see_opponent_cards(self):
        if self.bet:
            opponent_cards = self.uip.opponent_cards()
            self.log += "  opponent: "
            for o in opponent_cards:
                self.log += o + " "
                self.opponent_cards.append(card2int(o))
            self.log += "\n"
            return [Messages.OPPONENT_CARDS_MSG], {Messages.OPPONENT_CARDS_MSG: opponent_cards}, 0
        return

    def get_final_comb(self):
        if self.bet:
            my_hand = np.array(self.table + self.hole_cards, dtype=np.int)
            opp_hand = np.array(self.table + self.opponent_cards, dtype=np.int)
            whole_table = np.array(self.hole_cards + self.table + self.opponent_cards, dtype=np.int8)

            my_comb = find_comb(my_hand)
            opp_comb = find_comb(opp_hand)
            self.i = -1

            self.log += "  bot comb: "
            for e in my_comb:
                if type(e) == np.ndarray:
                    self.log += np.array2string(e) + " "
                else:
                    self.log += str(e) + " "
            self.log += "\n"
            self.log += "  opp comb: "
            for e in opp_comb:
                if type(e) == np.ndarray:
                    self.log += np.array2string(e) + " "
                else:
                    self.log += str(e) + " "
            self.log += "\n"

            c = comp_combs_44(my_comb, opp_comb)

            self.hand_outcome_stack = self.uip.stack()
            self.hand_balance = self.hand_outcome_stack - self.hand_preblind_stack

            # stack, hand_balance, outcome, table, ev, player_river_comb, opp_river_comb
            self.sa.hand_outcome(self.hand_outcome_stack,
                                 self.hand_balance, c,
                                 self.table,
                                 self.ev,
                                 my_comb,
                                 opp_comb)

            return ([Messages.OPP_RIVER_COMB_MSG,
                     Messages.BOT_RIVER_COMB_MSG,
                     Messages.OUTCOME_MSG,
                     Messages.WHOLE_TABLE_MSG,
                     Messages.HAND_BALANCE_MSG],
                    {Messages.OPP_RIVER_COMB_MSG: opp_comb,
                     Messages.BOT_RIVER_COMB_MSG: my_comb,
                     Messages.OUTCOME_MSG: c,
                     Messages.WHOLE_TABLE_MSG: whole_table,
                     Messages.HAND_BALANCE_MSG: self.hand_balance},
                    0)
        return

    def get_flop_equity(self):
        hole_cards = self.uip.hole()
        int_hole_cards = []
        for c in hole_cards:
            card = card2int(c)
            int_hole_cards.append(card)

        table = []
        flop_cards = self.uip.flop()
        for c in flop_cards:
            card = card2int(c)
            table.append(card)

        hand_equity, comb_equity, hand_histogram = flop_equity_44(table=np.array(table, dtype=np.int8),
                                                                  hand=np.array(int_hole_cards, dtype=np.int8))

        return hand_equity, comb_equity, hand_histogram, casino_poker_ev(comb_equity, hand_histogram)

    def change_stakes(self):
        if self.regulator is None:
            return [], {}, 0
        stake = self.regulator.regulate()
        print(stake)
        if not np.isnan(stake) and stake != self.current_stake:
            self.log += "Stake: " + str(stake) + "\n"
            if self.min_stake <= stake <= self.max_stake:
                # начать проверку чипов с конца (от самого большого)
                reversed_available_chips = self.available_chips[::-1]
                self.uip.open_stake_change_menu()
                sleep(2)
                remainder = stake
                for chip in reversed_available_chips:
                    if remainder - chip >= 0:
                        # минимум один чип данного достоинства будет положен на стол
                        # открыть меню выбора чипов
                        self.uip.open_stakes_spinbox()
                        sleep(3)
                        # выбрать нужный
                        self.chip_choose_funcs[chip]()
                        # положить его столько раз, сколько нужно поставить
                        while remainder - chip >= 0:
                            remainder -= chip
                            self.uip.put_chip()
                            sleep(2)
                        # если ставка сделана, прекратить дальнейшую раскладку чипов
                        if remainder == 0:
                            break
                self.current_stake = stake
                self.stakes_changed = True
            else:
                raise ValueError("Stake given does not fit into the allowed interval." +
                                 str(self.min_stake) + " </= " + str(stake) + " </= " + str(self.max_stake))
        return [], {}, 0

# interstate
# blind
# preflop
# flop
# turn
# river
class Logic:
    def blind(self):
        pass

    def preflop(self):
        pass

    def flop(self):
        pass

    def turn(self):
        pass

    def river(self):
        pass

    def __init__(self, uip, states=None):
        if states is None:
            self.states = [BLIND, PREFLOP, FLOP, TURN, RIVER]
        elif states == 'totalcasino':
            self.states = [PREFLOP, FLOP, RIVER, OPPONENT]
        elif states == 'casino-poker':
            self.states = [BLIND, FLOP]
        else:
            self.states = states

        self.uip = uip
        self.state = 0

    def map(self, s):
        state_name = self.states[s]
        if state_name == PREFLOP:
            return self.uip.is_preflop
        elif state_name == FLOP:
            return self.uip.is_flop
        elif state_name == TURN:
            return self.uip.is_turn
        elif state_name == RIVER:
            return self.uip.is_river
        elif state_name == OPPONENT:
            return self.uip.is_opponent
        return None

    def get_state(self):
        return self.states[self.state]

    def next_state(self):
        self.state = (self.state + 1) % len(self.states)

    def observe(self):
        state = self.map(self.state)
        if state is not None:
            is_state = False
            cards = ()
            while not is_state:
                t = state()
                is_state = t[0]
                cards = t[1:]
            self.next_state()
            return cards
        return ()

    def begin(self):
        for s in self.states:
            self.decide(s)

    def decide(self):
        if self.state == BLIND:
            self.blind()
            self.state = PREFLOP
        elif self.state == PREFLOP:
            self.preflop()
            self.state = FLOP
        elif self.state == FLOP:
            self.flop()
            self.state = TURN
        elif self.state == TURN:
            self.turn()
            self.state = RIVER
        elif self.state == RIVER:
            self.river()
            self.state = BLIND