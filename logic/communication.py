from abc import ABC, abstractmethod


class Outcomes:
    NQ   =  2
    WON  =  1
    DRAW =  0
    LOST = -1
    FOLD = -2

    @staticmethod
    def to_str(o):
        if o == Outcomes.WON:
            return "Won"
        elif o == Outcomes.DRAW:
            return "Draw"
        elif o == Outcomes.LOST:
            return "Lost"
        elif o == Outcomes.FOLD:
            return "Fold"
        elif o == Outcomes.NQ:
            return "Dealer doesn't qualify"


class Messages:
    HOLE_CARDS_MSG                = "msg:hole_cards"
    FLOP_CARDS_MSG                = "msg:flop_cards"
    TURN_CARDS_MSG                = "msg:turn_cards"
    RIVER_CARDS_MSG               = "msg:river_cards"
    BOT_RIVER_COMB_MSG            = "msg:bot_river_comb"
    OPP_RIVER_COMB_MSG            = "msg:opp_river_comb"
    HAND_EQUITY_MSG               = "msg:hand_equity"
    SLEEP_MSG                     = "msg:sleep"
    HAND_HISTOGRAM_MSG            = "msg:hand_histogram"
    COMB_EQUITY_AND_HISTOGRAM_MSG = "msg:comb_equity_and_histogram"
    DECISION_MADE_MSG             = "msg:decision"
    OPPONENT_CARDS_MSG            = "msg:opponent_cards"
    HAND_DEALT_MSG                = "msg:hand_dealt"
    OUTCOME_MSG                   = "msg:outcome"
    EV_MSG                        = "msg:ev"
    EVAL_TIME_MSG                 = "msg:eval_time"
    WHOLE_TABLE_MSG               = "msg:whole_table"
    OUTCOME_STACK_MSG             = "msg:outcome_stack"
    STARTING_STACK_MSG            = "msg:starting_stack"
    HAND_BALANCE_MSG              = "msg:hand_balance"
    END_OF_SESSION_MSG            = "msg:end_of_session"
    CURRENT_STAKE_MSG             = "msg:current_stake"


class MessageListener(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def handle_message(self, message, val):
        pass


class TaskExecutor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def exec(self, task, delay):
        pass