import pickle
from scr.game import date_time


AGGREGATORS_DIR = "stats/aggregators/"


class StatsAggregator:
    def __init__(self, backup_every=25):
        self.stack_trend = []
        self.current_stake = 0
        self.stakes_history = []

        self.tables = []
        self.hand_balances = []
        self.normalized_hb = []
        self.evs = []
        self.outcomes = []

        self.player_river_combs = []
        self.opp_river_combs = []

        self.backup_every = backup_every
        self.hand_num = 0

    def backup(self, prefix='backup'):
        filename = AGGREGATORS_DIR + prefix + date_time() + ".sa"
        file = open(filename, 'wb')
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def hand_outcome(self, stack, hand_balance, outcome, table, ev, player_river_comb=None, opp_river_comb=None):
        self.stack_trend.append(stack)
        self.outcomes.append(outcome)
        self.hand_balances.append(hand_balance)
        self.tables.append(table)
        self.normalized_hb.append(round(hand_balance / self.current_stake))
        self.stakes_history.append(self.current_stake)
        self.player_river_combs.append(player_river_comb)
        self.evs.append(ev)
        self.opp_river_combs.append(opp_river_comb)
        self.hand_num += 1
        if self.hand_num % self.backup_every == 0:
            self.backup()

    def start_session(self, init_stake, init_balance):
        self.current_stake = init_stake
        self.stack_trend.append(init_balance)

    def end_session(self):
        self.backup("session")

    def stakes_changed(self, stake):
        self.current_stake = stake
    """
    def handle_message(self, message, val):
        if message == Messages.WHOLE_TABLE_MSG:
            self.tables.append(val)
        if message == Messages.HAND_EQUITY_MSG:
            self.equities.append(val)
        elif message == Messages.OUTCOME_STACK_MSG:
            self.stack_trend.append(val)
        elif message == Messages.EV_MSG:
            self.evs.append(val)
        elif message == Messages.OUTCOME_MSG:
            self.outcomes.append(val)
        elif message == Messages.HAND_BALANCE_MSG:
            self.hand_balances.append(val)
        elif message == Messages.CURRENT_STAKE_MSG:
            self.current_stake = val
            self.stakes_history.append(val)
        elif message == Messages.BOT_RIVER_COMB_MSG:
            self.player_river_combs.append(val)
        elif message == Messages.OPP_RIVER_COMB_MSG:
            self.opp_river_combs.append(val)
        elif message == Messages.STARTING_STACK_MSG:
            self.stack_trend.append(val)
        elif message == Messages.END_OF_SESSION_MSG:
            self.backup('session')
        else:
            self.change_num -= 1
        self.change_num += 1

        if self.change_num >= self.backup_every:
            self.backup()
            self.change_num = 0
    """
