from scr.pokercmb import HighCard, Pair, TwoPair, Set, Straight, Flush, FullHouse, Quads, StraightFlush
import numpy as np
from scr.combinatorics import combinations, pair, pair_wr, c
from time import time
from scr.game import is_straight, gen_deck
from scr.game import deck_from, int2rank, suit2word, deck_wo_dead_cards, str2cards
from scr.basicfunc import eq
from scr.basicfunc import vstacker
from scr.basicfunc import nandiv
from scr.basicfunc import bincount3d
from scr.combinatorics import random_probe
import pickle


comb_find_funcs_desc = [StraightFlush.find,
                        Quads.find,
                        FullHouse.find,
                        Flush.find,
                        Straight.find,
                        Set.find,
                        TwoPair.find,
                        Pair.find,
                        HighCard.find]

lose, draw, win = 0, 1, 2


def tables_to_combs(probe, or_better=-1, five_of_a_kind=False):
    """
    :param probe:
    :param or_better:
    :param five_of_a_kind:
    :return:
    """

    tn, cn = probe.shape
    hand_suit_identity = -np.ones(shape=tn, dtype=np.int8)
    hand_rank_identity = np.zeros(shape=(tn, 13), dtype=np.int8)

    suits = np.arange(4)
    ranks = np.arange(13)
    prime_rank_map = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int64)
    prime_suit_map = np.array([2, 3, 5, 7], dtype=np.int64)

    tables_ranks = probe % 13
    tables_suits = probe // 13

    hrpi = np.prod(prime_rank_map[tables_ranks], axis=1, dtype=np.int64)
    hspi = np.prod(prime_suit_map[tables_suits], axis=1, dtype=np.int64)

    conseq_count = np.zeros(shape=tn, dtype=np.int8)
    straight_tops = np.zeros(shape=tn, dtype=np.int8)

    wheel_div = 2 * 3 * 5 * 7 * 41
    straight_tops[(hrpi % wheel_div) == 0] = -1

    for prime, rank in zip(prime_rank_map, ranks):
        single = prime
        pair = prime * prime
        set = prime * prime * prime
        quads = prime * prime * prime * prime
        fives = prime * prime * prime * prime * prime

        if five_of_a_kind:
            hand_rank_identity[:, rank] += (hrpi % fives == 0)
        hand_rank_identity[:, rank] += (hrpi % quads == 0)
        hand_rank_identity[:, rank] += (hrpi % set == 0)
        hand_rank_identity[:, rank] += (hrpi % pair == 0)
        has_rank = (hrpi % single == 0)
        hand_rank_identity[:, rank] += has_rank

        conseq_count += has_rank
        conseq_count[~has_rank] = 0
        # стрит начинается от 6: 0 - 2, 1 - 3, 2 - 4, > 3 - 5
        if rank > 3:
            straight_tops[conseq_count > 4] = rank

    for prime, suit in zip(prime_suit_map, suits):
        flush = prime * prime * prime * prime * prime
        hand_suit_identity[(hspi % flush) == 0] = suit

    only_flush_ranks = np.array(-np.sort(
        (-(tables_ranks + 1) *
         (tables_suits.T == hand_suit_identity).T),
        axis=1
    ), dtype=np.int64)

    HC = 0
    OP = 1
    TP = 2
    SE = 3
    ST = 4
    FL = 5
    FH = 6
    QU = 7
    SF = 8
    RF = 9
    FK = 10

    hand_combs = np.zeros(shape=tn, dtype=np.uint8)

    ranks2d = np.reshape(np.arange(1, 14), newshape=[1, 13])

    pairs_sparse = (hand_rank_identity == 2) * ranks2d
    pairs_sparse = -np.sort(-pairs_sparse, axis=1)
    pairs = (pairs_sparse[:, 0] > 0)
    hand_combs[pairs] = OP

    two_pairs = (pairs_sparse[:, 0] > 0) * (pairs_sparse[:, 1] > 0)
    hand_combs[two_pairs] = TP

    sets_sparse = (hand_rank_identity == 3) * ranks2d
    sets_sparse = -np.sort(-sets_sparse, axis=1)

    sets = sets_sparse[:, 0] > 0
    hand_combs[sets] = SE

    straight_ranks = straight_tops + 1
    straights = straight_ranks != 1
    hand_combs[straights] = ST

    flushes = hand_suit_identity != -1
    hand_combs[flushes] = FL

    pairs_num = np.sum(pairs_sparse > 0, axis=1)
    sets_num = np.sum(sets_sparse > 0, axis=1)
    full_houses = ((pairs_num > 0) & (sets_num > 0))

    # они не пересекаются, тк не может быть двух сетов и пары за столом одновременно (8 карта - макс 7)
    hand_combs[full_houses] = FH

    quads_sparse = (hand_rank_identity == 4) * ranks2d
    quads_sparse = np.max(quads_sparse, axis=1)
    quads = quads_sparse != 0
    hand_combs[quads] = QU

    straights_ri = [2 * 3 * 5 * 7 * 41,  # wheel
                    2 * 3 * 5 * 7 * 11,
                    3 * 5 * 7 * 11 * 13,
                    5 * 7 * 11 * 13 * 17,
                    7 * 11 * 13 * 17 * 19,
                    11 * 13 * 17 * 19 * 23,
                    13 * 17 * 19 * 23 * 29,
                    17 * 19 * 23 * 29 * 31,
                    19 * 23 * 29 * 31 * 37,
                    23 * 29 * 31 * 37 * 41]
    straight_conseq_tops = [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    royal_flush_ri = 23 * 29 * 31 * 37 * 41
    possible_straight_flushes = straights & flushes
    prime_zero_out = np.array([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int64)
    straight_flush_hrpi = np.prod(prime_zero_out[only_flush_ranks], axis=1, dtype=np.int64) * possible_straight_flushes

    straight_flush_ranks = np.zeros(shape=tn, dtype=np.int8)
    for rank_identity, top_rank in zip(straights_ri, straight_conseq_tops):
        straight_flush_ranks[((straight_flush_hrpi % rank_identity) == 0) * possible_straight_flushes] = top_rank

    straight_flush_ranks += 1
    straight_flushes = straight_flush_ranks != 1

    hand_combs[straight_flushes] = SF

    royal_flushes = straight_flush_ranks == 13
    hand_combs[royal_flushes] = RF

    if five_of_a_kind:
        fives_sparse = (hand_rank_identity == 5) * ranks2d
        fives_sparse = np.max(fives_sparse, axis=1)
        fives = fives_sparse > 0
        hand_combs[fives] = FK

    # or higher
    if or_better > 0:
        pairs_at = hand_combs == OP
        hand_combs[(pairs_sparse[:, 0] < (or_better + 1)) & pairs_at] = HC

    return hand_combs


class JacksOrBetter:
    def __init__(self, paytable=None):
        if paytable is None:
            self.paytable = np.array([0, 1, 2, 3, 4, 6, 9, 25, 50, 976])
        else:
            self.paytable = paytable
        self.hold_sets = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.bool)
        jacks = 9
        standard_deck = np.arange(52)
        self.tables = combinations(standard_deck, 5)
        self.combs = tables_to_combs(self.tables, or_better=jacks, five_of_a_kind=False)
        self.payouts = self.paytable[self.combs]
        self.values = self.payouts - 1

    def play_ranking(self, table):
        hold_set_evs = np.zeros(32, dtype=np.float64)
        histograms = np.zeros(shape=[32, 10], dtype=np.float64)
        for n in range(32):
            holds = self.hold_sets[n]
            discards = ~holds
            held_cards = table[holds]
            discarded_cards = table[discards]

            # столы, на которых лежат все оставленные карты и на которых нет сброшенных карт
            tables = (
                    (np.sum(np.isin(self.tables, held_cards), axis=1) == len(held_cards))
                    & (np.sum(np.isin(self.tables, discarded_cards), axis=1) == 0)
            )

            histograms[n] = np.bincount(self.combs[tables], minlength=10)
            histograms[n] = np.true_divide(histograms[n], np.sum(histograms[n]))
            ev = np.mean(self.values[tables])
            hold_set_evs[n] = ev
        sort = np.argsort(-hold_set_evs)
        return self.hold_sets[sort], hold_set_evs[sort], histograms[sort]


class VideoPokerEquityCalc:
    def __init__(self, pay_table=None, or_better=9, has_joker=True):
        self.has_joker = has_joker
        self.hold_sets = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.bool)

        if pay_table is None:
            """
                0 - No Combinations
                1 - One Pair
                2 - Two Pairs
                3 - Set
                4 - Straight
                5 - Flush
                6 - Full House
                7 - Quads
                8 - Straight Flush
                9 - Royal Flush
                10 - Five of a Kind
            """
            pay_table = np.array([0, 1, 2, 3, 5, 10, 15, 25, 50, 100, 500], dtype=np.int64)
        self.pay_table = pay_table
        standard_deck = np.arange(52)
        self.tables = combinations(standard_deck, 5)
        self.tables_ranks = self.tables % 13
        if has_joker:
            tn = c(52, 4)
            self.joker_tables = np.zeros(shape=(tn, 5), dtype=np.int8)
            self.joker_combs = np.zeros(shape=tn, dtype=np.int8)
            self.joker_payouts = np.zeros(shape=tn, dtype=np.int16)
            self.joker_tables[:, :4] = combinations(standard_deck, 4)
            for n in range(tn):
                table = self.joker_tables[n]
                joker_table = np.empty(shape=(52, 5), dtype=np.int8)
                joker_table[:, :4] = table[:4]
                joker_table[:, 4] = standard_deck
                combs = tables_to_combs(joker_table, or_better=or_better, five_of_a_kind=True)
                payouts = self.pay_table[combs]
                max_payout_index = np.argmax(payouts)
                max_payout = payouts[max_payout_index]
                best_payout_table = joker_table[max_payout_index]
                best_payout_comb = combs[max_payout_index]
                self.joker_tables[n, :] = best_payout_table
                self.joker_combs[n] = best_payout_comb
                self.joker_payouts[n] = max_payout
            self.joker_tables_ranks = self.joker_tables % 13
        self.combs = tables_to_combs(self.tables, or_better=or_better, five_of_a_kind=False)
        self.payouts = self.pay_table[self.combs]

    def best_play(self, table, magic_number=2):
        joker = table == 52
        magic_rank = magic_number - 2
        all_joker_tables = self.joker_tables[:, :4]
        if np.any(joker):
            # в любом случае джокер остается, без вариантов
            hold_sets = self.hold_sets[np.any(self.hold_sets & joker, axis=1)]
            histograms = np.zeros(shape=[hold_sets.shape[0], 11], dtype=np.int64)
            hold_set_evs = np.zeros(hold_sets.shape[0], dtype=np.float64)

            for n in range(len(hold_sets)):
                holds = hold_sets[n]
                discards = ~holds
                held_cards = table[holds]
                discarded_cards = table[discards]

                non_joker_held_cards = table[hold_sets[n] & ~joker]
                # просматриваются только столы с джокером, тк джокер уже в руке
                tables = (
                    (np.sum(np.isin(all_joker_tables, non_joker_held_cards), axis=1) == len(non_joker_held_cards))
                    & (np.sum(np.isin(all_joker_tables, discarded_cards), axis=1) == 0)
                )
                magic_mult = np.power(2, np.sum(self.joker_tables_ranks == magic_rank, axis=1))
                payouts = (magic_mult * tables * self.joker_payouts)[tables]
                values = payouts - 1
                hold_set_evs[n] = np.mean(values)
                histograms[n] = np.bincount(self.joker_combs[tables], minlength=11)
        else:
            hold_sets = self.hold_sets
            hold_set_evs = np.zeros(32, dtype=np.float64)

            histograms = np.zeros(shape=[32, 11], dtype=np.int64)
            for n in range(32):
                holds = hold_sets[n]
                discards = ~holds
                held_cards = table[holds]
                discarded_cards = table[discards]

                # столы, на которых лежат все оставленные карты и на которых нет сброшенных карт
                tables = (
                    (np.sum(np.isin(self.tables, held_cards), axis=1) == len(held_cards))
                    & (np.sum(np.isin(self.tables, discarded_cards), axis=1) == 0)
                )
                magic_mult = np.power(2, np.sum(self.tables_ranks == magic_rank, axis=1))
                payouts = (magic_mult * tables * self.payouts)[tables]
                values = payouts - 1
                # если в колоде есть джокер, существует вероятность его выпадения
                if self.has_joker:
                    # в этом случае предполагается, что набран один джокер
                    joker_tables = (
                        (np.sum(np.isin(all_joker_tables, held_cards), axis=1) == np.sum(holds))
                        & (np.sum(np.isin(all_joker_tables, discarded_cards), axis=1) == 0)
                    )
                    if joker_tables.size < 1:
                        ev = np.mean(values)
                        histograms[n] = np.bincount(self.combs[tables], minlength=11)
                    else:
                        magic_mult_joker = np.power(2, np.sum(self.joker_tables_ranks == magic_rank, axis=1))
                        joker_payouts = (magic_mult_joker * joker_tables * self.joker_payouts)[joker_tables]
                        joker_values = joker_payouts - 1
                        ev = float((np.sum(values) + np.sum(joker_values))) / (payouts.size + joker_payouts.size)
                        histograms[n] = (np.bincount(self.combs[tables], minlength=11) +
                                         np.bincount(self.joker_combs[joker_tables], minlength=11))
                else:
                    ev = np.mean(values)
                    histograms[n] = np.bincount(self.combs[tables], minlength=11)
                hold_set_evs[n] = ev

        sort = np.argsort(-hold_set_evs, kind='mergesort')
        return hold_sets[sort], hold_set_evs[sort], histograms[sort]

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

"""
ss = time()
ec = pickle.load(open('double_poker.vpc', 'rb'))
ee = time()
print(ee - ss)
ss = time()
table = np.array([10, 9, 8, 32, 0])
bp = ec.best_play(table, 3)
ee = time()
print(ee - ss)
print(bp[0])
print("-~*~-")
print(bp[1])
print("-~*~-")
print(bp[2])
"""
"""
table = np.array([0, 12, 32, 42, 4])
f = open('double_poker.vpc', 'rb')
ec = pickle.load(f)
f.close()
ss = time()
print(ec.best_play(table))
ee = time()
print(ee - ss)
"""


class LiveEquityCalc:
    def __init__(self, holes):
        self.holes = holes
        self.deck = deck_wo_dead_cards(holes)
        self.table_cards = combinations(self.deck, 5)
        self.tn = c(len(self.deck), 5)
        self.pn = len(holes)
        self.players_tables = np.empty(shape=(self.pn, self.tn, 7), dtype=np.int8)
        self.players_tables[:, :, :5] = self.table_cards
        self.players_tables[:, :, 5:] = np.reshape(self.holes, newshape=(self.pn, 1, 2))

        suits = np.arange(4)
        ranks = np.arange(13)
        prime_rank_map = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int64)
        prime_suit_map = np.array([2, 3, 5, 7], dtype=np.int64)

        self.tables_ranks = self.players_tables % 13
        self.tables_suits = self.players_tables // 13

        self.hrpi = np.prod(prime_rank_map[self.tables_ranks], axis=2, dtype=np.int64)
        self.hspi = np.prod(prime_suit_map[self.tables_suits], axis=2, dtype=np.int64)

        self.hand_suit_identity = -np.ones(shape=(self.pn, self.tn), dtype=np.int8)
        self.hand_rank_identity = np.zeros(shape=(self.pn, self.tn, 13), dtype=np.int8)

        conseq_count = np.zeros(shape=(self.pn, self.tn), dtype=np.int8)
        self.straight_tops = np.zeros(shape=(self.pn, self.tn), dtype=np.int8)

        wheel_div = 2 * 3 * 5 * 7 * 41
        self.straight_tops[(self.hrpi % wheel_div) == 0] = -1
        for prime, rank in zip(prime_rank_map, ranks):
            single = prime
            pair = prime * prime
            set = prime * prime * prime
            quads = prime * prime * prime * prime

            self.hand_rank_identity[:, :, rank] += (self.hrpi % quads == 0)
            self.hand_rank_identity[:, :, rank] += (self.hrpi % set == 0)
            self.hand_rank_identity[:, :, rank] += (self.hrpi % pair == 0)
            has_rank = (self.hrpi % single == 0)
            self.hand_rank_identity[:, :, rank] += has_rank

            conseq_count += has_rank
            conseq_count[~has_rank] = 0
            # стрит начинается от 6: 0 - 2, 1 - 3, 2 - 4, > 3 - 5
            if rank > 3:
                self.straight_tops[conseq_count > 4] = rank

        for prime, suit in zip(prime_suit_map, suits):
            flush = prime * prime * prime * prime * prime
            self.hand_suit_identity[(self.hspi % flush) == 0] = suit

        only_flush_ranks = np.array(-np.sort(
            (-(self.tables_ranks + 1) *
             (self.tables_suits == np.reshape(self.hand_suit_identity, newshape=(self.pn, self.tn, 1)))),
            axis=2
        ), dtype=np.int64)

        HC = 0
        OP = 1
        TP = 2
        SE = 3
        ST = 4
        FL = 5
        FH = 6
        QU = 7
        SF = 8
        RF = 9

        comb_rank_mult = 10000000000
        fst_card_mult  =   100000000
        snd_card_mult  =     1000000
        trd_card_mult  =       10000
        frt_card_mult  =         100
        fft_card_mult  =           1

        self.hand_uniq_classifiers = np.zeros(shape=(self.pn, self.tn), dtype=np.int64)

        ranks3d = np.reshape(np.arange(1, 14), newshape=[1, 1, 13])
        kickers_sparse = (self.hand_rank_identity == 1) * ranks3d
        kickers_sparse = np.array(-np.sort(-kickers_sparse, axis=2), dtype=np.int64)

        self.hand_uniq_classifiers += (
            HC * comb_rank_mult +
            fst_card_mult * kickers_sparse[:, :, 0] +
            snd_card_mult * kickers_sparse[:, :, 1] +
            trd_card_mult * kickers_sparse[:, :, 2] +
            frt_card_mult * kickers_sparse[:, :, 3] +
            fft_card_mult * kickers_sparse[:, :, 4]
        )

        pairs_sparse = (self.hand_rank_identity == 2) * ranks3d
        pairs_sparse = np.array(-np.sort(-pairs_sparse, axis=2), dtype=np.int64)
        pairs = (pairs_sparse[:, :, 0] > 0)
        self.hand_uniq_classifiers[pairs] = 0
        # если первая пара - ноль, то будет ноль
        self.hand_uniq_classifiers += (
             pairs * (
                OP * comb_rank_mult +
                fst_card_mult * pairs_sparse[:, :, 0] +
                snd_card_mult * pairs_sparse[:, :, 0] +
                trd_card_mult * kickers_sparse[:, :, 0] +
                frt_card_mult * kickers_sparse[:, :, 1] +
                fft_card_mult * kickers_sparse[:, :, 2]
            )
        )

        # очистка от предыдущих пар в местах, где есть 2 пары
        two_pairs = (pairs_sparse[:, :, 0] > 0) * (pairs_sparse[:, :, 1] > 0)
        self.hand_uniq_classifiers *= ~two_pairs

        self.hand_uniq_classifiers += (
            two_pairs * (
                TP * comb_rank_mult +
                fst_card_mult * pairs_sparse[:, :, 0] +
                snd_card_mult * pairs_sparse[:, :, 0] +
                trd_card_mult * pairs_sparse[:, :, 1] +
                frt_card_mult * pairs_sparse[:, :, 1] +
                fft_card_mult * kickers_sparse[:, :, 0]
            )
        )

        sets_sparse = (self.hand_rank_identity == 3) * ranks3d
        sets_sparse = np.array(-np.sort(-sets_sparse, axis=2), dtype=np.int64)

        # очистка от предыдущих комбинаций в местах, где есть сет
        sets = sets_sparse[:, :, 0] > 0
        self.hand_uniq_classifiers *= ~sets

        self.hand_uniq_classifiers += (
            sets * (
                SE * comb_rank_mult +
                fst_card_mult * sets_sparse[:, :, 0] +
                snd_card_mult * sets_sparse[:, :, 0] +
                trd_card_mult * sets_sparse[:, :, 0] +
                frt_card_mult * kickers_sparse[:, :, 0] +
                fft_card_mult * kickers_sparse[:, :, 1]
            )
        )

        # очистка от предыдущих комбинаций в местах, где есть стрит
        straight_ranks = self.straight_tops + 1
        straights = straight_ranks != 1
        self.hand_uniq_classifiers *= ~straights

        self.hand_uniq_classifiers += (
            straights * (
                ST * comb_rank_mult +
                fst_card_mult * straight_ranks
            )
        )

        # очистка от предыдущих комбинаций в местах, где есть флеш
        flushes = self.hand_suit_identity != -1
        self.hand_uniq_classifiers[flushes] = 0

        self.hand_uniq_classifiers += (
            flushes * (
                FL * comb_rank_mult +
                fst_card_mult * only_flush_ranks[:, :, 0] +
                snd_card_mult * only_flush_ranks[:, :, 1] +
                trd_card_mult * only_flush_ranks[:, :, 2] +
                frt_card_mult * only_flush_ranks[:, :, 3] +
                fft_card_mult * only_flush_ranks[:, :, 4]
            )
        )

        # очистка от предыдущих комбинаций в местах, где есть фулхаус
        pairs_num = np.sum(pairs_sparse > 0, axis=2)
        sets_num = np.sum(sets_sparse > 0, axis=2)
        set_n_set_fhs = sets_num > 1
        set_n_pair_fhs = (pairs_num > 0) & (sets_num == 1)
        full_houses = set_n_pair_fhs | set_n_set_fhs

        # они не пересекаются, тк не может быть двух сетов и пары за столом одновременно (8 карта - макс 7)
        self.hand_uniq_classifiers[full_houses] = 0
        self.hand_uniq_classifiers += (
            set_n_pair_fhs * (
                FH * comb_rank_mult +
                fst_card_mult * sets_sparse[:, :, 0] +
                snd_card_mult * sets_sparse[:, :, 0] +
                trd_card_mult * sets_sparse[:, :, 0] +
                frt_card_mult * pairs_sparse[:, :, 0] +
                fft_card_mult * pairs_sparse[:, :, 0]
            )
        )
        self.hand_uniq_classifiers += (
            set_n_set_fhs * (
                FH * comb_rank_mult +
                fst_card_mult * sets_sparse[:, :, 0] +
                snd_card_mult * sets_sparse[:, :, 0] +
                trd_card_mult * sets_sparse[:, :, 0] +
                frt_card_mult * sets_sparse[:, :, 1] +
                fft_card_mult * sets_sparse[:, :, 1]
            )
        )

        quads_sparse = (self.hand_rank_identity == 4) * ranks3d
        quads_sparse = np.array(np.max(quads_sparse, axis=2), dtype=np.int64)

        max_single_sparse = kickers_sparse[:, :, 0]
        max_pairs_sparse = pairs_sparse[:, :, 0]
        max_sets_sparse = sets_sparse[:, :, 0]
        max_quads_kicker = (
            ((max_single_sparse > max_pairs_sparse) * max_single_sparse) +
            ((max_single_sparse < max_pairs_sparse) * max_pairs_sparse)
        )
        max_quads_kicker = (
                ((max_quads_kicker > max_sets_sparse) * max_quads_kicker) +
                ((max_quads_kicker < max_sets_sparse) * max_sets_sparse)
        )

        quads = quads_sparse != 0
        self.hand_uniq_classifiers[quads] = 0
        self.hand_uniq_classifiers += (
            quads * (
                QU * comb_rank_mult +
                fst_card_mult * quads_sparse +
                snd_card_mult * quads_sparse +
                trd_card_mult * quads_sparse +
                frt_card_mult * quads_sparse +
                fft_card_mult * max_quads_kicker
            )
        )

        straights_ri = [2 * 3 * 5 * 7 * 41,  # wheel
                        2 * 3 * 5 * 7 * 11,
                        3 * 5 * 7 * 11 * 13,
                        5 * 7 * 11 * 13 * 17,
                        7 * 11 * 13 * 17 * 19,
                        11 * 13 * 17 * 19 * 23,
                        13 * 17 * 19 * 23 * 29,
                        17 * 19 * 23 * 29 * 31,
                        19 * 23 * 29 * 31 * 37,
                        23 * 29 * 31 * 37 * 41]
        straight_conseq_tops = [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        royal_flush_ri = 23 * 29 * 31 * 37 * 41
        possible_straight_flushes = straights & flushes
        prime_zero_out = np.array([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int64)
        straight_flush_hrpi = np.prod(prime_zero_out[only_flush_ranks], axis=2, dtype=np.int64) * possible_straight_flushes

        straight_flush_ranks = np.zeros(shape=[self.pn, self.tn], dtype=np.int8)
        for rank_identity, top_rank in zip(straights_ri, straight_conseq_tops):
            straight_flush_ranks[((straight_flush_hrpi % rank_identity) == 0) * possible_straight_flushes] = top_rank

        straight_flush_ranks += 1
        straight_flushes = straight_flush_ranks != 1

        self.hand_uniq_classifiers[straight_flushes] = 0
        self.hand_uniq_classifiers += (
            straight_flushes * (
                SF * comb_rank_mult +
                fst_card_mult * straight_flush_ranks
            )
        )

        royal_flushes = straight_flush_ranks == 13
        self.hand_uniq_classifiers[royal_flushes] = 0
        self.hand_uniq_classifiers += royal_flushes * RF * comb_rank_mult

        # самые главные подсчеты
        self.hand_comb_matrix = np.zeros(shape=[self.pn, self.tn], dtype=np.int8)
        self.hand_comb_matrix[pairs] = 1
        self.hand_comb_matrix[two_pairs] = 2
        self.hand_comb_matrix[sets] = 3
        self.hand_comb_matrix[straights] = 4
        self.hand_comb_matrix[flushes] = 5
        self.hand_comb_matrix[full_houses] = 6
        self.hand_comb_matrix[quads] = 7
        self.hand_comb_matrix[straight_flushes] = 8
        self.hand_comb_matrix[royal_flushes] = 9

        self.huc_max = np.max(self.hand_uniq_classifiers, axis=0)

        self.init_tn = self.tn
        self.init_table_cards = self.table_cards
        self.init_win_matrix = self.hand_uniq_classifiers == self.huc_max
        self.init_tables_win_comb = np.max(self.init_win_matrix * self.hand_comb_matrix, axis=0)
        self.win_matrix = self.init_win_matrix
        self.tables_win_comb = self.init_tables_win_comb
        """
        # casino expected profit off each hand bet on each round
        # blind, preflop, flop, turn
        self.casino_hep = np.zeros(shape=[4, self.pn], dtype=np.float64)
        # casino expected profit off each combination bet on each round
        self.casino_cep = np.zeros(shape=[4, 10], dtype=np.float64)
        self.table = -np.ones(shape=5, dtype=np.int8)
        """

    def clear_table(self):
        self.table_cards = self.init_table_cards
        self.win_matrix = self.init_win_matrix
        self.tables_win_comb = self.init_tables_win_comb
        self.tn = self.init_tn

    def set_table(self, t):
        hits = np.sum(np.isin(self.table_cards, t), axis=1)
        selection = hits == t.size
        self.tn = np.sum(selection)
        self.table_cards = self.table_cards[selection, :]
        self.win_matrix = self.win_matrix[:, selection]
        self.tables_win_comb = self.tables_win_comb[selection]

    def win_draw_odds(self):
        """
        :return: win or draw real odds
        """
        player_win_or_draw_n = np.sum(self.win_matrix, axis=1)
        odds = player_win_or_draw_n / self.tn
        return nandiv(1, odds), odds

    def win_comb_odds(self):
        """
        :return: hand combination real odds
        """
        counts = np.bincount(self.tables_win_comb, minlength=10)
        coeff = counts / self.tn
        return nandiv(1, coeff), coeff

    def win_draw_value(self, casino_odds):
        real_odds = self.win_draw_odds()[0]
        value = nandiv(casino_odds - real_odds, casino_odds)
        return value, real_odds

    def win_comb_value(self, casino_odds):
        real_odds = self.win_comb_odds()[0]
        value = nandiv(casino_odds - real_odds, casino_odds)
        return value, real_odds

    def casino_profit_matrix(self, casino_odds, current_stakes):
        profit_matrix = ((self.win_matrix.transpose() * -current_stakes * casino_odds) +
                         (~self.win_matrix.transpose() * current_stakes)).transpose()
        return profit_matrix

    def casino_comb_profit(self, casino_odds, stakes=None):
        if stakes is None:
            stakes = np.ones(shape=10, dtype=np.float64)
        counts = np.bincount(self.tables_win_comb, minlength=10)
        m = np.eye(10, dtype=np.bool)
        casino_loss = -(m * counts * stakes * casino_odds)
        casino_profit = ~m * counts * stakes
        casino_profit_matrix = casino_loss + casino_profit

        casino_profit_vector = np.sum(casino_profit_matrix, axis=1)

        return casino_profit_vector, np.mean(casino_profit_vector)

    def casino_hand_profit(self, casino_odds, current_stakes=None):
        """
        когда рука выигрывает - это минус для казино, тк они должны выплатить ставку, умноженную на
        множитель, игрокам. но остальные руки проигрывают (бывает, выигрывает несколько рук),
        и разница между двумя показателями забирается казино.
        :param casino_odds: множители, определенные казино
        :param current_stakes: ставка в данном раунде торгов (т.е. пока один и тот же множитель)
        :return: среднестатистический баланс казино в случае победы каждого отдельного игрока и
                 ожидаемое значение казино
        """
        if current_stakes is None:
            current_stakes = np.ones(shape=self.pn, dtype=np.float64)

        profit_matrix = ((self.win_matrix.transpose() * -current_stakes * casino_odds)  +
                         (~self.win_matrix.transpose() * current_stakes)).transpose()

        table_profit_vector = np.sum(profit_matrix, axis=0)
        mean_casino_profit = np.empty(shape=self.pn, dtype=np.float64)

        for player in range(self.pn):
            player_wins = self.win_matrix[player, :]
            mean_casino_profit[player] = np.mean(table_profit_vector[player_wins])

        return mean_casino_profit, np.sum(table_profit_vector) / self.tn

    @staticmethod
    def casino_blind_hand_profit(casino_odds, stakes=None, pn=6):
        if stakes is None:
            stakes = np.ones(shape=pn, dtype=np.float64)
        casino_loss = -(np.eye(pn, dtype=np.float64) * casino_odds) * stakes
        casino_profit = ~np.eye(pn, dtype=np.bool) * stakes
        casino_profit_matrix = casino_loss + casino_profit
        casino_profit_vector = np.sum(casino_profit_matrix, axis=1) / pn
        return casino_profit_vector, np.mean(casino_profit_vector)

    @staticmethod
    def casino_blind_comb_profit(casino_odds, stakes=None, comb_freq=None):
        if comb_freq is None:
            comb_freq = np.array([0.000533 ,  0.1612254,  0.3050534,  0.139631 ,  0.1658926,
                                  0.1080948,  0.1088776,  0.008945 ,  0.001568 ,  0.0001792],
                                 dtype=np.float64)
        if stakes is None:
            stakes = np.ones(shape=10, dtype=np.float64)
        casino_loss = -(np.eye(10, dtype=np.float64) * casino_odds) * stakes
        casino_profit = ~np.eye(10, dtype=np.bool) * stakes
        casino_profit_matrix = casino_loss + casino_profit
        casino_profit_vector = np.sum(casino_profit_matrix, axis=1) * comb_freq
        return casino_profit_vector, np.mean(casino_profit_vector)


class MHEqCalc:
    def __init__(self, hole_cards):
        self.hole_cards = hole_cards
        self.deck = deck_wo_dead_cards(hole_cards)
        self.table_cards = combinations(self.deck, 5)
        self.tn = c(len(self.deck), 5)
        self.pn = len(hole_cards)
        self.players_tables = np.empty(shape=(self.pn, self.tn, 7), dtype=np.int8)
        self.players_tables[:, :, :5] = self.table_cards
        self.players_tables[:, :, 5:] = np.reshape(self.hole_cards, newshape=(self.pn, 1, 2))

        self.tables_ranks = self.players_tables % 13
        self.tables_suits = self.players_tables // 13

        start = time()
        self.hand_rank_identity = bincount3d(self.tables_ranks, 13)
        end = time()
        print(end - start)


hole_cards = "Jd 5d 2c 6h Qc 4h Ac Ks 8h Tc Qs 9c"
MHEqCalc(str2cards(hole_cards).reshape([6, 2]))
LiveEquityCalc(str2cards(hole_cards).reshape([6, 2]))


def test():
    hole_cards = "Jd 5d 2c 6h Qc 4h Ac Ks 8h Tc Qs 9c"
    flop = "Qh Jc Kc"
    casino_hand_odds = np.array([5.75, 25.00, 6.65, 2.08, 3.40, 16.50])
    casino_comb_odds = np.array([0.00, 6.10, 4.98, 5.30, 2.24, 40.00, 16.00, 250.00, 0.00, 0.00])
    hole_cards = str2cards(hole_cards).reshape([6, 2])
    flop = str2cards(flop)
    eqc = LiveEquityCalc(hole_cards)
    eqc.set_table(flop)
    hand_values = eqc.win_draw_value(casino_hand_odds)
    print("Real hand odds: ")
    print(hand_values[1])
    print("Hand values: ")
    print(hand_values[0])
    print("***")
    comb_vals = eqc.win_comb_value(casino_comb_odds)
    print("Real comb odds: ")
    print(comb_vals[1])
    print("Comb values: ")
    print(comb_vals[0])


def straight_masks():
    return np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ], dtype=np.bool)


def probe_combs(probe):
    pn = probe.shape[0]
    cn = probe.shape[1]

    probe_ranks = probe % 13
    probe_suits = probe // 13

    ranks = np.arange(1, 14)

    hand_rank_count = np.zeros(shape=[pn, 13], dtype=np.int8)
    hand_suit_count = np.zeros(shape=[pn, 4], dtype=np.int8)
    players = np.arange(pn, dtype=np.int8)
    for n in range(cn):
        hand_rank_count[players, probe_ranks[:, n]] += 1
        hand_suit_count[players, probe_suits[:, n]] += 1
    # kickers = (-np.sort((hand_rank_count == 1) * -ranks))[:, :7]
    pairs = (-np.sort((hand_rank_count == 2) * -ranks))[:, :3]
    sets = (-np.sort((hand_rank_count == 3) * -ranks))[:, :2]
    quads = np.max((hand_rank_count == 4) * ranks, axis=1)

    players_combs = np.zeros(shape=pn, dtype=np.int8)
    # pairs
    one_pair = pairs[:, 0] != 0
    players_combs[one_pair] = 1
    # two pairs
    players_combs[one_pair & (pairs[:, 1] != 0)] = 2
    # sets
    one_set = sets[:, 0] != 0
    players_combs[one_set] = 3
    # straights
    straights = np.zeros(shape=pn, dtype=np.bool)
    str_masks = straight_masks()
    for player in range(pn):
        straights[player] = np.any(np.sum(str_masks & (hand_rank_count[player, :] > 0), axis=1) == 5)
    players_combs[straights] = 4
    # flushes
    flush_suits = hand_suit_count > 4
    flushes = np.any(flush_suits, axis=1)
    players_combs[flushes] = 5
    # full houses
    players_combs[(one_pair & one_set) | (one_set & sets[:, 1] != 0)] = 6
    # quads
    players_combs[quads != 0] = 7
    # straight flush
    if np.any(flushes & straights):
        straight_flushes = np.zeros(shape=pn, dtype=np.bool)
        royal_flushes = np.zeros(shape=pn, dtype=np.bool)
        player_flush_suits = np.max(flush_suits * np.array([1, 2, 3, 4], dtype=np.int8), axis=1) - 1
        player_flush_ranks = np.zeros_like(hand_rank_count, dtype=np.bool)
        for player in range(pn):
            if flushes[player] & straights[player]:
                player_flush_ranks[player, probe_ranks[player, (probe_suits[player, :] == player_flush_suits[player])]] = True
                straight_flushes[player] = np.any(np.sum(str_masks[:-1, :] & player_flush_ranks[player, :], axis=1) == 5)
                royal_flushes[player] = np.sum(str_masks[-1, :] & player_flush_ranks[player, :]) == 5
        players_combs[straight_flushes] = 8
        players_combs[royal_flushes] = 9
    return players_combs


def max_table_comb_odds(probes=1000):
    count = np.zeros(shape=10, dtype=np.int64)
    for n in range(probes):
        count[np.max(probe_combs(random_probe()))] += 1
    return nandiv(1, count / probes)



"""
5000000 probes:

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
"""


def eval7card(hands):
    """
    Каждая рука имеет уникальное 32-битное обозначение.
    | Комбинация 5 |           Биты комбинации 13            |     Биты старшинства кикера 13       |
    |00|01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|
    | 0| 0| 0| 0| 0| - хай карта
    | 0| 0| 0| 0| 1| - пара
    | 0| 0| 0| 1| 0| - две пары
    | 0| 0| 0| 1| 1| - сет
    | 0| 0| 1| 0| 0| - стрит
    | 0| 0| 1| 0| 1| - флеш
    | 0| 0| 1| 1| 0| - фулл-хаус
    | 0| 0| 1| 1| 1| - каре
    | 0| 1| 0| 0| 0| - стрит-флеш
    | 1| 0| 0| 0| 0| - роял флеш

    Смещения:
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
     1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
     2147483648]
    :param hands: numpy array 8 bit ints
    :return:
    """
    nh, nc = hands.shape

    bitmap = np.zeros(shape=(nh, 52), dtype=np.bool)
    for n in range(hands.shape[1]):
        bitmap[:, hands[:, n]] = True
    # bitmap ready

    hearts = bitmap[:, :13]
    diamonds = bitmap[:, 13:26]
    spades = bitmap[:, 26:39]
    clubs = bitmap[:, 39:]

    rank_count = hearts + diamonds + spades + clubs

    kicker_offsets = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=np.uint32)
    main_offsets = np.array([16384, 32768, 65536, 131072, 262144, 524288, 1048576,
                             2097152, 4194304, 8388608, 16777216, 33554432, 67108864], dtype=np.uint32)
    comb_offsets = np.array([134217728, 268435456, 536870912, 1073741824, 2147483648], dtype=np.uint32)

    pair_offset = comb_offsets[0]
    two_pair_offset = comb_offsets[1]
    set_offset = comb_offsets[0] + comb_offsets[1]
    straight_offset = comb_offsets[2]
    flush_offset = comb_offsets[2] + comb_offsets[0]
    full_house_offset = comb_offsets[2] + comb_offsets[1]
    four_of_a_kind_offset = comb_offsets[2] + comb_offsets[1] + comb_offsets[0]
    straight_flush_offset = comb_offsets[3]
    royal_flush_offset = comb_offsets[4]

    """
    pairs_at = rank_count == 2

    pairs_count = np.count_nonzero(rank_count == 2, axis=1)
    pairs_ranks = pairs_at * ranks

    one_pairs = pairs_count == 1
    pairs_ranks

    sets_count = np.count_nonzero(rank_count == 3, axis=1)
    fours_count = np.count_nonzero(rank_count == 4, axis=1)
    """
    ranks = np.arange(13)
    indexing = np.arange(nh)

    # если за столом есть флеш, то нет вариантов на большую комбинацию, кроме стрит-флеша и роял-флеша.
    hearts_flushes = np.count_nonzero(hearts, axis=1) > 4
    diamonds_flushes = np.count_nonzero(diamonds, axis=1) > 4
    spades_flushes = np.count_nonzero(spades, axis=1) > 4
    clubs_flushes = np.count_nonzero(clubs, axis=1) > 4
    # двух флешей в семи картах не бывает.
    flushes_at = hearts_flushes | diamonds_flushes | spades_flushes | clubs_flushes
    # я отделяю ранги флеш карт от остальных. масть и оставшиеся ранги не имеют значения.
    flush_ranks = hearts
    del hearts
    del hearts_flushes
    flush_ranks[diamonds_flushes] = diamonds[diamonds_flushes]
    del diamonds
    del diamonds_flushes
    flush_ranks[spades_flushes] = spades[spades_flushes]
    del spades
    del spades_flushes
    flush_ranks[clubs_flushes] = clubs[clubs_flushes]
    del clubs
    del clubs_flushes

    flush_ranks = flush_ranks[flushes_at]
    # индексация позволяет сохранить естественный порядок и не обрабатывать лишние руки.
    flush_indexing = indexing[flushes_at]

    indexing = indexing[~flushes_at]
    rank_count = rank_count[~flushes_at]

    del flushes_at

    # поиск стрит-флешей
    str_masks = straight_masks()
    top_straight_ranks = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    straight_flushes_top_ranks = np.zeros(shape=flush_indexing.shape, dtype=np.int8)
    straight_flushes_at = np.zeros(shape=straight_flushes_top_ranks.shape, dtype=np.bool)

    for mask, rank in zip(str_masks, top_straight_ranks):
        at = np.count_nonzero(flush_ranks & mask, axis=1) == 5
        straight_flushes_at = straight_flushes_at | at
        straight_flushes_top_ranks[at] = rank

    royal_flushes_at = straight_flushes_top_ranks == 12

    # стрит флеш и роял флеш допускаются как разные комбинации
    only_straight_flushes = straight_flushes_at & (~royal_flushes_at)

    straight_flush_indexing = flush_indexing[only_straight_flushes]                 # индексы стрит флешей
    straight_flushes_top_ranks = straight_flushes_top_ranks[only_straight_flushes]  # ранги стрит флешей
    del only_straight_flushes

    royal_flush_indexing = flush_indexing[royal_flushes_at]                         # индексы роял флешей

    # отчленение роял флешей и стрит флешей от флешей
    wo_straight_flushes = ~straight_flushes_at
    flush_indexing = flush_indexing[wo_straight_flushes]                            # индексы флешей
    flush_ranks = flush_ranks[wo_straight_flushes]
    del wo_straight_flushes

    flush_ranks = -flush_ranks * ranks
    # убывающая сортировка, чтобы набрать самый большой кикер из 5-ти карт
    flush_ranks.sort()
    flush_ranks = -flush_ranks[:5]
    flush_kicker_sums = np.sum(kicker_offsets[flush_ranks], axis=1)                 # кикеры флешей (уже готовая сумма)
    del flush_ranks



class EquityCalc:
    def __init__(self, deck, table, dealer_qualifies_from=None):
        deck.sort()
        table.sort()

        self.deck = deck
        self.table = table

        hand_cases = pair_wr(deck, len(table) + 2)
        hand_cases[:, 2:] = table

        hand_cases_ranks = hand_cases % 13
        hand_cases_suits = hand_cases // 13

        suits = np.arange(4)
        ranks = np.arange(13)
        prime_number_map = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int64)

        cases_rank_map = prime_number_map[hand_cases_ranks]
        cases_suit_map = prime_number_map[hand_cases_suits]

        hand_rank_identifiers = np.prod(cases_rank_map, axis=1, dtype=np.int64)
        hand_suit_identifiers = np.prod(cases_suit_map, axis=1, dtype=np.int32)

        N = hand_cases.shape[0]

        straight_counts = np.zeros(shape=N, dtype=np.int8)
        n = 0
        # характеристика раздачи
        # [ количество пар, количество сетов, количество четверок, стрит, флеш ]
        flush_suit = -1
        identity_matrix = np.zeros(shape=[N, 5], dtype=np.int8)

        for suit, prime in zip(suits, prime_number_map[:4]):
            suit_div = prime * prime * prime * prime * prime
            flushes = (hand_suit_identifiers % suit_div) == 0
            if np.any(flushes):
                identity_matrix[:, 4] += flushes
                flush_suit = suit
                break

        for rank, prime in zip(ranks, prime_number_map):
            pair_div = prime * prime
            set_div = prime * prime * prime
            quads_div = prime * prime * prime * prime
            quads = (hand_rank_identifiers % quads_div) == 0
            sets = ((hand_rank_identifiers % set_div) == 0) & ~quads
            pairs = ((hand_rank_identifiers % pair_div) == 0) & ~sets

            identity_matrix[:, 0] += pairs
            identity_matrix[:, 1] += sets
            identity_matrix[:, 2] += quads

            prime_locations = (hand_rank_identifiers % prime) == 0
            straight_counts[prime_locations] += 1
            straight_counts[~prime_locations] = 0
            n += 1
            if n > 3:
                identity_matrix[:, 3] += straight_counts > 4

        wheel_div = 2 * 3 * 5 * 7 * 41
        identity_matrix[((hand_rank_identifiers % wheel_div) == 0) & (hand_rank_identifiers >= 8160), 3] = 1

        hand_combinations = np.zeros(shape=N, dtype=np.int8)
        # pairs
        hand_combinations[identity_matrix[:, 0] == 1] = 1
        # two pairs
        hand_combinations[identity_matrix[:, 0] > 1] = 2
        # sets
        hand_combinations[identity_matrix[:, 1] == 1] = 3
        # straights
        straights_mask = identity_matrix[:, 3] > 0
        hand_combinations[straights_mask] = 4
        # flushes
        if flush_suit > -1:
            flushes_mask = identity_matrix[:, 4] > 0
            hand_combinations[flushes_mask] = 5
        # full houses
        hand_combinations[((identity_matrix[:, 0] > 0) & (identity_matrix[:, 1] > 0)) | (identity_matrix[:, 1] > 1)] = 6
        # quads
        hand_combinations[identity_matrix[:, 2] > 0] = 7
        # straight flushes
        if flush_suit > -1:
            straights_n_flushes = straights_mask & flushes_mask
            if np.any(straights_n_flushes):
                snf_ranks = hand_cases_ranks[straights_n_flushes]
                snf_suits = hand_cases_suits[straights_n_flushes]

                snf_ranks[snf_suits != flush_suit] = -1
                snf_ranks.sort()

                prime_number_map = np.array(
                    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                     1],
                    dtype=np.int64)
                cases_rank_map = prime_number_map[snf_ranks]
                opp_flush_hri = np.prod(cases_rank_map, axis=1)

                sf_top_rank = np.zeros(shape=snf_ranks.shape[0], dtype=np.int8)
                sf_top_rank[((opp_flush_hri % wheel_div) == 0) & (opp_flush_hri >= 8160)] = -1

                straight_counts = np.zeros(shape=snf_ranks.shape[0], dtype=np.int8)
                n = 0
                for rank, prime in zip(ranks, prime_number_map):
                    prime_locations = (opp_flush_hri % prime) == 0
                    straight_counts[prime_locations] += 1
                    straight_counts[~prime_locations] = 0
                    n += 1
                    if n > 3:
                        sf_top_rank[straight_counts > 4] = rank
                        if rank == 12:
                            royal_flushes = np.copy(straights_n_flushes)
                            royal_flushes[straights_n_flushes] = sf_top_rank == 12
                            hand_combinations[royal_flushes] = 9

                self.str_fl_tops = np.zeros(shape=N, dtype=np.int8)
                self.str_fl_tops[straights_n_flushes] = sf_top_rank
                hand_combinations[((self.str_fl_tops > 3) | (self.str_fl_tops == -1)) & (hand_combinations < 9)] = 8
            else:
                self.str_fl_tops = None
        else:
            self.str_fl_tops = None

        if dealer_qualifies_from == '44':
            # рука дилера играет только в том случае, если она пара четверок или больше
            # в остальных случаях она проигрывает, даже если комбинация игрока хуже
            pair_of_threes = 3 * 3
            pair_of_deuces = 2 * 2
            # пары двоек и троек
            pairs_of_deuces_or_threes = (
                    (((hand_rank_identifiers % pair_of_deuces) == 0) | ((hand_rank_identifiers % pair_of_threes) == 0)) &
                    (hand_combinations == 1)
            )
            high_cards = hand_combinations == 0

            dealer_doesnt_qualify = pairs_of_deuces_or_threes | high_cards
            dealer_doesnt_qualify_count = np.sum(dealer_doesnt_qualify)

            dealer_qualifies = ~dealer_doesnt_qualify
            hand_rank_identifiers = hand_rank_identifiers[dealer_qualifies]
            hand_suit_identifiers = hand_suit_identifiers[dealer_qualifies]
            hand_cases = hand_cases[dealer_qualifies]
            hand_combinations = hand_combinations[dealer_qualifies]

            self.dnqual_count = dealer_doesnt_qualify_count
            dnqual_comb_count = np.zeros(shape=10, dtype=np.int)
            dnqual_comb_count[0] = np.sum(high_cards)
            dnqual_comb_count[1] = np.sum(pairs_of_deuces_or_threes)
            self.dnqual_comb_count = dnqual_comb_count
            self.dealer_qualifies_from = dealer_qualifies_from

        self.ranks = ranks
        self.suits = suits
        self.cases_rank_map = cases_rank_map
        self.cases_suit_map = cases_suit_map
        self.prime_number_map = prime_number_map
        self.hri = hand_rank_identifiers
        self.hsi = hand_suit_identifiers

        self.hand_cases = hand_cases
        self.hand_combinations = hand_combinations

        self.histogram = np.bincount(hand_combinations, minlength=10)
        self.N = N

    def compare(self, me):
        """
        :param me: my hand.
        :return:
        """
        my_hand = np.concatenate((self.table, me))
        my_comb = find_comb(my_hand)
        return self.compare_comb(my_comb, my_hand)

    def compare_comb(self, my_comb, my_hand):
        binary_map = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], np.int16)
        my_ranks = my_hand % 13

        loses = np.sum(my_comb[0] < self.hand_combinations)
        wins = np.sum(my_comb[0] > self.hand_combinations)

        same_comb_mask = self.hand_combinations == my_comb[0]
        same_comb_cases = self.hand_cases[same_comb_mask]
        same_comb_hri = self.hri[same_comb_mask]
        L = same_comb_cases.shape[0]

        if my_comb[0] == 1:
            # pair
            opp_hand_pair_rank = np.empty(shape=L, dtype=np.int8)
            for rank, prime in zip(self.ranks, self.prime_number_map):
                pair_rank_mask = same_comb_hri % (prime * prime) == 0
                opp_hand_pair_rank[pair_rank_mask] = rank
            my_pair_rank = my_comb[1]
            lose_count = np.sum(my_pair_rank < opp_hand_pair_rank)
            win_count = np.sum(my_pair_rank > opp_hand_pair_rank)
            draws = opp_hand_pair_rank == my_pair_rank

            same_paired_hands = same_comb_cases[draws]
            same_paired_hands_ranks = same_paired_hands % 13
            M, W = same_paired_hands_ranks.shape
            kickers = same_paired_hands_ranks[same_paired_hands_ranks != my_pair_rank]
            kickers = kickers.reshape((M, W - 2))
            kickers.sort()
            if W - 2 > 3:
                kickers = kickers[:, -3:]

            my_kicker = my_ranks[my_ranks != my_pair_rank]
            my_kicker.sort()
            W = my_ranks.shape[0]
            if W - 2 > 3:
                my_kicker = my_kicker[-3:]

            opp_binary_kickers = np.sum(binary_map[kickers], axis=1)
            my_binary_kicker = np.sum(binary_map[my_kicker])
            lose_count += np.sum(my_binary_kicker < opp_binary_kickers)
            draw_count = np.sum(opp_binary_kickers == my_binary_kicker)
            win_count += np.sum(my_binary_kicker > opp_binary_kickers)

            return loses, lose_count, draw_count, win_count, wins
        elif my_comb[0] == 2:
            # two pair
            opp_hand_1st_pair_rank = np.empty(shape=L, dtype=np.int8)
            opp_hand_2nd_pair_rank = np.empty(shape=L, dtype=np.int8)

            for rank, prime in zip(self.ranks, self.prime_number_map):
                pair_rank_mask = same_comb_hri % (prime * prime) == 0
                opp_hand_1st_pair_rank[pair_rank_mask] = rank
            for rank, prime in zip(self.ranks, self.prime_number_map):
                pair_rank_mask = same_comb_hri % (prime * prime) == 0
                opp_hand_2nd_pair_rank[pair_rank_mask & (opp_hand_1st_pair_rank > rank)] = rank
            my_1st_pair_rank = my_comb[1]
            my_2nd_pair_rank = my_comb[2]
            lose_count = np.sum(
                (my_1st_pair_rank < opp_hand_1st_pair_rank) | (
                        (my_1st_pair_rank == opp_hand_1st_pair_rank) & (my_2nd_pair_rank < opp_hand_2nd_pair_rank)
                )
            )
            win_count = np.sum(
                (my_1st_pair_rank > opp_hand_1st_pair_rank) | (
                        (my_1st_pair_rank == opp_hand_1st_pair_rank) & (my_2nd_pair_rank > opp_hand_2nd_pair_rank)
                )
            )
            draw_pairs = (opp_hand_1st_pair_rank == my_1st_pair_rank) & (opp_hand_2nd_pair_rank == my_2nd_pair_rank)

            same_paired_hands = same_comb_cases[draw_pairs]
            same_paired_hands_ranks = same_paired_hands % 13

            opp_kickers = same_paired_hands_ranks
            opp_kickers[opp_kickers == my_1st_pair_rank] = -1
            opp_kickers[opp_kickers == my_2nd_pair_rank] = -1

            opp_kickers = np.amax(opp_kickers, axis=1)

            my_kicker = my_ranks[(my_ranks != my_1st_pair_rank) & (my_ranks != my_2nd_pair_rank)]
            my_kicker = np.amax(my_kicker)

            lose_count += np.sum(my_kicker < opp_kickers)
            draw_count = np.sum(my_kicker == opp_kickers)
            win_count += np.sum(my_kicker > opp_kickers)

            return loses, lose_count, draw_count, win_count, wins
        elif my_comb[0] == 3:
            # set
            opp_hand_set_rank = np.empty(shape=L, dtype=np.int8)
            for rank, prime in zip(self.ranks, self.prime_number_map):
                set_rank_mask = same_comb_hri % (prime * prime * prime) == 0
                opp_hand_set_rank[set_rank_mask] = rank
            my_set_rank = my_comb[1]
            lose_count = np.sum(my_set_rank < opp_hand_set_rank)
            win_count = np.sum(my_set_rank > opp_hand_set_rank)
            draws = opp_hand_set_rank == my_set_rank

            same_set_hands = same_comb_cases[draws]
            same_set_hands_ranks = same_set_hands % 13
            M, W = same_set_hands_ranks.shape
            opp_kickers = same_set_hands_ranks[same_set_hands_ranks != my_set_rank]
            opp_kickers = np.reshape(opp_kickers, newshape=(M, W - 3))
            opp_kickers.sort()
            if W - 3 > 2:
                opp_kickers = opp_kickers[:, -2:]

            my_kicker = my_ranks[my_ranks != my_set_rank]
            my_kicker.sort()
            W = my_ranks.shape[0]
            if W - 3 > 2:
                my_kicker = my_kicker[-2:]

            opp_binary_kickers = np.sum(binary_map[opp_kickers], axis=1)
            my_binary_kicker = np.sum(binary_map[my_kicker])
            lose_count += np.sum(my_binary_kicker < opp_binary_kickers)
            draw_count = np.sum(opp_binary_kickers == my_binary_kicker)
            win_count += np.sum(my_binary_kicker > opp_binary_kickers)

            return loses, lose_count, draw_count, win_count, wins
        elif my_comb[0] == 4:
            # straight
            my_toprank = my_comb[1]
            straight_count = np.zeros(shape=L, dtype=np.int8)
            straight_toprank = np.zeros(shape=L, dtype=np.int8)
            wheel_div = 2 * 3 * 5 * 7 * 41
            straight_toprank[(same_comb_hri // wheel_div) == 0] = -1

            n = 0
            for rank, prime in zip(self.ranks, self.prime_number_map):
                contain_rank = (same_comb_hri % prime) == 0
                straight_count += contain_rank
                straight_count[~contain_rank] = 0
                if n > 3:
                    straight_toprank[straight_count > 4] = rank
                n += 1

            return (loses,
                    np.sum(my_toprank < straight_toprank),
                    np.sum(my_toprank == straight_toprank),
                    np.sum(my_toprank > straight_toprank),
                    wins)
        elif my_comb[0] == 5:
            # flush
            my_flush_ranks = my_comb[1]
            flush_suit = my_comb[2]
            same_comb_cases_suits = same_comb_cases // 13

            flush_cards = same_comb_cases
            flush_cards[same_comb_cases_suits != flush_suit] = -1
            flush_cards.sort()
            opp_flush_ranks = flush_cards[:, -5:] % 13

            opp_binary_flush = np.sum(binary_map[opp_flush_ranks], axis=1)
            my_binary_flush = np.sum(binary_map[my_flush_ranks])

            lose_cases = np.sum(my_binary_flush < opp_binary_flush)
            draw_cases = np.sum(my_binary_flush == opp_binary_flush)
            win_cases = np.sum(my_binary_flush > opp_binary_flush)

            return loses, lose_cases, draw_cases, win_cases, wins
        elif my_comb[0] == 6:
            # full house
            opp_hand_set_rank = np.empty(shape=L, dtype=np.int8)
            opp_hand_pair_rank = np.empty(shape=L, dtype=np.int8)

            for rank, prime in zip(self.ranks, self.prime_number_map):
                set_rank_mask = same_comb_hri % (prime * prime * prime) == 0
                opp_hand_set_rank[set_rank_mask] = rank
            for rank, prime in zip(self.ranks, self.prime_number_map):
                pair_rank_mask = same_comb_hri % (prime * prime) == 0
                opp_hand_pair_rank[pair_rank_mask & (opp_hand_set_rank != rank)] = rank
            my_set_rank = my_comb[1]
            my_pair_rank = my_comb[2]
            lose_count = np.sum(
                (my_set_rank < opp_hand_set_rank) | (
                        (my_set_rank == opp_hand_set_rank) & (my_pair_rank < opp_hand_pair_rank)
                )
            )
            win_count = np.sum(
                (my_set_rank > opp_hand_set_rank) | (
                        (my_set_rank == opp_hand_set_rank) & (my_pair_rank > opp_hand_pair_rank)
                )
            )
            draw_count = np.sum((opp_hand_set_rank == my_set_rank) & (opp_hand_pair_rank == my_pair_rank))

            return loses, lose_count, draw_count, win_count, wins
        elif my_comb[0] == 7:
            # quads
            opp_hand_quads_rank = np.empty(shape=L, dtype=np.int8)
            for rank, prime in zip(self.ranks, self.prime_number_map):
                quads_rank_mask = same_comb_hri % (prime * prime * prime * prime) == 0
                opp_hand_quads_rank[quads_rank_mask] = rank
            my_quads_rank = my_comb[1]
            lose_count = np.sum(my_quads_rank < opp_hand_quads_rank)
            win_count = np.sum(my_quads_rank > opp_hand_quads_rank)
            draws = opp_hand_quads_rank == my_quads_rank

            if np.sum(draws) > 0:
                same_quads_hands = same_comb_cases[draws]
                same_quads_hands_ranks = same_quads_hands % 13

                opp_kickers = same_quads_hands_ranks[same_quads_hands_ranks != my_quads_rank]

                opp_kickers = np.amax(opp_kickers)

                my_kicker = my_ranks[my_ranks != my_quads_rank]
                my_kicker = np.amax(my_kicker)

                lose_count += np.sum(my_kicker < opp_kickers)
                draw_count = np.sum(my_kicker == opp_kickers)
                win_count += np.sum(my_kicker > opp_kickers)

                return loses, lose_count, draw_count, win_count, wins
            return loses, lose_count, 0, win_count, wins
        elif my_comb[0] == 8:
            # straight flush
            if self.str_fl_tops is not None:
                my_toprank = my_comb[1]
                only_str_fl = self.str_fl_tops[self.str_fl_tops != 0]
                return (loses,
                        np.sum(my_toprank < only_str_fl),
                        np.sum(my_toprank == only_str_fl),
                        np.sum(my_toprank > only_str_fl),
                        wins)
            else:
                # best combination eva
                return 0, 0, 0, 0, wins
        elif my_comb[0] == 9:
            # royal flush
            return 0, 0, 0, 0, wins
        else:
            # High card
            same_comb_ranks = same_comb_cases % 13
            same_comb_ranks.sort()
            high_cards = same_comb_ranks[:, -1]
            my_ranks = my_hand % 13
            my_ranks.sort()
            my_high_card = my_ranks[-1]
            win_count = np.sum(my_high_card > high_cards)
            lose_count = np.sum(my_high_card < high_cards)
            draws = same_comb_ranks[my_high_card == high_cards]

            opp_kickers = draws[:, -5:-1]
            my_kicker = my_ranks[-5:-1]

            opp_binary_kickers = np.sum(binary_map[opp_kickers], axis=1)
            my_binary_kicker = np.sum(binary_map[my_kicker])

            lose_count += np.sum(my_binary_kicker < opp_binary_kickers)
            draw_count = np.sum(opp_binary_kickers == my_binary_kicker)
            win_count += np.sum(my_binary_kicker > opp_binary_kickers)

            return loses, lose_count, draw_count, win_count, wins

    def equity(self, cards):
        res = np.array(self.compare(cards), dtype=np.double)
        return res / np.sum(res)


class HandMap:
    def __init__(self, deck, table):
        deck.sort()
        table.sort()

        self.deck = deck
        self.table = table
        card1_ind_map = {}
        card2_ind_map = {}
        i = 0
        for card in deck:
            card2_ind_map[card] = i
            i += 1

        N = deck.shape[0]
        CN = c(N, 2)
        hand_map = np.empty(shape=(CN, 2), dtype=deck.dtype)
        start = 0
        end = N - 1
        for n in range(N - 1):
            card1_ind_map[deck[n]] = (start, end)
            hand_map[start:end, 0] = deck[n]
            hand_map[start:end, 1] = deck[n + 1:]
            start = start + (N - n - 1)
            end = start + (N - n - 2)

        self.hand_map = hand_map
        self.comb_map = np.zeros(shape=CN, dtype=np.int8)
        self.card1_ind_map = card1_ind_map
        self.card2_ind_map = card2_ind_map
        self.N = N
        # инициализация базы данных завершена

        # личная характеристика раздачи
        # [ ранг первой пары,
        #   ранг второй пары,
        #   ранг первой тройки,
        #   ранг второй тройки,
        #   топ карта стрита,
        #   импакт карта флеша,
        #   ранг четверки ]
        identity_matrix = np.zeros(shape=[CN, 7], dtype=np.int)
        # руки, в которых максимальная комбинация - это комбинация на столе, помечены True
        # например, на столе лежит 4 четверки, а в руке что бы ни лежало, круче уже не будет.
        is_table_comb = np.ones(shape=CN, dtype=np.bool)
        # поиск комбинаций
        table_comb = find_comb(table)
        # which combination is already on the table?
        if table_comb == 1:
            # pair on the table
            identity_matrix[:, 0] = 1
        elif table_comb == 2:
            # two-pair
            identity_matrix[:, 0] = 2
        elif table_comb == 3:
            # set
            identity_matrix[:, 1] = 1
        elif table_comb == 6:
            # full house
            identity_matrix[:, 0] = 1
            identity_matrix[:, 1] = 1

        self.comb_map[:] = table_comb

        pairs = vstacker.stack(genpairs(deck, table))
        # базовые типы сразу опознаются, чтобы не смотреть отдельно их наличие,
        # впоследствии заменяются комбинациями базовых типов, если возможно

        # найти пары
        for pair in pairs:
            index = self.index_of(pair)
            if self.comb_map[index] < 1:
                self.comb_map[index] = 1
            is_table_comb[index] = False
            identity_matrix[index, 0] += 1

        # найти 2 пары
        twopair_mask = identity_matrix[:, 0] > 1
        self.comb_map[twopair_mask] = 2

        # найти сеты
        sets = vstacker.stack(gensets(deck, table))
        for set in sets:
            index = self.index_of(set)
            if self.comb_map[index] < 3:
                self.comb_map[index] = 3
            is_table_comb[index] = False
            identity_matrix[index, 1] += 1

        # найти флеши
        flushes = vstacker.stack(genflushes(deck, table))
        for flush in flushes:
            index = self.index_of(flush)
            if self.comb_map[index] <= 5:
                self.comb_map[index] = 5
                is_table_comb[index] = False

        # найти стриты и стрит-флеши
        straights = vstacker.stack(genstraights(deck, table))
        for straight in straights:
            index = self.index_of(straight)
            if self.comb_map[index] == 5:
                self.comb_map[index] = 8
                is_table_comb[index] = False
            elif self.comb_map[index] <= 4:
                self.comb_map[index] = 4
                is_table_comb[index] = False

        # фул-хаусы
        self.comb_map[((identity_matrix[:, 0] > 0) & (identity_matrix[:, 1] > 0)) | (identity_matrix[:, 1] > 1)] = 6

        # последний из проверяемых
        quads = vstacker.stack(genquads(deck, table))
        for quad in quads:
            index = self.index_of(quad)
            # если не стрит-флеш, замени на каре
            if self.comb_map[index] < 8:
                self.comb_map[index] = 7

        self.histogram = np.bincount(self.comb_map, minlength=9)

    def index_of(self, cards):
        card1, card2 = cards
        mn = min(card1, card2)
        mx = max(card1, card2)
        card1 = mn
        card2 = mx
        i1 = self.card1_ind_map[card1][0]
        i2 = self.card2_ind_map[card2]
        shift = self.card2_ind_map[card1]
        index = i1 - shift + i2 - 1
        return index

    def comb(self, cards):
        index = self.index_of(cards)
        return self.comb_map[index]

    def _every(self, card, comb):
        start, end = self.card1_ind_map[card]
        self.comb_map[self.hand_map[:, 1] == card] = comb
        self.comb_map[start:end] = comb

    def _set(self, cards, comb):
        try:
            if len(cards) == 1:
                card = cards
                self._every(card, comb)
            else:
                self.comb_map[self.index_of(cards)] = comb
        except TypeError:
            card = cards
            self._every(card, comb)


def find_comb(hand):
    hand_ranks = hand % 13
    hand_suits = hand // 13

    comb = 0
    straight_or_flush = False
    flush_suit_mask = np.bincount(hand_suits, minlength=4) >= 5

    is_str8, str8_rank = is_straight(hand_ranks)
    if is_str8:
        # straight
        comb = 4
        straight_or_flush = True
    if np.any(flush_suit_mask):
        if comb == 4:
            # straight flush?
            flush_suit = np.array([0, 1, 2, 3], dtype=np.int8)[flush_suit_mask][0]
            flush_ranks = hand_ranks[hand_suits == flush_suit]
            is_sf, sf_rank = is_straight(flush_ranks)
            if is_sf:
                royal_flush_ranks = np.array([8, 9, 10, 11, 12], dtype=np.int8)
                if np.all(np.isin(royal_flush_ranks, flush_ranks)):
                    return 9, flush_suit
                return 8, sf_rank, flush_suit
        straight_or_flush = True

    uniq, count = np.unique(hand_ranks, return_counts=True)
    histogram = np.bincount(count, minlength=5)

    if histogram[4] > 0:
        # four of a kind
        quads_rank = np.max(uniq[count == 4])
        kicker = np.max(uniq[uniq != quads_rank])
        return 7, quads_rank, kicker
    elif histogram[3] > 0 and histogram[2] > 0:
        # full house
        return 6, np.max(uniq[count == 3]), np.max(uniq[count == 2])
    elif histogram[3] > 1:
        # another case of full house
        fh = uniq[count == 3]
        return 6, np.max(fh), np.min(fh)

    if not straight_or_flush:
        if histogram[3] > 0:
            # three of a kind
            set_rank = uniq[count == 3][0]
            kicker = np.sort(uniq[count == 1])[::-1]
            return 3, set_rank, kicker[:2]
        elif histogram[2] > 1:
            # two pair
            tp = uniq[count == 2]
            tp.sort()
            kicker = np.max(uniq[(uniq != tp[-1]) | (uniq != tp[-2])])
            return 2, tp[-1], tp[-2], kicker
        elif histogram[2] == 1:
            # just a pair
            kicker = np.sort(uniq[count == 1])[::-1]
            return 1, uniq[count == 2][0], kicker[:3]

    if is_str8:
        return 4, str8_rank
    elif not is_str8 and straight_or_flush:
        flush_suit = np.array([0, 1, 2, 3], dtype=np.int8)[flush_suit_mask][0]
        flush_ranks = hand[hand // 13 == flush_suit] % 13
        return 5, np.sort(flush_ranks)[-5:], flush_suit
    return 0, np.max(hand_ranks), np.sort(hand_ranks)[::-1][1:5]


def ev_dnq(equity, lose_value=-1.0, draw_value=0, win_value=1.5, dnq_value=0.5):
    l = np.sum(equity[:2])
    d = equity[2]
    w = np.sum(equity[3:5])
    dnq = equity[5]
    return l * lose_value +  d * draw_value + w * win_value + dnq * dnq_value


def casino_poker_ev(comb_equity, histogram, ante=0.5, bet=1.0,
                    fl_mult=2,
                    fh_mult=3,
                    qd_mult=10,
                    sf_mult=20,
                    rf_mult=100):
    # NOTE: in casino poker, draw does not have any value. The value is always zero.
    # for high card - impossible to win, only for the dealer to not qualify, or lose
    hc_ev = np.sum(comb_equity[0, :2]) * -bet + comb_equity[0, 5] * ante
    # for pairs
    p_ev = np.sum(comb_equity[1, :2]) * -bet + np.sum(comb_equity[1, 3:5]) * (ante + bet) + comb_equity[1, 5] * ante
    # for two pairs
    tp_ev = np.sum(comb_equity[2, :2]) * -bet + np.sum(comb_equity[2, 3:5]) * (ante + bet) + comb_equity[2, 5] * ante
    # for sets
    se_ev = np.sum(comb_equity[3, :2]) * -bet + np.sum(comb_equity[3, 3:5]) * (ante + bet) + comb_equity[3, 5] * ante
    # for straights
    st_ev = np.sum(comb_equity[4, :2]) * -bet + np.sum(comb_equity[4, 3:5]) * (ante + bet) + comb_equity[4, 5] * ante
    # for flushes
    fl_ev = np.sum(comb_equity[5, :2]) * -bet + np.sum(comb_equity[5, 3:5]) * (fl_mult * ante + bet) + comb_equity[5, 5] * (ante * fl_mult)
    # for full houses
    fh_ev = np.sum(comb_equity[6, :2]) * -bet + np.sum(comb_equity[6, 3:5]) * (fh_mult * ante + bet) + comb_equity[6, 5] * (ante * fh_mult)
    # for quads
    qd_ev = np.sum(comb_equity[7, :2]) * -bet + np.sum(comb_equity[7, 3:5]) * (qd_mult * ante + bet) + comb_equity[7, 5] * (ante * qd_mult)
    # for straight flushes
    sf_ev = np.sum(comb_equity[8, :2]) * -bet + np.sum(comb_equity[8, 3:5]) * (sf_mult * ante + bet) + comb_equity[8, 5] * (ante * sf_mult)
    # for royal flushes
    rf_ev = np.sum(comb_equity[9, :2]) * -bet + np.sum(comb_equity[9, 3:5]) * (rf_mult * ante + bet) + comb_equity[9, 5] * (ante * rf_mult)
    return histogram[0] * hc_ev + \
           histogram[1] * p_ev + \
           histogram[2] * tp_ev + \
           histogram[3] * se_ev + \
           histogram[4] * st_ev + \
           histogram[5] * fl_ev + \
           histogram[6] * fh_ev + \
           histogram[7] * qd_ev + \
           histogram[8] * sf_ev + \
           histogram[9] * rf_ev


def comp_kickers(k1, k2):
    binary_map = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], np.int16)
    k1_bin = np.sum(binary_map[k1])
    k2_bin = np.sum(binary_map[k2])
    if k1_bin > k2_bin:
        return 1
    elif k1_bin < k2_bin:
        return -1
    else:
        return 0


def comp_combs_44(comb1, comb2):
    if comb2[0] == 0 or (comb2[0] == 1 and comb2[1] < 2):
        # krupier sie nie kwalifikuje
        return 2
    else:
        return comp_combs(comb1, comb2)


def comp_combs(comb1, comb2):
    if comb1[0] > comb2[0]:
        return 1
    elif comb1[0] < comb2[0]:
        return -1
    else:
        comb_type = comb1[0]
        if comb_type == 0:
            # HIGH CARD
            if comb1[1] > comb2[1]:
                return 1
            elif comb1[1] < comb2[1]:
                return -1
            else:
                return comp_kickers(comb1[2], comb2[2])
        elif comb_type == 1:
            # PAIR
            if comb1[1] > comb2[1]:
                return 1
            elif comb1[1] < comb2[1]:
                return -1
            else:
                return comp_kickers(comb1[2], comb2[2])
        elif comb_type == 2:
            # TWO PAIR
            highest_pair1 = comb1[1]
            highest_pair2 = comb2[1]
            snd_pair1 = comb1[2]
            snd_pair2 = comb2[2]
            kicker1 = comb1[3]
            kicker2 = comb2[3]

            if highest_pair1 > highest_pair2:
                return 1
            elif highest_pair1 < highest_pair2:
                return -1
            else:
                if snd_pair1 > snd_pair2:
                    return 1
                elif snd_pair1 < snd_pair2:
                    return -1
                else:
                    if kicker1 > kicker2:
                        return 1
                    elif kicker1 < kicker2:
                        return -1
                    else:
                        return 0
        elif comb_type == 3:
            # SET
            if comb1[1] > comb2[1]:
                return 1
            elif comb1[1] < comb2[1]:
                return -1
            else:
                return comp_kickers(comb1[2], comb2[2])
        elif comb_type == 4:
            # STRAIGHT
            topcard1 = comb1[1]
            topcard2 = comb2[1]
            if topcard1 > topcard2:
                return 1
            elif topcard1 < topcard2:
                return -1
            else:
                return 0
        elif comb_type == 5:
            # FLUSH
            flush_cards1 = comb1[1]
            flush_cards2 = comb2[1]
            return comp_kickers(flush_cards1, flush_cards2)
        elif comb_type == 6:
            # FULL HOUSE
            set_rank1 = comb1[1]
            pair_rank1 = comb1[2]

            set_rank2 = comb2[1]
            pair_rank2 = comb2[2]

            if set_rank1 > set_rank2:
                return 1
            elif set_rank1 < set_rank2:
                return -1
            else:
                if pair_rank1 > pair_rank2:
                    return 1
                elif pair_rank1 < pair_rank2:
                    return -1
                else:
                    return 0
        elif comb_type == 7:
            # QUADS
            quads_rank1 = comb1[1]
            quads_rank2 = comb2[1]

            if quads_rank1 > quads_rank2:
                return 1
            elif quads_rank1 < quads_rank2:
                return -1
            else:
                kicker1 = comb1[2]
                kicker2 = comb2[2]
                if kicker1 > kicker2:
                    return 1
                elif kicker1 < kicker2:
                    return -1
                else:
                    return 0
        elif comb_type == 8:
            # STRAIGHT FLUSH
            topcard1 = comb1[1]
            topcard2 = comb2[1]
            if topcard1 > topcard2:
                return 1
            elif topcard1 < topcard2:
                return -1
            else:
                return 0
        elif comb_type == 9:
            # ROYAL FLUSH
            return 0
    raise ValueError("Bad combination number. \ncomb1[0] == " + str(comb1[0]) + "\ncomb2[0] == " + str(comb2[0]))


def comb2str(comb):
    s = ""
    if comb[0] == 0:
        s += "High card " + int2rank(comb[1])
        s += "\n"
        for k in comb[2]:
            s += int2rank(k) + " "
        s += "kicker.\n"
    elif comb[0] == 1:
        s += "Pair of " + int2rank(comb[1]) + "s."
        s += "\n"
        for k in comb[2]:
            s += int2rank(k) + " "
        s += "kicker.\n"
    elif comb[0] == 2:
        s += "Two pairs: \n" + int2rank(comb[1]) + "s & " + int2rank(comb[2]) + "s."
        s += "\n" + int2rank(comb[3]) + " kicker.\n"
    elif comb[0] == 3:
        s += "Set of " + int2rank(comb[1]) + "s"
        s += "\n"
        for k in comb[2]:
            s += int2rank(k) + " "
        s += "kicker.\n"
    elif comb[0] == 4:
        s += "Straight " + int2rank(comb[1]) + "-high\n"
    elif comb[0] == 5:
        s += "Flush: \n"
        for k in comb[1]:
            s += int2rank(k) + "-"
        s += "high\n"
    elif comb[0] == 6:
        s += "Full house: \n"
        s += int2rank(comb[1]) + "s full of " + int2rank(comb[2]) + "s.\n"
    elif comb[0] == 7:
        s += "Quads of " + int2rank(comb[1]) + "s\n"
        s += int2rank(comb[2]) + " kicker.\n"
    elif comb[0] == 8:
        s += "Straight flush " + int2rank(comb[1]) + "-high\n"
    elif comb[0] == 9:
        s += "Royal flush of\n" + suit2word(comb[1]) + ".\n"
    return s


def fc(hand):
    i = 0
    for find in comb_find_funcs_desc:
        c = find(hand)
        if c is not None:
            return c, i
        i += 1
    return None


def ldw(b_comb, opp_hand):
    """
    lose draw win function
    :param b_comb: bot combination (PokerComb)
    :param opp_hand: opponent's hand (numpy array)
    :return: lose draw win array (if bot loses [1 0 0], if wins [0 0 1], draw [0 1 0])
             opponent's combination type (0 - straight flush, 1 - Quads ...)
    """
    opp_comb, opp_comb_n = fc(opp_hand)
    ldw_ar = np.zeros(shape=3, dtype=np.uint32)
    if b_comb == opp_comb:
        # it is a draw
        ldw_ar[draw] += 1
    elif b_comb < opp_comb:
        # bot loses
        ldw_ar[lose] += 1
    else:
        # bot wins
        ldw_ar[win] += 1
    return ldw_ar, opp_comb_n


def gensets(deck, table):
    """
    :param deck:
    :param table:
    :return: возвращает лист таблиц numpy 3 в ширину, где первые два элемента - рука, делающая сет, а 3-ий - ранг сета
    """
    deck_ranks = deck % 13
    table_ranks = table % 13
    uniq, count = np.unique(table_ranks, return_counts=True)

    sets = []

    for card_rank, rank_count in zip(uniq, count):
        if rank_count == 1:
            # pocket pairs
            deck_set_cards = deck[deck_ranks == card_rank]
            if len(deck_set_cards) > 1:
                s = pair_wr(deck_set_cards, w=3)
                # ранг карты, для которой существует сет
                s[:, 2] = card_rank
                sets.append(s)
        elif rank_count == 2:
            # добавочная одна карта
            deck_set_cards = deck[deck_ranks == card_rank]
            if len(deck_set_cards) > 0:
                other_deck_cards = deck[deck_ranks != card_rank]
                for set_card in deck_set_cards:
                    set_hands = np.empty(shape=(other_deck_cards.shape[0], 3), dtype=np.int8)
                    set_hands[:, 0] = set_card
                    set_hands[:, 1] = other_deck_cards
                    # ранг карты, для которой существует сет
                    set_hands[:, 2] = card_rank
                    sets.append(set_hands)
    return sets


def genpairs(deck, table):
    """
    :param deck:
    :param table:
    :return: см. gensets(deck, table)
    """
    deck_ranks = deck % 13
    table_ranks = table % 13
    uniq, count = np.unique(table_ranks, return_counts=True)

    pairs = []

    for card_rank, rank_count in zip(uniq, count):
        if rank_count == 1:
            # одна добавочная карта
            deck_pair_cards = deck[deck_ranks == card_rank]
            other_deck_cards = deck[deck_ranks != card_rank]
            for pair_card in deck_pair_cards:
                pair_hands = np.empty(shape=(other_deck_cards.shape[0], 3), dtype=np.int8)
                pair_hands[:, 0] = pair_card
                pair_hands[:, 1] = other_deck_cards
                # ранг пары
                pair_hands[:, 2] = card_rank
                pairs.append(pair_hands)
    # pocket pairs
    # любые две карты, ранги которых не представлены на столе
    pp_deck_mask = ~eq(deck_ranks, table_ranks)
    pp_ranks, pp_card_count = np.unique(deck_ranks[pp_deck_mask], return_counts=True)
    for rank, card_count in zip(pp_ranks, pp_card_count):
        if card_count == 2:
            p = np.empty(3, dtype=np.int8)
            p[:2] = deck[deck_ranks == rank]
            p[3] = rank
            pairs.append(p)
        elif card_count > 2:
            pp_cards = deck[deck_ranks == rank]
            pair_hands = pair_wr(pp_cards, w=3)
            pair_hands[:, 2] = rank
            pairs.append(pair_hands)
    return pairs


def genquads(deck, table):
    deck_ranks = deck % 13
    table_ranks = table % 13
    uniq, count = np.unique(table_ranks, return_counts=True)

    quads = []
    for card_rank, rank_count in zip(uniq, count):
        if rank_count == 3:
            # не хватает одной карты
            deck_quad_card = deck[deck_ranks == card_rank]
            # если она есть в деке
            if len(deck_quad_card) > 0:
                quad_card = deck_quad_card[0]
                other_deck_cards = deck[deck_ranks != card_rank]
                quad_hands = np.empty(shape=(other_deck_cards.shape[0], 3), dtype=np.int8)
                quad_hands[:, 0] = quad_card
                quad_hands[:, 1] = other_deck_cards
                quad_hands[:, 2] = card_rank
                quads.append(quad_hands)
        elif rank_count == 2:
            # не хватает двух карт
            # pocket pair
            deck_quad_cards = deck[deck_ranks == card_rank]
            if len(deck_quad_cards) > 1:
                quads.append(np.concatenate((deck_quad_cards, [card_rank])))
    return quads


def genstraights(deck, table):
    """
    :param deck: complementary set. A set of cards available to put on the table as a straight complement.
    :param table: target cards
    :return: лист numpy таблиц 3 элемента шириной, где первые два - рука, делающая стрит,
             последний - высшая (топ) карта стрита
    """
    table_ranks = table % 13
    uniq_table_ranks = np.unique(table_ranks)
    if len(uniq_table_ranks) < 3:
        return []

    deck_ranks = deck % 13
    uniq_deck_ranks = np.unique(deck_ranks)

    sparse_fit_ar = -np.ones(shape=13, dtype=np.int)
    sparse_fit_ar[uniq_table_ranks] = uniq_table_ranks
    sparse_fit_ar[uniq_deck_ranks] = uniq_deck_ranks
    mn = np.min(table_ranks)
    mx = np.max(table_ranks)
    start = mn
    end = mx

    straights = []
    # если возможен стрит-колесо, то начать проверку с туза
    if np.sum(sparse_fit_ar[:4] > -1) > 3 and sparse_fit_ar[12] == 12:
        # possible straight
        ps = np.empty(shape=5, dtype=np.int)
        ps[1:5] = sparse_fit_ar[0:4]
        ps[0] = 12
        ps_table_cards = np.isin(uniq_table_ranks, ps)
        ps_deck_cards = np.isin(uniq_deck_ranks, ps)
        ps_deck_cards[uniq_table_ranks] = False

        onhand_cards_sum = np.sum(ps_deck_cards)

        if np.all(ps != -1) and np.sum(ps_table_cards) > 2 and 0 < onhand_cards_sum < 3:
            # Если все карты в выборке имеются в игре,
            # И карт, делающих стрит, на столе больше двух,
            # И карт, взятых из деки, в выборке не больше двух,
            # то это наш пациент.
            if onhand_cards_sum == 1:
                # на руке дожлжна быть одна карта, которая закончит стрит
                straight_cards_rank = uniq_deck_ranks[ps_deck_cards][0]
                other_deck_cards = deck[deck_ranks != straight_cards_rank]
                straight_cards = deck[deck_ranks == straight_cards_rank]
                for straight_card in straight_cards:
                    straight_hands = np.empty(shape=[len(other_deck_cards), 3], dtype=np.int8)
                    straight_hands[:, 0] = straight_card
                    straight_hands[:, 1] = other_deck_cards
                    straight_hands[:, 2] = 12
                    straights.append(straight_hands)
            elif onhand_cards_sum == 2:
                # на руке должно быть 2 карты, чтобы закончить стрит
                straight_cards_ranks = uniq_deck_ranks[ps_deck_cards]
                straight_card_rank1, straight_card_rank2 = straight_cards_ranks
                straight_cards1 = deck[deck_ranks == straight_card_rank1]
                straight_cards2 = deck[deck_ranks == straight_card_rank2]
                if len(straight_cards1) > 0 and len(straight_cards2) > 0:
                    for sc1 in straight_cards1:
                        straight_hands = np.empty(shape=[straight_cards2.shape[0], 3], dtype=np.int8)
                        straight_hands[:, 0] = sc1
                        straight_hands[:, 1] = straight_cards2
                        straight_hands[:, 2] = 12
                        straights.append(straight_hands)
        start = 0
    elif mn >= 2:
        start = start - 2
    else:
        start = 0
    if mx > 9:
        end = 9

    for n in range(start, end):
        if np.sum(sparse_fit_ar[n:] != -1) >= 5:
            # possible straight
            ps = sparse_fit_ar[n:n+5]
            toprank = np.max(ps) % 12

            ps_table_cards = np.isin(uniq_table_ranks, ps)
            ps_deck_cards = np.isin(uniq_deck_ranks, ps)
            ps_deck_cards[uniq_table_ranks] = False

            onhand_cards_sum = np.sum(ps_deck_cards)

            if np.all(ps != -1) and np.sum(ps_table_cards) > 2 and 0 < onhand_cards_sum < 3:
                # Если все карты в выборке имеются в игре,
                # И карт, делающих стрит, на столе больше двух,
                # И карт, взятых из деки, в выборке не больше двух,
                # то это наш пациент.
                if onhand_cards_sum == 1:
                    # на руке дожлжна быть одна карта, которая закончит стрит
                    straight_cards_rank = uniq_deck_ranks[ps_deck_cards][0]
                    other_deck_cards = deck[deck_ranks != straight_cards_rank]
                    straight_cards = deck[deck_ranks == straight_cards_rank]
                    for straight_card in straight_cards:
                        straight_hands = np.empty(shape=[len(other_deck_cards), 3], dtype=np.int8)
                        straight_hands[:, 0] = straight_card
                        straight_hands[:, 1] = other_deck_cards
                        straight_hands[:, 2] = toprank
                        straights.append(straight_hands)
                elif onhand_cards_sum == 2:
                    # на руке должно быть 2 карты, чтобы закончить стрит
                    straight_cards_ranks = uniq_deck_ranks[ps_deck_cards]
                    straight_card_rank1, straight_card_rank2 = straight_cards_ranks
                    straight_cards1 = deck[deck_ranks == straight_card_rank1]
                    straight_cards2 = deck[deck_ranks == straight_card_rank2]
                    if len(straight_cards1) > 0 and len(straight_cards2) > 0:
                        for sc1 in straight_cards1:
                            straight_hands = np.empty(shape=[straight_cards2.shape[0], 3], dtype=np.int8)
                            straight_hands[:, 0] = sc1
                            straight_hands[:, 1] = straight_cards2
                            straight_hands[:, 2] = toprank
                            straights.append(straight_hands)
        else:
            # если карт во всем проверяемом пространстве меньше 5, то стрит автоматически невозможен
            break

    return straights


def genflushes(deck, table):
    """
    :param deck:
    :param table:
    :return: numpy array или лист numpy array'ev, где первые два элемента - рука, которая делает флеш, а 3-ий -
             карта, которая делает разницу между флешами.
             Например, нужно две карты, чтобы закончить флеш. Добираешь 4 и 7 пик - делаешь флеш пик. Четверка выступает,
             как филлер: если у кого-то еще будет флеш, то именно семерка будет делать разницу между флешами.
    """
    # complement - дополнение
    comp = 2
    table_suits = table // 13
    n_suits = np.bincount(table_suits)
    suits_possibility = n_suits >= (5 - comp)
    if np.any(suits_possibility):
        flush_suit = np.where(suits_possibility)[0][0]
        deck_suits = deck // 13

        flush_table_cards = table[table_suits == flush_suit]
        flush_table_cards.sort()

        flush_deck_cards = deck[deck_suits == flush_suit]
        flush_deck_cards.sort()

        # с флешем никогда не бывает ничьи, кроме случая, когда флеш на столе
        if flush_table_cards.size == 5:
            # все карты флеша собраны на столе
            ftc_ranks = flush_table_cards % 13
            fds_ranks = flush_deck_cards % 13
            impact_deck_cards = flush_deck_cards[fds_ranks > np.min(ftc_ranks)]
            flushes = []
            n = 0
            for impact_card in impact_deck_cards:
                # available cards
                available_flush_cards = impact_deck_cards[:n]
                av_deck_cards = deck[~np.isin(deck, available_flush_cards)]
                flush_hands = np.empty(shape=(av_deck_cards.shape[0], 3), dtype=np.int8)
                flush_hands[:, 0] = impact_card
                flush_hands[:, 1] = av_deck_cards
                flush_hands[:, 2] = impact_card
                flushes.append(flush_hands)
                n += 1
            return flushes
        elif flush_table_cards.size == 4:
            # не хватает одной карты до флеша
            flushes = []
            n = 0
            for card in flush_deck_cards:
                # available cards
                available_flush_cards = flush_deck_cards[:n]
                av_deck_cards = deck[~np.isin(deck, available_flush_cards)]
                flush_hands = np.empty(shape=(av_deck_cards.shape[0], 3), dtype=np.int8)
                flush_hands[:, 0] = card
                flush_hands[:, 1] = av_deck_cards
                flush_hands[:, 2] = card
                flushes.append(flush_hands)
                n += 1
            return flushes

        elif flush_table_cards.size == 3:
            # две карты до флеша
            flush_hands = pair_wr(flush_deck_cards, w=3)
            flush_hands[:, 2] = np.amax(flush_hands[:, :2], axis=1)
            return flush_hands
    return []


def equity(hole, table, hl=2):
    # overall number of hands which lose, draw, win
    ldw_all = np.zeros(shape=(9, 3), dtype=np.uint32)
    # indexes of those hands mapped to holes_left array

    hole = np.asarray(hole, dtype=np.int8)
    table = np.asarray(table, dtype=np.int8)

    hand = np.concatenate((hole, table))
    b_comb, b_comb_n = fc(hand)
    cards_left = list(range(0, 52))
    for card in hand:
        cards_left.remove(card)

    opp_holes = combinations(cards_left, hl)
    matrix = np.empty(shape=(opp_holes.shape[0], len(table) + hl), dtype=np.int8)
    matrix[:, :hl] = opp_holes
    matrix[:, hl:] = table

    for row in matrix:
        ldw_case, opp_comb = ldw(b_comb, row)
        ldw_all[opp_comb] += ldw_case

    return ldw_all, b_comb_n


def final_equity(hole, table, river=5):
    """
    :param hole:
    :param table:
    :param river:
    :return:
    """
    ldw_all = np.zeros(shape=(9, 3), dtype=np.uint32)

    cards_left = list(range(0, 52))
    hand = np.concatenate((hole, table))

    for card in hand:
        cards_left.remove(card)

    v_is_iterable = True
    # length of cards from the deck to complement so that the desired state of game is reached
    complement = river - len(table)
    if complement == 0:
        return equity(hole, table)
    elif complement == 1:
        compl_var = np.asarray(cards_left, dtype=np.int8)
        v_is_iterable = False
    else:
        compl_var = combinations(cards_left, complement)

    # iterate through every possible compliment
    for v in compl_var:
        # overall number of hands which lose, draw, win
        vldw_all = np.zeros(shape=(9, 3), dtype=np.uint32)

        # creating a matrix of all possible hole cards on that table
        vcards_left = cards_left.copy()
        if v_is_iterable:
            for card in v:
                vcards_left.remove(card)
            vtable = np.concatenate((v, table))
        else:
            vcards_left.remove(v)
            vtable = np.concatenate(([v], table))
        vopp_holes = combinations(vcards_left, 2)
        vmatrix = np.empty(shape=(vopp_holes.shape[0], river + 2), dtype=np.int8)
        vmatrix[:, :2] = vopp_holes
        vmatrix[:, 2:] = vtable

        vb_comb = find_comb(np.concatenate((hole, vtable)))

        for row in vmatrix:
            vldw_case, vopp_comb_n = ldw(vb_comb, row)
            vldw_all[vopp_comb_n] += vldw_case
        ldw_all += vldw_all
    return ldw_all


# print(equity([31, 5], [2, 18, 19, 29, 12]))
# 0. high card +
# 1. pair +
# 2. two pair +
# 3. set +
# 4. straight +
# 5. flush +
# 6. full house +
# 7. quads +
# 8. straight flush +

def flop_equity_44(table, hand):
    t1 = time()
    hand_histogram = np.zeros(shape=10, dtype=np.int)

    deck = deck_from(table, hand)
    if table.size != 3 or hand.size != 2:
        raise ValueError("Hand or table length not compatible. "
                         "Hand is expected 2-cards long. Table must be 3-cards long")
    cases = pair_wr(deck, w=5)
    cases[:, 2:] = table
    comparisons = np.zeros(shape=6, dtype=np.int64)
    comb_comparisons = np.zeros(shape=(10, 6), dtype=np.int64)
    for case in cases:
        outs = case[:2]
        priv_deck = deck[~np.isin(deck, outs)]
        e = EquityCalc(priv_deck, case, '44')
        my_hand = np.concatenate((case, hand))
        my_comb = find_comb(my_hand)
        hand_histogram[my_comb[0]] += 1
        c_comp = e.compare_comb(my_comb, my_hand)
        comparisons[:5] += c_comp
        comparisons[5] += e.dnqual_count
        comb_comparisons[my_comb[0], :5] += c_comp
        comb_comparisons[my_comb[0], 5] += e.dnqual_count
        # print(comparisons)
    t2 = time()
    print(t2 - t1)
    div = np.sum(comb_comparisons, axis=1).reshape([10, 1])
    div[div == 0] = 1

    comb_equity = comb_comparisons / div
    overall_equity = comparisons / np.sum(comparisons)
    return overall_equity, comb_equity, hand_histogram / np.sum(hand_histogram)


def turn_equity(table, hand):
    t1 = time()
    deck = deck_from(table, hand)
    if table.size != 4 or hand.size != 2:
        raise ValueError("Hand or table length not compatible. "
                         "Hand is expected 2-cards long. Table must be 4-cards long")
    cases = np.empty(shape=[len(deck), 5], dtype=np.int8)
    cases[:, :4] = table
    cases[:, 4] = deck
    comparisons = np.zeros(shape=3, dtype=np.int64)
    for case in cases:
        out = case[4]
        priv_deck = deck[deck != out]
        e = EquityCalc(priv_deck, case)
        comparisons += e.compare(hand)
        # print(comparisons)
    t2 = time()
    print(t2 - t1)
    return comparisons / np.sum(comparisons)


def river_equity(table, hand):
    t1 = time()
    if table.size != 5 or hand.size != 2:
        raise ValueError("Hand or table length not compatible. "
                         "Hand is expected 2-cards long. Table must be 4-cards long")
    deck = deck_from(table, hand)
    e = EquityCalc(deck, table)
    t2 = time()
    print(t2 - t1)
    return e.equity(hand)
