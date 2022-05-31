from queue import Queue
import numpy as np
from scr.basicfunc import dist, conseq
from datetime import datetime


hearts = 'h'
diamonds = 'd'
spades = 's'
clubs = 'c'

ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
nums = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


# def generate_deck():
#     d = []
#     for suit in suits:
#         for rank in ranks:
#             signature = rank + suit
#             d.append(Card(signature))
#     return d


def str2cards(s, joker=False):
    cards = s.split()
    card_ints = []
    for card in cards:
        if joker:
            if card == 'X':
                card_ints.append(52)
            else:
                card_ints.append(card2int(card))
        else:
            card_ints.append(card2int(card))
    return np.array(card_ints, dtype=np.int8)


def str_to_deck(outs):
    table = outs.split()
    fulldeck = np.arange(52)
    mask = np.ones_like(fulldeck, dtype=np.bool)
    t = []
    for card in table:
        rank = ranks[card[0]]
        suit = suit_to_int(card[1])
        mask[13 * suit + rank] = False
        t.append(13 * suit + rank)
    return fulldeck[mask], np.array(t, dtype=np.int)


def deck_wo_dead_cards(deads):
    fulldeck = np.arange(52)
    return fulldeck[~np.isin(fulldeck, deads)]


def deck_from(table, hand=None):
    fulldeck = np.arange(52)
    if hand is not None:
        deck = fulldeck[~np.isin(fulldeck, table) & ~np.isin(fulldeck, hand)]
    else:
        deck = fulldeck[~np.isin(fulldeck, table)]
    return deck


def gen_deck(table, hand=None):
    if hand is None:
        table = table.split()
        fulldeck = np.arange(52)
        mask = np.ones_like(fulldeck, dtype=np.bool)
        for card in table:
            rank = ranks[card[0]]
            suit = suit_to_int(card[1])
            mask[13 * suit + rank] = False
        return fulldeck[mask], fulldeck[~mask], []
    else:
        table = table.split()
        hand = hand.split()
        fulldeck = np.arange(52)
        deck_mask = np.ones_like(fulldeck, dtype=np.bool)
        table_mask = np.zeros_like(fulldeck, dtype=np.bool)
        hand_mask = np.zeros_like(fulldeck, dtype=np.bool)
        for card in table:
            rank = ranks[card[0]]
            suit = suit_to_int(card[1])
            deck_mask[13 * suit + rank] = False
            table_mask[13 * suit + rank] = True
        for card in hand:
            rank = ranks[card[0]]
            suit = suit_to_int(card[1])
            deck_mask[13 * suit + rank] = False
            hand_mask[13 * suit + rank] = True
        return fulldeck[deck_mask], fulldeck[table_mask], fulldeck[hand_mask]


def max_elements_bin(arr, n):
    if len(arr) < n:
        raise(ValueError("Cannot find " + str(n) + " max elements in " + str(len(arr)) + " long sequence."))
    elif len(arr) == n:
        return arr
    else:
        count = np.bincount(arr)
        last = len(count) - 1
        i = last
        res = []
        while len(res) < n:
            for j in range(count[i]):
                if len(res) < n:
                    res.append(i)
                else:
                    return res
            i -= 1
        return res

def straight_possible(cards, complement, straight_len=5):
    """
    :param cards: array of cards
    :param complement: maximum number of complementary cards
    :param straight_len: straight size
    :return: True if straight can be completed using given number of complement cards
             False otherwise
    """
    required = straight_len - complement
    j = 0
    sequence_len = 1

    ace = 12

    cs = Queue()
    c_sum = 0

    if ace in cards:
        c = conseq(ace, cards[0])
        cs.put(c)
        c_sum += c
        sequence_len += 1
        j = -1

    for i in range(len(cards) - 1):
        if len(cards) - j < required:
            return False
        elif sequence_len >= required and c_sum <= complement:
            return True
        c = conseq(cards[i], cards[i + 1])
        cs.put(c)
        c_sum += c
        sequence_len += 1
        if c_sum > complement:
            while not cs.empty() and c_sum > complement:
                c_sum -= cs.get()
                j += 1
                sequence_len -= 1

    if sequence_len >= required and c_sum <= complement:
        return True
    return False


def straight_gap(cards):
    """
    :param cards: MUST BE SORTED ASCENDINGLY!!!
    :return: gap value of elements in the straight sequence
    """
    cards = cards % 13
    cards = np.unique(cards)
    cards.sort()
    conseq_count = np.zeros_like(cards)
    for n in range(cards.size - 1):
        conseq_count[n+1] += conseq(cards[n], cards[n+1])
    if cards[-1] == 12:
        tail_conseq_count = np.zeros_like(cards)
        tail_cards = np.empty_like(cards)
        tail_cards[0] = cards[-1]
        tail_cards[1:] = cards[:-1]
        for n in range(tail_cards.size - 1):
            tail_conseq_count[n] += conseq(tail_cards[n], tail_cards[n+1])
        if np.sum(conseq_count) > np.sum(tail_conseq_count):
            return tail_conseq_count
        else:
            return conseq_count
    return conseq_count


def hand_to_str(hand):
    str = ""
    for h in hand:
        suit = h // 13
        rank = h % 13
        suit = int_to_suit(suit)
        rank = nums[rank]
        str += rank + suit + " "
    return str


def min_straight_comp(cards):
    cards = cards % 13
    cards = np.unique(cards)
    cards.sort()
    conseq_sum = 0
    for n in range(cards.size - 1):
        conseq_sum += conseq(cards[n], cards[n + 1])
    if cards[-1] == 12:
        tail_conseq_sum = 0
        tail_cards = np.empty_like(cards)
        tail_cards[0] = cards[-1]
        tail_cards[1:] = cards[:-1]
        for n in range(tail_cards.size - 1):
            tail_conseq_sum += conseq(tail_cards[n], tail_cards[n + 1])
        conseq_sum = min(conseq_sum, tail_conseq_sum)
    return conseq_sum


def is_straight(cards):
    cards = cards % 13
    cards = np.unique(cards)
    if len(cards) < 5:
        return False, -1
    else:
        cards = np.sort(cards)
        wheel = np.array([12, 3, 2, 1, 0])
        if np.all(np.isin(wheel, cards)):
            return True, 12
        count = 0
        toprank = -1
        topcount = 0
        for i in range(len(cards) - 1):
            d = dist(cards[i], cards[i + 1])
            if d != 1:
                count = 0
            else:
                count += 1
            if count >= 4:
                topcount = count
                toprank = cards[i + 1]
        if topcount > 0:
            return True, toprank
        return False, -1



def predict_win(player_comb, table, known):
    h = []
    d = []
    s = []
    c = []

    suits = [h, d, s, c]
    for card in table:
        suit = card // 13
        rank = card % 13
        suits[suit].append(rank)

    p_flushes = []
    lens = [len(x) for x in suits]
    # checking for possibility of a flush/straight flush
    for n in range(len(lens)):
        if lens[n] > 2:
            p_flushes.append(suits[n])

def date_time():
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    return datetime_str


class Room:
    def __init__(self, window, name, room_size, money_type, bb, game):
        self.window = window
        self.name = name
        self.room_size = room_size
        self.money_type = money_type
        self.big_blind = bb
        self.game = game

        descendants = self.window.descendants()

        for descendant in descendants:
            if descendant.class_name() == "PokerStarsChatClass":
                self.chat = descendant
                break

        for descendant in descendants:
            if descendant.class_name() == "PokerStarsChatEditorClass":
                self.chat_editor = descendant
                break


class Player:
    def __init__(self, nickname):
        self.nickname = nickname


def int_to_suit(i):
    if i == 0:
        return hearts
    elif i == 1:
        return diamonds
    elif i == 2:
        return spades
    elif i == 3:
        return clubs
    else:
        raise(ValueError("No such suit"))


def suit_to_int(suit):
    if suit == hearts:
        return 0
    elif suit == diamonds:
        return 1
    elif suit == spades:
        return 2
    elif suit == clubs:
        return 3
    else:
        raise(ValueError("No such suit as " + suit))


def suit2word(i):
    if i == 0:
        return 'hearts'
    elif i == 1:
        return 'diamonds'
    elif i == 2:
        return 'spades'
    elif i == 3:
        return 'clubs'
    else:
        raise(ValueError("No such suit under # " + str(i)))


def int2rank(i):
    return nums[i]


def card2int(card):
    r, s = card[0], card[1]
    ri = ranks[r]
    si = suit_to_int(s)
    return ri + 13 * si
