import numpy as np
from abc import ABC, abstractmethod
from scr.basicfunc import is_sorted_desc, dist


class PokerComb(ABC):
    N = -1

    @staticmethod
    @abstractmethod
    def find(hand):
        pass

    @staticmethod
    @abstractmethod
    def weaker_combs():
        pass

    @staticmethod
    def check_type(other):
        if not isinstance(other, PokerComb):
            error_str = "Can't compare PokerComb to " + str(type(other))
            raise(TypeError(error_str))

    @abstractmethod
    def greater_than(self, other):
        pass

    @abstractmethod
    def equal_to(self, other):
        pass

    def __gt__(self, other):
        PokerComb.check_type(other)
        if type(other) in self.weaker_combs():
            # combination is weaker
            return True
        elif type(self) == type(other):
            return self.greater_than(other)
        else:
            # combination is stronger
            return False

    def __eq__(self, other):
        PokerComb.check_type(other)
        if type(self) == type(other):
            return self.equal_to(other)
        return False


class OneRankComb(PokerComb):
    def __init__(self, card, kicker):
        self.card = card
        self.kicker = kicker

    @staticmethod
    @abstractmethod
    def find(hand):
        pass

    @staticmethod
    @abstractmethod
    def weaker_combs():
        pass

    def greater_than(self, other):
        if self.card > other.card:
            return True
        elif self.card == other.card:
            return self.kicker > other.kicker
        return False

    def equal_to(self, other):
        return self.card == other.card and self.kicker == other.kicker


class HandSizeComb(PokerComb):
    @staticmethod
    def weaker_combs():
        pass

    @staticmethod
    def find(hand):
        pass

    def __init__(self, cards):
        if not is_sorted_desc(cards):
            cards = cards % 13
            self.cards = np.sort(cards)[::-1]
        else:
            self.cards = cards

    def greater_than(self, other):
        for a, b in zip(self.cards, other.cards):
            if a > b:
                return True
            elif a < b:
                return False
        return False

    def equal_to(self, other):
        return np.all(self.cards == other.cards)


class MultiRankComb(PokerComb):
    @staticmethod
    def find(hand):
        pass

    @staticmethod
    def weaker_combs():
        pass

    def __init__(self, cards, kicker):
        if not is_sorted_desc(cards):
            self.cards = np.sort(cards)[::-1]
        else:
            self.cards = cards
        self.kicker = kicker

    def greater_than(self, other):
        for a, b in zip(self.cards, other.cards):
            if a > b:
                return True
            elif a < b:
                return False
        # all cards equal -> check kickers
        return self.kicker > other.kicker

    def equal_to(self, other):
        return np.all(self.cards == other.cards) and self.kicker == other.kicker


class Kicker:
    def __init__(self, cards):
        if not is_sorted_desc(cards):
            self.cards = np.sort(cards)[::-1]
        else:
            self.cards = cards

    def __gt__(self, other):
        for a, b in zip(self.cards, other.cards):
            if b > a:
                return False
            elif a > b:
                return True
        return False

    def __eq__(self, other):
        return np.all(self.cards == other.cards)


class HighCard(OneRankComb):
    N = 0

    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)
        hand = hand % 13
        hand_sorted = np.sort(hand)[::-1]
        card = hand_sorted[0]
        kicker = hand_sorted[1:]

        hc = HighCard(card, Kicker(kicker))
        hc.hand_sorted = hand_sorted
        return hc

    @staticmethod
    def weaker_combs():
        return []


class Pair(OneRankComb):
    N = 1

    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)
        hand = hand % 13
        count = np.bincount(hand)
        pairs = np.nonzero(count == 2)[::-1]
        # wtf
        pairs = pairs[0]
        if len(pairs) > 0:
            wo_pairs = list(filter(lambda x: x not in pairs, hand))
            wo_pairs = np.array(wo_pairs)
            wo_pairs[::-1].sort()
            return Pair(pairs[0], Kicker(wo_pairs))
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard]


class TwoPair(MultiRankComb):
    N = 2
    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)
        hand = hand % 13
        count = np.bincount(hand)
        pairs = np.nonzero(count == 2)[::-1]
        # get it off
        pairs = pairs[0]
        if len(pairs) > 1:
            wo_pairs = list(filter(lambda x: x not in pairs, hand))
            wo_pairs = np.array(wo_pairs)
            return TwoPair(np.array([pairs[0], pairs[1]]), Kicker(wo_pairs))
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair]


class Set(OneRankComb):
    N = 3
    @staticmethod
    def find(hand):
        hand = hand % 13
        count = np.bincount(hand)
        sets = np.nonzero(count == 3)[::-1]
        # it's funny
        sets = sets[0]

        if len(sets) > 0:
            wo_trips = list(filter(lambda x: x not in sets, hand))
            wo_trips = np.array(wo_trips)
            wo_trips[::-1].sort()
            return Set(sets[0], kicker=Kicker(wo_trips))
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair, TwoPair]


class Straight(PokerComb):
    N = 4
    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)
        hand = hand % 13
        hand.sort()

        ace = 12

        if ace in hand:
            np.insert(arr=hand, obj=0, values=ace)

        first = hand[0]
        prev = hand[0]
        count = 1

        for i in range(len(hand)):
            card = hand[i]
            d = dist(card, prev)
            if d > 1:
                if len(hand) - i - 1 < 4:
                    return None
                count = 1
            elif d == 1:
                count += 1
                if count == 5:
                    return Straight(max(first, card))
            prev = card
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair, TwoPair, Set]

    def __init__(self, top_card):
        self.top_card = top_card

    def greater_than(self, other):
        return self.top_card > other.top_card

    def equal_to(self, other):
        return self.top_card == other.top_card


class Flush(HandSizeComb):
    N = 5

    def __init__(self, cards, suit):
        HandSizeComb.__init__(self, cards)
        self.suit = suit

    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)

        h = []
        d = []
        s = []
        c = []

        suits = [h, d, s, c]

        for card in hand:
            suit = card // 13
            rank = card % 13
            suits[suit].append(rank)

        suit = 0
        for cards in suits:
            if len(cards) > 4:
                cards = np.array(cards)
                cards[::-1].sort()
                return Flush(cards, suit)
            suit += 1

        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair, TwoPair, Set, Straight]


class FullHouse(PokerComb):
    N = 6
    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)
        hand = hand % 13
        count = np.bincount(hand)
        pairs = np.nonzero(count == 2)[::-1]
        sets = np.nonzero(count == 3)[::-1]
        # really what's the reason??
        pairs = pairs[0]
        sets = sets[0]
        if len(pairs) > 0 and len(sets) > 0:
            return FullHouse(pairs[0], sets[0])
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair, TwoPair, Set, Straight, Flush]

    def __init__(self, pair_card, set_card):
        self.pair_card = pair_card % 13
        self.set_card = set_card % 13

    def greater_than(self, other):
        if self.set_card > other.set_card:
            return True
        elif self.set_card == other.set_card:
            return self.pair_card > other.pair_card
        return False

    def equal_to(self, other):
        return self.set_card == other.set_card and self.pair_card == other.pair_card


class Quads(OneRankComb):
    N = 7

    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)
        hand = hand % 13
        count = np.bincount(hand)
        q = np.nonzero(count == 4)[::-1]
        # for some reason numpy is returning a tuple
        q = q[0]
        if len(q) > 0:
            wo_quads = list(filter(lambda x: x not in q, hand))
            return Quads(q[0], Kicker(np.array(wo_quads)))
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair, TwoPair, Set, Straight, Flush, FullHouse]


class StraightFlush(Straight):
    N = 8

    @staticmethod
    def find(hand):
        hand = np.asarray(hand, dtype=np.int8)

        h = []
        d = []
        s = []
        c = []

        suits = [h, d, s, c]

        for card in hand:
            suit = card // 13
            rank = card % 13
            suits[suit].append(rank)

        count = 0
        for suit in suits:
            if len(suit) > 4:
                st = Straight.find(np.array(suit))
                if st is not None:
                    return StraightFlush(st.top_card, count)
            count += 1
        return None

    @staticmethod
    def weaker_combs():
        return [HighCard, Pair, TwoPair, Set, Straight, Flush, FullHouse, Quads]

    def __init__(self, top_card, suit):
        Straight.__init__(self, top_card)
        self.suit = suit


comb_num = {
    HighCard: 0,
    Pair: 1,
    TwoPair: 2,
    Set: 3,
    Straight: 4,
    Flush: 5,
    FullHouse: 6,
    Quads: 7,
    StraightFlush: 8
}
