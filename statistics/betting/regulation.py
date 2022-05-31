from scr.statistics.smath import DTR, threshold, sinact, qact
import numpy as np
from abc import ABC, abstractmethod
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit


class ResponsibleGambling(ABC):
    """
    Класс, ответственный за регуляцию ставок.
    Он оперирует найденными в собранных данных соответствиями и предугадывает развитие событий,
    чтобы выбрать оптимальную ставку для текущей ситуации.
    """
    def __init__(self, sa, mn, mx, minchip):
        self.minchip = minchip
        self.sa = sa
        self.mn = mn
        self.mx = mx

    def chipify(self, stake):
        # превратить ставку в кратность минимального чипа
        chip_count = round(stake / self.minchip)
        return threshold(chip_count * self.minchip, self.mx, self.mn)

    @abstractmethod
    def regulate(self):
        pass


class StashGambling(ResponsibleGambling):
    def __init__(self, sa, minchip, mn, mx,
                 window=5, stash=0.3, n=2):
        ResponsibleGambling.__init__(self, sa, mn, mx, minchip)
        self.available_money = 0
        self.window = window
        self.stash = stash
        self.n = n

    def regulate(self):
        # то, сколько денег бот выиграл/проиграл в данном окне
        if len(self.sa.hand_balances) < self.window:
            win = len(self.sa.hand_balances)
        else:
            win = self.window
        available_money = np.sum(self.sa.hand_balances[-win:])

        if available_money > 0:
            # если он выиграл на повышение ставки, то вперед
            # исключение составляют деньги в "заначке" - бот должен оставить часть средств, как неприкосновенные
            stash_money = available_money * self.stash
            stake_money = available_money - stash_money
            new_stake = stake_money / self.n
            return self.chipify(new_stake)
        else:
            return self.mn



class PolynomialGambling(ResponsibleGambling):
    def __init__(self, sa,
                 window=20, mass=5, overweight=1, weights='quadratic',
                 df=4, mn=0.01, mx=0.10, minchip=0.01, bbl=3, alpha=0.5, beta=0.2, gamma=1.5):
        """
        :param sa: stats aggregator of the session
        :param window: number of elements from the stack trend to make an estimation from
        :param df: degrees of freedom of the polynomial
        :param mn: minimum stake
        :param mx: maximum stake
        :param minchip: minimum chip available at the table
        :param bbl: Big Blinds to Leave, in case there is a loosing prediction
        :param alpha: an attribute to a function which decides how big the bet will be in case of a winning window
        :param beta: min value of sinact function
        :param gamma:
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bbl = bbl
        self.df = df
        self.window = window
        self.overweight = overweight
        self.mass = mass
        self.masses = np.zeros(shape=mass, dtype=np.float)
        self.weights = np.arange(mass, 0, -1) / mass
        if weights == 'quadratic':
            self.weights = self.weights ** 2
        self.y_predict = []
        self.sin_predict = []
        ResponsibleGambling.__init__(self, sa, mn, mx, minchip)

    def fatten(self, m):
        self.masses[1:] = self.masses[0:-1]
        self.masses[0] = m

    def weigh(self):
        return self.overweight * np.sum(self.weights * self.masses)

    def regulate(self):
        slen = len(self.sa.stack_trend)
        if slen < 3:
            return self.sa.current_stake
        elif slen < self.window:
            win = slen
            x = np.arange(slen - win, slen)
            if slen < 5:
                coefs = polyfit(x, self.sa.stack_trend[slen - win:], deg=1)
            elif slen < 8:
                coefs = polyfit(x, self.sa.stack_trend[slen - win:], deg=2)
            else:
                coefs = polyfit(x, self.sa.stack_trend[slen - win:], deg=3)
        else:
            win = self.window
            x = np.arange(slen - win, slen)
            coefs = polyfit(x, self.sa.stack_trend[slen - win:], deg=self.df)
        window_balance = np.sum(np.array(self.sa.hand_balances[slen - win:]))
        fit = Polynomial(coefs)
        prediction = fit(slen)
        self.y_predict.append(prediction)

        prev_stack = self.sa.stack_trend[-1]
        prediction_dist = np.sqrt(1 + (prediction - prev_stack) ** 2)
        stack_diff = prediction - prev_stack
        prediction_sin = stack_diff / prediction_dist

        self.fatten(prediction_sin)
        w = self.weigh()

        print("-~*~-")
        print("balc: " + str(window_balance))
        print("pred: " + str(prediction))
        print("p_st: " + str(prev_stack))
        print("sdif: " + str(stack_diff))
        print("pdst: " + str(prediction_dist))
        print("psin: " + str(prediction_sin))
        prediction_sin = abs(prediction_sin)
        if stack_diff < 0:
            # тренд уходит вниз
            print("Тренд вниз.")
            if window_balance <= 0:
                # стабильное или проигрывающее окно
                print("Окно стабильно/проигрывает.")
                self.fatten(0)
                return self.mn
            else:
                # выигрывающее окно
                # уменьшить ставку, чтобы в случае поражения не потерять все сбережения (если возможно)
                max_stake = window_balance - self.bbl * self.mn
                if max_stake < self.mn:
                    return self.mn
                elif max_stake > self.mx:
                    max_stake = self.mx
                print("Окно выигрывает.")
                print("mxst: " + str(max_stake))
                stake = threshold((1 - prediction_sin) * self.sa.current_stake, upper=max_stake, bottom=self.mn)
                print(str(w))
                return self.chipify(w)
        elif stack_diff == 0:
            # тренд одинаков
            self.fatten(self.sa.current_stake)
            return self.sa.current_stake
        else:
            print("Тренд вверх.")
            # тренд уходит вверх
            if window_balance <= 0:
                # стабильное или проигрывающее окно
                # если тренд мощно уходит вверх, то пойдет большая ставка
                print("Окно стабильно/проигрывает.")
                stake = self.mx * qact(prediction_sin, p=self.gamma, beta=self.beta)
                self.fatten(stake)
                w = self.weigh()
                print(str(w))
                return self.chipify(w)
            else:
                print("Окно выигрывает.")
                # выигрывающее окно
                # если тренд мощный, ставка будет соответствующей
                # однако если он слишком мощный, то ставка будет чуть меньше максимальной
                stake = self.mx * sinact(prediction_sin, alpha=self.alpha, beta=self.beta)
                self.fatten(stake)
                w = self.weigh()
                print(str(w))
                return self.chipify(w)


class MinMaxRegulator(ResponsibleGambling):
    def __init__(self, sa, minchip=0.01, mn=0.01, mx=0.10, l1_len=10, l2_len=3):
        self.dtr = DTR(l1_len, l2_len, np.mean, np.mean)
        ResponsibleGambling.__init__(self, sa, mn, mx, minchip)

    def regulate(self):
        if len(self.sa.hand_balances) > self.dtr.act_len:
            self.dtr.put(self.sa.hand_balances[-self.dtr.act_len:])
            current = self.dtr.roll()
            self.dtr.put(self.sa.hand_balances[-self.dtr.act_len - 1:-1])
            previous = self.dtr.roll()
            r = current - previous
            if r < 0:
                return self.mn
            else:
                return self.mx
        return np.nan



