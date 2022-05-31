from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, \
    ElementNotInteractableException
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import numpy as np
from scr.game import card2int
from scr.games.livestream.parimatch.poker.statemachines import LivePokerSM

NO_ELEM = "no element"
DISABLED = "disabled"


class UnableToScrapeException(Exception):
    pass


class UnknownStateException(Exception):
    pass


class PM:
    # ожидание
    WAIT = -4
    # начальное положение, неопределенное до первого вызова функции state()
    IDLE = -3
    # ставки на неразданные карты
    BLIND_BETTING = -2
    # ставки закрыты, ожидание префлопа
    BLIND_INTERSTATE = -1
    # ставки на префлоп
    PREFLOP_BETTING = 0
    # ставки закрыты, ожидание флопа
    PREFLOP_INTERSTATE = 1
    # ставки на флоп
    FLOP_BETTING = 2
    # ставки закрыты, ожидание терна
    FLOP_INTERSTATE = 3
    # ставки на терн
    TURN_BETTING = 4
    # ставки закрыты, ожидание ривера
    TURN_INTERSTATE = 5
    # ривер
    RIVER = 6

    @staticmethod
    def state2str(s):
        if s == PM.WAIT:
            return "Ожидание"
        elif s == PM.IDLE:
            return "Начальное"
        elif s == PM.BLIND_BETTING:
            return "Принятие ставок на блайнд"
        elif s == PM.BLIND_INTERSTATE:
            return "Блайнд окончен"
        elif s == PM.PREFLOP_BETTING:
            return "Принятие ставок на префлоп"
        elif s == PM.PREFLOP_INTERSTATE:
            return "Префлоп окончен"
        elif s == PM.FLOP_BETTING:
            return "Принятие ставок на флоп"
        elif s == PM.FLOP_INTERSTATE:
            return "Флоп окончен"
        elif s == PM.TURN_BETTING:
            return "Принятие ставок на терн"
        elif s == PM.TURN_INTERSTATE:
            return "Терн окончен"
        elif s == PM.RIVER:
            return "Ривер"
        else:
            return str(s) + " - не определен"

    POKER = 3

    def __init__(self):
        self.player_num = 6
        driver = webdriver.Chrome()
        #driver.fullscreen_window()
        driver.delete_all_cookies()

        self.prev_state = PM.IDLE
        self.driver = driver
        self.frames = []
        self.sm = None

    def start(self):
        self.driver.get("https://air.pm.by/ru/betgames")
        #self.driver.execute_script("html")
        #self.driver.switch_to.frame(self.driver.find_element_by_id("betgames_div_iframe"))

    def quit(self):
        self.driver.quit()

    def choose_provider(self, i):
        navigation_menu = self.driver.find_element_by_class_name("head-section-main-menu")
        navigation = navigation_menu.find_element_by_class_name("navigation")
        items = navigation.find_elements_by_class_name("navigation__item")
        items[i].click()

    def choose_game(self, i):
        frame = self.driver.find_element_by_id("betgames_iframe_1")
        self.driver.switch_to.frame(frame)

        WebDriverWait(self.driver, 30).until(presence_of_element_located((By.CLASS_NAME, "games-navigation")))
        games_navigation = self.driver.find_element_by_class_name("games-navigation")
        tabs_bar = games_navigation.find_element_by_class_name("tabs-bar-scrollable")
        games = tabs_bar.find_elements_by_class_name("tabs-bar-item")
        games[i].click()

    def switch_off_video(self):
        WebDriverWait(self.driver, 30).until(presence_of_element_located((By.CLASS_NAME, "game-content")))
        game_content = self.driver.find_element_by_class_name("game-content")
        media_container = game_content.find_element_by_class_name("media-container")
        video_switcher = media_container.find_element_by_class_name("video-switcher")
        video_switcher.click()
        qualities = video_switcher.find_element_by_class_name("video-switcher-qualities")
        off_quality = qualities.find_elements_by_tag_name('li')[-1]
        try:
            off_quality.click()
        except ElementNotInteractableException:
            pass



    @staticmethod
    def str2rank(s):
        if s == 'q':
            return 'Q'
        elif s == 'a':
            return 'A'
        elif s == 'k':
            return 'K'
        elif s == 'j':
            return 'J'
        elif s == '10':
            return 'T'
        else:
            return s

    @staticmethod
    def str2suit(s):
        if s == 'spades':
            return 's'
        elif s == 'diamonds':
            return 'd'
        elif s == 'clubs':
            return 'c'
        elif s == 'hearts':
            return 'h'

    def hole_cards(self, i):
        screen_odds = self.driver.find_element_by_class_name("screen-odds-poker")
        player = screen_odds.find_element_by_class_name("player-" + str(i))
        odd_item = player.find_element_by_class_name("screen-odd-item")
        cards_group = odd_item.find_element_by_class_name("cards-group")
        try:
            cards = ""
            cards_num = []
            for n in range(2):
                card_elems = cards_group.find_elements_by_class_name("card")
                if len(card_elems) == 2:
                    elem_class = card_elems[n].get_attribute("class")
                    rank = card_elems[n].text
                    rank = PM.str2rank(rank)
                    suit = PM.str2suit(elem_class.split()[1])

                    card = rank + suit
                    cards += card + " "
                    cards_num.append(card2int(card))
            return cards, cards_num
        except NoSuchElementException:
            return NO_ELEM

    def all_holes(self):
        holes = ""
        holes_num = np.zeros(shape=self.player_num * 2, dtype=np.int8)
        for i in range(1, self.player_num + 1):
            player_holes, player_holes_num = self.hole_cards(i)
            if player_holes == NO_ELEM or player_holes.isspace() or player_holes == '':
                raise UnableToScrapeException("Player #" + str(i) + " cards value == " + player_holes)
            else:
                holes += player_holes
                holes_num[i*2-2:i*2] = player_holes_num
        return holes, holes_num

    def all_player_odds(self):
        odds = np.empty(shape=self.player_num, dtype=np.float64)
        for i in range(1, self.player_num + 1):
            player_odd = self.player_odds(i)
            odds[i-1] = player_odd
        return odds

    def all_player_stakes(self):
        stakes = np.empty(shape=self.player_num, dtype=np.float64)
        for i in range(1, self.player_num+1):
            player_stake = self.player_stake(i)
            stakes[i-1] = player_stake
        return stakes

    def player_stake(self, i):
        screen_odds = self.driver.find_element_by_class_name("screen-odds-poker")
        player = screen_odds.find_element_by_class_name("player-" + str(i))
        try:
            poker_bet_sum = player.find_element_by_class_name("poker-bet-sum").text[:-3]
            print(poker_bet_sum)
            if poker_bet_sum.isspace() or poker_bet_sum == '':
                return 0.0
            return float(poker_bet_sum)
        except NoSuchElementException:
            return 0.0

    def player_odds(self, i):
        screen_odds = self.driver.find_element_by_class_name("screen-odds-poker")
        player = screen_odds.find_element_by_class_name("player-" + str(i))
        odd_item = player.find_element_by_class_name("screen-odd-item")
        if "disabled" in odd_item.get_attribute("class"):
            return 0.0
        try:
            odds = float(odd_item.find_element_by_class_name("odd-value").text)
            return odds
        except NoSuchElementException:
            return 0.0

    def table_cards(self):
        tc = self.driver.find_element_by_class_name("table-cards")
        cards_group = tc.find_element_by_class_name("cards-group")
        try:
            card_elems = cards_group.find_elements_by_class_name("card")
            cards = ""
            num_cards = []
            for elem in card_elems:
                card = PM.str2rank(elem.text) + PM.str2suit(elem.get_attribute("class").split()[1])
                num_cards.append(card2int(card))
                cards += card + " "
            return cards, np.array(num_cards, dtype=np.int8)
        except NoSuchElementException:
            raise UnableToScrapeException("No cards found on the table.")

    def configure_tabs(self, i):
        betting_container = self.driver.find_element_by_class_name("betting-container")
        odds_tabs = betting_container.find_element_by_class_name("tabs-bar")
        tabs_bar_scrollable = odds_tabs.find_element_by_class_name("tabs-bar-scrollable")
        tabs = tabs_bar_scrollable.find_elements_by_class_name("tabs-bar-item")
        tabs[i].click()

    def comb_odds(self):
        betting_container = self.driver.find_element_by_class_name("betting-container")
        odds_list = betting_container.find_element_by_class_name("odds-list")
        odds_elems = odds_list.find_elements_by_class_name("odd-item")
        odd_values = np.empty(shape=10, dtype=np.float64)
        for odd_elem, comb_rank in zip(odds_elems, range(10)):
            if "disable" in odd_elem.get_attribute("class"):
                odd_values[comb_rank] = 0
            else:
                odd_item_info = odd_elem.find_element_by_class_name("odd-item-info")
                try:
                    odd_value = float(odd_item_info.find_element_by_class_name("odd-value").text)
                    odd_values[comb_rank] = odd_value
                except NoSuchElementException:
                    odd_values[comb_rank] = 0.0
        return odd_values

    def time(self):
        timer = self.driver.find_element_by_class_name("live-game-timer")
        if "visible" in timer.get_attribute("class"):
            t = timer.find_element_by_class_name("time-string")
            return int(t.text)
        else:
            return -1

    def previous_state(self):
        return self.prev_state

    def state(self):
        try:
            try:
                hole_cards = self.all_holes()
            except UnableToScrapeException:
                # нет карт игроков
                hand_odds = self.all_player_odds()
                comb_odds = self.comb_odds()

                if np.all(hand_odds == 0) or np.all(comb_odds == 0):
                    self.prev_state = PM.BLIND_INTERSTATE
                    return PM.BLIND_INTERSTATE,
                self.prev_state = PM.BLIND_BETTING
                return PM.BLIND_BETTING,
            table_cards = self.table_cards()
            if table_cards[1].size == 3:
                player_odds = self.all_player_odds()
                comb_odds = self.comb_odds()
                if np.all(player_odds == 0) or np.all(comb_odds == 0):
                    self.prev_state = PM.FLOP_INTERSTATE
                    return PM.FLOP_INTERSTATE, table_cards
                self.prev_state = PM.FLOP_BETTING
                return PM.FLOP_BETTING, table_cards, player_odds, comb_odds
            elif table_cards[1].size == 4:
                player_odds = self.all_player_odds()
                comb_odds = self.comb_odds()
                if np.all(player_odds == 0) or np.all(comb_odds == 0):
                    self.prev_state = PM.TURN_INTERSTATE
                    return PM.TURN_INTERSTATE, table_cards
                self.prev_state = PM.TURN_BETTING
                return PM.TURN_BETTING, table_cards, player_odds, comb_odds
            elif table_cards[1].size == 5:
                self.prev_state = PM.RIVER
                return PM.RIVER, table_cards
            else:
                hand_odds = self.all_player_odds()
                comb_odds = self.comb_odds()
                if np.all(hand_odds == 0) or np.all(comb_odds == 0):
                    self.prev_state = PM.PREFLOP_INTERSTATE
                    return PM.PREFLOP_INTERSTATE,
                self.prev_state = PM.PREFLOP_BETTING
                return PM.PREFLOP_BETTING, hole_cards, hand_odds, comb_odds
        except StaleElementReferenceException:
            return PM.WAIT,

    def collect_state(self):
        prev_state = self.prev_state
        state = self.state()
        state_found = state[0]
        print("Стол находится в состоянии <" + PM.state2str(state_found) + ">.")
        if state_found == PM.WAIT:
            print("Один из искомых элементов отсутствует на экране. Ожидание...")
            return False
        if state_found == PM.BLIND_BETTING:
            if self.sm is None or prev_state == PM.RIVER:
                self.sm = LivePokerSM()
            self.sm.blind_betting()
            return True
        elif self.sm is not None:
            if state_found == PM.BLIND_INTERSTATE:
                self.sm.blind_over()
            elif state_found == PM.PREFLOP_BETTING:
                hole_cards = state[1]
                hand_odds = state[2]
                comb_odds = state[3]
                self.sm.preflop_betting(hole_cards, hand_odds, comb_odds)
            elif state_found == PM.PREFLOP_INTERSTATE:
                self.sm.preflop_over()
            elif state_found == PM.FLOP_BETTING:
                table_cards = state[1]
                hand_odds = state[2]
                comb_odds = state[3]
                self.sm.flop_betting(table_cards, hand_odds, comb_odds)
            elif state_found == PM.FLOP_INTERSTATE:
                self.sm.flop_over()
            elif state_found == PM.TURN_BETTING:
                table_cards = state[1]
                hand_odds = state[2]
                comb_odds = state[3]
                self.sm.turn_betting(table_cards, hand_odds, comb_odds)
            elif state_found == PM.TURN_INTERSTATE:
                self.sm.turn_over()
            elif state_found == PM.RIVER:
                table_cards = state[1]
                self.sm.river(table_cards)
            return True
        else:
            print("Ожидание блайнда...")
            return False


class Game:
    # начальное положение, неопределенное до первого вызова функции state()
    IDLE = 0
    # ставки на неразданные карты
    BLIND_BETTING = 1
    # ставки закрыты, ожидание префлопа
    BLIND_INTERSTATE = 2
    # ставки на префлоп
    PREFLOP_BETTING = 3
    # ставки закрыты, ожидание флопа
    PREFLOP_INTERSTATE = 4
    # ставки на флоп
    FLOP_BETTING = 5
    # ставки закрыты, ожидание терна
    FLOP_INTERSTATE = 6
    # ставки на терн
    TURN_BETTING = 7
    # ставки закрыты, ожидание ривера
    TURN_INTERSTATE = 8
    # ривер
    RIVER = 9

