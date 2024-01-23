from agents.agent import Agent
from collections import deque
from environment.trade import Trade, TradeType
import numpy as np
from typing import Optional, Tuple


class MACDStochastic(Agent):
    def __init__(self, name: str, is_bank: bool = False, lookback: int = 12) -> None:
        super().__init__(name, is_bank=is_bank)
        self.curr_trade = None
        self.prev_macd, self.prev_macdsignal = None, None
        self.slowk_vals, self.slowd_vals = deque(maxlen=lookback), deque(maxlen=lookback)

    def trade_finished(self, net_profit: float) -> None:
        self.curr_trade = None

    def place_trade(self, state: np.array, curr_price: float) -> Optional[Trade]:
        if self.curr_trade is not None:
            return None

        slowk, slowd, macd, macdsignal = state[8, ], state[9, ], state[10, ], state[11, ]

        self.slowk_vals.append(slowk)
        self.slowd_vals.append(slowd)

        if self.prev_macd is None:
            self.prev_macd, self.prev_macdsignal = macd, macdsignal
            return None

        buy_signal = self.prev_macd < self.prev_macdsignal and macd > macdsignal
        sell_signal = self.prev_macd > self.prev_macdsignal and macd < macdsignal

        def _check_for_stochastic_cross() -> Tuple[bool, bool]:
            cross_up, cross_down = False, False

            for i in range(len(self.slowk_vals) - 1, 0, -1):
                slowk2, slowd2 = self.slowk_vals[i - 1], self.slowd_vals[i - 1]
                slowk1, slowd1 = self.slowk_vals[i], self.slowd_vals[i]

                if slowk2 < slowd2 and slowk1 > slowd1 and max([slowk2, slowd2, slowk1, slowd1]) < 20:
                    cross_up = True

                elif slowk2 > slowd2 and slowk1 < slowd1 and min([slowk2, slowd2, slowk1, slowd1]) > 80:
                    cross_down = True

            return cross_up, cross_down

        if buy_signal or sell_signal:
            stoch_cross_up, stoch_cross_down = _check_for_stochastic_cross()
            buy_signal = buy_signal and stoch_cross_up
            sell_signal = sell_signal and stoch_cross_down

        self.prev_macd, self.prev_macdsignal = macd, macdsignal

        if not (buy_signal or sell_signal):
            return None

        trade_type = TradeType.BUY if buy_signal else TradeType.SELL
        open_price = curr_price
        stop_loss = (open_price - self.pips_to_risk) if buy_signal else (open_price + self.pips_to_risk)
        stop_gain = (open_price + self.pips_to_risk * self.risk_reward_ratio) if buy_signal else \
            (open_price - self.pips_to_risk * self.risk_reward_ratio)
        trade = Trade(trade_type, open_price, stop_loss, stop_gain, self.percent_to_risk)

        self.curr_trade = trade.trade_type

        return trade
