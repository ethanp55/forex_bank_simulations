from agents.agent import Agent
from environment.trade import Trade, TradeType
import numpy as np
from typing import Optional


class MACD(Agent):
    def __init__(self, name: str, is_bank: bool = False) -> None:
        super().__init__(name, is_bank=is_bank)
        self.curr_trade = None
        self.prev_macd, self.prev_macdsignal = None, None

    def trade_finished(self, net_profit: float) -> None:
        self.curr_trade = None

    def place_trade(self, state: np.array, curr_price: float, n_buys: int = 0, n_sells: int = 0) -> Optional[Trade]:
        if self.curr_trade is not None:
            return None

        macd, macdsignal = state[10, ], state[11, ]

        if self.prev_macd is None:
            self.prev_macd, self.prev_macdsignal = macd, macdsignal
            return None

        buy_signal = self.prev_macd < self.prev_macdsignal and macd > macdsignal
        sell_signal = self.prev_macd > self.prev_macdsignal and macd < macdsignal

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
