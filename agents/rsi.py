from agents.agent import Agent
from environment.trade import Trade, TradeType
import numpy as np
from typing import Optional


class RSI(Agent):
    def __init__(self, name: str, is_bank: bool = False) -> None:
        super().__init__(name, is_bank=is_bank)
        self.curr_trade = None

    def trade_finished(self, net_profit: float) -> None:
        self.curr_trade = None

    def place_trade(self, state: np.array, curr_price: float, n_buys: int = 0, n_sells: int = 0) -> Optional[Trade]:
        if self.curr_trade is not None:
            return None

        rsi = state[6, ]

        buy_signal = rsi < 20
        sell_signal = rsi > 80

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
