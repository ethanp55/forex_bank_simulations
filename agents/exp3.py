from agents.agent import Agent
from environment.trade import Trade, TradeType
import numpy as np
import random
from typing import Optional


class EXP3(Agent):
    def __init__(self, name: str, is_bank: bool = False, gamma: float = 0.5) -> None:
        super().__init__(name, is_bank=is_bank)
        self.arms = [0, 1, 2]
        self.curr_trade = None
        self.empirical_rewards, self.weights, self.p_t_vals = {}, {}, None

        for arm in self.arms:
            self.empirical_rewards[arm] = 0
            self.weights[arm] = 1.0

        self.gamma, self.k = gamma, len(self.arms)

    def trade_finished(self, net_profit: float) -> None:
        x_hat = net_profit / self.p_t_vals[self.curr_trade]
        self.weights[self.curr_trade] = self.weights[self.curr_trade] * np.exp((self.gamma * x_hat) / self.k)
        self.curr_trade = None

    def place_trade(self, state: np.array, curr_price: float, n_buys: int = 0, n_sells: int = 0) -> Optional[Trade]:
        if self.curr_trade is not None:
            return None

        weight_sum = sum(self.weights.values())
        self.p_t_vals = [((1 - self.gamma) * (weight / weight_sum)) + (self.gamma / self.k) for weight in
                         self.weights.values()]
        action = random.choices(list(self.weights.keys()), weights=self.p_t_vals, k=1)[0]

        if action == 0:
            return None

        trade_type = TradeType.BUY if action == 1 else TradeType.SELL
        open_price = curr_price
        stop_loss = (open_price - self.pips_to_risk) if action == 1 else (open_price + self.pips_to_risk)
        stop_gain = (open_price + self.pips_to_risk * self.risk_reward_ratio) if action == 1 else \
            (open_price - self.pips_to_risk * self.risk_reward_ratio)
        trade = Trade(trade_type, open_price, stop_loss, stop_gain, self.percent_to_risk)

        self.curr_trade = action

        return trade
