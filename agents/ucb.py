from agents.agent import Agent
from environment.trade import Trade, TradeType
import numpy as np
import random
from typing import Optional


class UCB(Agent):
    def __init__(self, delta: float = 0.99, is_bank: bool = False, discount_factor: float = 0.99) -> None:
        super().__init__()
        self.delta = delta
        self.is_bank = is_bank
        self.discount_factor = discount_factor
        self.arms = [0, 1, 2]
        self.arm_idx = None
        self.empirical_rewards = [0, 0, 0]
        self.n_samples = [0, 0, 0]
        self.first_action = True

    def trade_finished(self, net_profit: float) -> None:
        self.empirical_rewards[self.arm_idx] += net_profit
        self.arm_idx = None

    def place_trade(self, state: np.array, curr_price: float, n_buys: int = 0, n_sells: int = 0) -> Optional[Trade]:
        if self.arm_idx is not None and not self.is_bank:
            return None

        elif self.arm_idx is not None:
            trade_type = TradeType.BUY if self.arm_idx == 0 else TradeType.SELL

            open_price = curr_price
            stop_loss = (open_price - self.pips_to_risk) if self.arm_idx == 1 else (open_price + self.pips_to_risk)
            stop_gain = (open_price + self.pips_to_risk * self.risk_reward_ratio) if self.arm_idx == 1 else \
                (open_price - self.pips_to_risk * self.risk_reward_ratio)
            trade = Trade(trade_type, open_price, stop_loss, stop_gain, self.percent_to_risk)

            return trade

        predictions = [0, 0, 0]

        for arm in self.arms:
            n_samples = self.n_samples[arm]
            arm_modifier = -1 if self.is_bank else 0

            if n_samples == 0:
                predictions[arm] = np.inf if arm > arm_modifier else 0

            else:
                empirical_avg = self.empirical_rewards[arm] / n_samples
                upper_bound = ((2 * np.log(1 / self.delta)) / n_samples) ** 0.5
                predictions[arm] = (empirical_avg + upper_bound) if arm > arm_modifier else 0

        predictions = predictions[0:-1] if self.is_bank else predictions

        if self.first_action:
            self.arm_idx = random.choice(self.arms)
            self.first_action = False

        else:
            self.arm_idx = predictions.index(max(predictions))

        arm = self.arms[self.arm_idx]
        self.n_samples[self.arm_idx] += 1

        if not self.is_bank and arm == 0:
            return None

        arm_modifier = 0 if self.is_bank else 1
        trade_type = TradeType.BUY if arm == arm_modifier else TradeType.SELL

        open_price = curr_price
        stop_loss = (open_price - self.pips_to_risk) if arm == arm_modifier else (open_price + self.pips_to_risk)
        stop_gain = (open_price + self.pips_to_risk * self.risk_reward_ratio) if arm == arm_modifier else \
            (open_price - self.pips_to_risk * self.risk_reward_ratio)
        trade = Trade(trade_type, open_price, stop_loss, stop_gain, self.percent_to_risk)

        return trade
