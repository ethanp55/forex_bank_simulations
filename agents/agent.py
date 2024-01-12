from environment.trade import Trade
import numpy as np
from typing import Optional


class Agent:
    def __init__(self, percent_to_risk: float = 0.02, pips_to_risk: float = 0.0050,
                 risk_reward_ratio: float = 1.5) -> None:
        self.percent_to_risk = percent_to_risk
        self.pips_to_risk = pips_to_risk
        self.risk_reward_ratio = risk_reward_ratio

    def trade_finished(self, net_profit: float) -> None:
        pass

    def place_trade(self, state: np.array, curr_price: float, n_buys: int = 0, n_sells: int = 0) -> Optional[Trade]:
        pass
