from agents.agent import Agent
from environment.trade import Trade, TradeType
import numpy as np
import random
from typing import Optional


class EEE(Agent):
    def __init__(self, name: str, is_bank: bool = False, explore_prob: float = 0.1) -> None:
        super().__init__(name, is_bank=is_bank)
        self.explore_prob = explore_prob
        self.arms = [0, 1, 2]
        self.curr_trade = None
        self.m_e, self.n_e, self.s_e = {}, {}, {}
        self.in_phase, self.phase_counter, self.phase_rewards, self.n_i = False, 0, [], 0
        self.action_in_use = random.choice(self.arms)
        self.explore_prob = explore_prob

        for i in range(len(self.arms)):
            self.m_e[i] = 0
            self.n_e[i] = 0
            self.s_e[i] = 0

    def trade_finished(self, net_profit: float) -> None:
        self.phase_rewards.append(net_profit)
        self.curr_trade = None

    def place_trade(self, state: np.array, curr_price: float) -> Optional[Trade]:
        if self.curr_trade is not None:
            return None

        if self.in_phase:
            if self.phase_counter < self.n_i:
                self.phase_counter += 1

            else:
                avg_phase_reward = np.array(self.phase_rewards).mean() if len(self.phase_rewards) > 0 else 0
                self.n_e[self.action_in_use] += 1
                self.s_e[self.action_in_use] += self.n_i
                self.m_e[self.action_in_use] = self.m_e[self.action_in_use] + (
                    self.n_i / self.s_e[self.action_in_use]) * (avg_phase_reward - self.m_e[self.action_in_use])
                self.phase_rewards, self.phase_counter, self.n_i, self.in_phase = [], 0, 0, False

        if not self.in_phase:
            explore = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])

            if explore:
                new_action = random.choice(self.arms)

                self.action_in_use = new_action

            else:
                max_reward, actions_to_consider = max(list(self.m_e.values())), []

                for key, val in self.m_e.items():
                    if val == max_reward:
                        actions_to_consider.append(key)

                new_action = random.choice(actions_to_consider)

                self.action_in_use = new_action

            self.n_i, self.in_phase = np.random.choice(list(range(1, 10))), True

        if self.action_in_use == 0:
            return None

        trade_type = TradeType.BUY if self.action_in_use == 1 else TradeType.SELL
        open_price = curr_price
        stop_loss = (open_price - self.pips_to_risk) if self.action_in_use == 1 else (open_price + self.pips_to_risk)
        stop_gain = (open_price + self.pips_to_risk * self.risk_reward_ratio) if self.action_in_use == 1 else \
            (open_price - self.pips_to_risk * self.risk_reward_ratio)
        trade = Trade(trade_type, open_price, stop_loss, stop_gain, self.percent_to_risk)

        self.curr_trade = trade.trade_type

        return trade
