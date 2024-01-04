from collections import deque
from environment.market_calculations import MarketCalculations
from environment.trade import Trade, TradeType
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class State:
    def __init__(self, num_agents: int, curr_price: float = 1.0000, max_iterations: int = 100,
                 starting_balance: float = 10000.0, bank_balance_multiplier: float = 1.0) -> None:
        self.num_agents = num_agents
        self.bank_agent_index = num_agents - 1
        self.prices = deque(maxlen=100)
        price = curr_price
        for i in range(50):
            self.prices.append(price)

            if i % 2 == 0:
                price += 0.0001

            else:
                price -= 0.0001

        self.max_iterations, self.n_iterations = max_iterations, 0
        self.agent_balances = [starting_balance] * self.num_agents
        # Bank balance
        self.agent_balances[self.bank_agent_index] = starting_balance * bank_balance_multiplier * (self.num_agents - 1)
        self.market_value = sum(self.agent_balances)
        self.open_trades = {}

    def update_balances(self, agent_index: int, trade_amount: float) -> None:
        self.agent_balances[agent_index] += trade_amount
        self.agent_balances[self.bank_agent_index] += -trade_amount  # The bank gains the opposite of the trader

    def step(self, trades: List[Optional[Trade]]) -> Tuple[np.array, List[float], bool]:
        # Two main phases:
        # - Update the price based on agent actions
        # - Use the updated price to check if any existing trades close out
        #   - This is used to calculate the rewards for each agent

        # Phase 1 (update the price)
        new_price = self.curr_price()

        for agent_index in range(self.num_agents):
            agent_trade, agent_balance = trades[agent_index], self.agent_balances[agent_index]

            # New trade order and the agent doesn't have an existing trade
            if agent_trade is not None and self.open_trades.get(agent_index, None) is None:
                percent_to_risk, trade_type = agent_trade.percent_to_risk, agent_trade.trade_type
                dollar_amount = agent_balance * percent_to_risk
                dollar_amount = (dollar_amount * 0.0001) / self.num_agents
                new_price = (new_price + dollar_amount) if trade_type is TradeType.BUY else (new_price - dollar_amount)

                if agent_index != self.bank_agent_index:
                    self.open_trades[agent_index] = agent_trade

        self.prices.append(new_price)

        # Phase 2 (check if existing trades should close out and calculate rewards)
        rewards = [0] * self.num_agents

        for agent_index in range(self.num_agents - 1):
            existing_trade = self.open_trades.get(agent_index, None)

            if existing_trade is not None:
                agent_balance = self.agent_balances[agent_index]
                trade_amount, trade_closed = \
                    MarketCalculations.calculate_trade_amount(existing_trade, new_price, agent_balance)
                rewards[agent_index] = trade_amount
                rewards[self.bank_agent_index] += -trade_amount  # The bank gains the opposite of the trader

                self.update_balances(agent_index, trade_amount)

                if trade_closed:
                    self.open_trades[agent_index] = None

        assert len(rewards) == self.num_agents

        self.n_iterations += 1
        bank_balance = self.agent_balances[self.bank_agent_index]
        done = self.n_iterations >= self.max_iterations or round(bank_balance, 4) <= 0.0 or \
            round(bank_balance, 4) == self.market_value

        assert round(sum(self.agent_balances), 4) == self.market_value

        # Return the state (in vector form), agent rewards, and whether the episode is done
        return self.vectorize(), rewards, done

    def curr_price(self) -> float:
        return self.prices[-1]

    def vectorize(self) -> np.array:
        def smma(closes, length):
            smma = []

            for i in range(len(closes)):
                if i < length:
                    smma.append(closes.iloc[:i + 1, ].rolling(length).mean().iloc[-1,])

                else:
                    smma.append((smma[i - 1] * (length - 1) + closes[i]) / length)

            return pd.Series(smma)

        def rsi(closes, periods=14):
            close_delta = closes.diff()

            up = close_delta.clip(lower=0)
            down = -1 * close_delta.clip(upper=0)
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

            rsi = ma_up / ma_down
            rsi = 100 - (100 / (1 + rsi))

            return rsi

        def stoch_rsi(rsi, k_window=3, d_window=3, window=14):
            min_val = rsi.rolling(window=window, center=False).min()
            max_val = rsi.rolling(window=window, center=False).max()

            stoch = ((rsi - min_val) / (max_val - min_val)) * 100

            slow_k = stoch.rolling(window=k_window, center=False).mean()

            slow_d = slow_k.rolling(window=d_window, center=False).mean()

            return slow_k, slow_d

        def qqe_mod(closes, rsi_period=6, smoothing=5, qqe_factor=3, threshold=3, mult=0.35, sma_length=50):
            Rsi = rsi(closes, rsi_period)
            RsiMa = Rsi.ewm(span=smoothing).mean()
            AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
            Wilders_Period = rsi_period * 2 - 1
            MaAtrRsi = AtrRsi.ewm(span=Wilders_Period).mean()
            dar = MaAtrRsi.ewm(span=Wilders_Period).mean() * qqe_factor

            longband = pd.Series(0.0, index=Rsi.index)
            shortband = pd.Series(0.0, index=Rsi.index)
            trend = pd.Series(0, index=Rsi.index)

            DeltaFastAtrRsi = dar
            RSIndex = RsiMa
            newshortband = RSIndex + DeltaFastAtrRsi
            newlongband = RSIndex - DeltaFastAtrRsi
            longband = pd.Series(np.where((RSIndex.shift(1) > longband.shift(1)) & (RSIndex > longband.shift(1)),
                                          np.maximum(longband.shift(1), newlongband), newlongband))
            shortband = pd.Series(np.where((RSIndex.shift(1) < shortband.shift(1)) & (RSIndex < shortband.shift(1)),
                                           np.minimum(shortband.shift(1), newshortband), newshortband))
            cross_1 = (longband.shift(1) < RSIndex) & (longband > RSIndex)
            cross_2 = (RSIndex > shortband.shift(1)) & (RSIndex.shift(1) < shortband)
            trend = np.where(cross_2, 1, np.where(cross_1, -1, trend.shift(1).fillna(1)))
            FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband))

            basis = (FastAtrRsiTL - 50).rolling(sma_length).mean()
            dev = mult * (FastAtrRsiTL - 50).rolling(sma_length).std()
            upper = basis + dev
            lower = basis - dev

            Greenbar1 = RsiMa - 50 > threshold
            Greenbar2 = RsiMa - 50 > upper

            Redbar1 = RsiMa - 50 < 0 - threshold
            Redbar2 = RsiMa - 50 < lower

            Greenbar = Greenbar1 & Greenbar2
            Redbar = Redbar1 & Redbar2

            return Greenbar, Redbar, RsiMa - 50

        price_series = pd.Series(self.prices)

        ma_50 = price_series.rolling(50).mean()
        ma_25 = price_series.rolling(25).mean()
        ema_50 = pd.Series.ewm(price_series, span=50).mean()
        ema_25 = pd.Series.ewm(price_series, span=25).mean()
        smma_50 = smma(price_series, 50)
        smma_25 = smma(price_series, 25)
        curr_rsi = rsi(price_series)
        rsi_sma = curr_rsi.rolling(25).mean()
        slowk_rsi, slowd_rsi = stoch_rsi(curr_rsi)
        macd = pd.Series.ewm(price_series, span=12).mean() - pd.Series.ewm(price_series, span=26).mean()
        macdsignal = pd.Series.ewm(macd, span=9).mean()
        macdhist = macd - macdsignal
        qqe_up, qqe_down, qqe_val = qqe_mod(price_series)

        vector = [ma_50.iloc[-1, ], ma_25.iloc[-1, ], ema_50.iloc[-1, ], ema_25.iloc[-1, ], smma_50.iloc[-1, ],
                  smma_25.iloc[-1, ], curr_rsi.iloc[-1, ], rsi_sma.iloc[-1, ], slowk_rsi.iloc[-1, ],
                  slowd_rsi.iloc[-1, ], macd.iloc[-1, ], macdsignal.iloc[-1, ], macdhist.iloc[-1, ], qqe_up.iloc[-1, ],
                  qqe_down.iloc[-1, ], qqe_val.iloc[-1, ], self.curr_price()]

        final_matrix = []

        for i in range(self.num_agents):
            vector_with_agent_i_balance = vector + [self.agent_balances[i]]
            final_matrix.append(vector_with_agent_i_balance)

        return np.array(final_matrix)
