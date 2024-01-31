from environment.trade import Trade, TradeType
from typing import Tuple


class MarketCalculations(object):
    @staticmethod
    def calculate_trade_amount(trade: Trade, curr_price: float, account_balance: float,
                               max_periods_open: int, day_fee_percentage: float = 0.01) -> Tuple[float, bool]:
        trade_type, open_price, stop_gain, stop_loss, percent_to_risk = \
            trade.trade_type, trade.open_price, trade.stop_gain, trade.stop_loss, trade.percent_to_risk
        dollar_amount = account_balance * percent_to_risk
        day_fees = dollar_amount * day_fee_percentage * trade.time_active

        # Trade is a buy and loses
        if trade_type is TradeType.BUY and curr_price <= stop_loss:
            trade_amount = -dollar_amount

            return trade_amount - day_fees, True

        # Trade is a buy and wins
        elif trade_type is TradeType.BUY and curr_price >= stop_gain:
            pips_risked = abs(open_price - stop_loss)
            pips_gained = abs(stop_gain - open_price)
            risk_reward_ratio = pips_gained / pips_risked
            trade_amount = risk_reward_ratio * dollar_amount

            return trade_amount - day_fees, True

        # Trade is a buy and has been open for too long
        elif trade_type is TradeType.BUY and trade.time_active >= max_periods_open:
            pips_risked = abs(open_price - stop_loss)
            pips_gained = curr_price - trade.open_price
            risk_reward_ratio = pips_gained / pips_risked
            trade_amount = risk_reward_ratio * dollar_amount

            return trade_amount - day_fees, True

        # Trade is a sell and loses
        elif trade_type is TradeType.SELL and curr_price >= stop_loss:
            trade_amount = -dollar_amount

            return trade_amount - day_fees, True

        # Trade is a sell and wins
        elif trade_type is TradeType.SELL and curr_price <= stop_gain:
            pips_risked = abs(stop_loss - open_price)
            pips_gained = abs(open_price - stop_gain)
            risk_reward_ratio = pips_gained / pips_risked
            trade_amount = risk_reward_ratio * dollar_amount

            return trade_amount - day_fees, True

        # Trade is a sell and has been open for too long
        elif trade_type is TradeType.SELL and trade.time_active >= max_periods_open:
            pips_risked = abs(stop_loss - open_price)
            pips_gained = trade.open_price - curr_price
            risk_reward_ratio = pips_gained / pips_risked
            trade_amount = risk_reward_ratio * dollar_amount

            return trade_amount - day_fees, True

        # Otherwise, the trade hasn't closed out yet
        return 0.0, False
