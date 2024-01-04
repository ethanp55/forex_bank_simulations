from dataclasses import dataclass
import enum


class TradeType(enum.Enum):
    BUY = 1
    SELL = 2


@dataclass
class Trade:
    trade_type: TradeType
    open_price: float
    stop_loss: float
    stop_gain: float
    percent_to_risk: float
