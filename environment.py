from dataclasses import dataclass
from cryptocurrency import Crypto
import numpy as np

""" Balance to start with, in USD """
STARTING_BALANCE = 100000.0


class Action:
    buy = 0
    sell = 1
    hold = 2


States = np.zeros([101, 101, 101])


@dataclass(eq=True, frozen=True)
class State:
    # Price in USD
    market_price: float
    # Price percentage delta in USD
    market_delta: int  # -1000 - 1000
    hourly_delta: int  # 0 - 1000
    crypto_balance: float
    cash_balance: float
    daily_sentiment_change: int  # 0 - 100

    @property
    def total_balance(self):
        return (self.crypto_balance * self.market_price) + self.cash_balance

    @property
    def overall_profit(self) -> int:
        return (
            round((self.total_balance - STARTING_BALANCE) / STARTING_BALANCE, 2) * 100
        )

    @property
    def market_percent_change(self) -> int:
        return round(self.market_delta)

    @property
    def market_hourly_change(self) -> int:
        return round(self.hourly_delta)

    @property
    def is_terminal(self) -> bool:
        """
        A terminal state is reached when the total balance is 10x less, or 10x the original.
        """
        is_bankrupt = self.total_balance <= STARTING_BALANCE / 10.0
        is_rich = self.total_balance >= STARTING_BALANCE * 10.0  # 1 Million USD

        return is_bankrupt or is_rich

    @property
    def was_profitable(self) -> bool:
        return self.total_balance >= STARTING_BALANCE * 10.0

    def __hash__(self):
        return hash(
            (
                self.overall_profit,
                self.market_delta,
                self.market_hourly_change,
                self.daily_sentiment_change,
            )
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return f"Overall Profit: {self.overall_profit}"


class Environment:

    fetch_data = Crypto.get_current_data

    @classmethod
    def with_live_fetcher(cls):
        return cls(Crypto.get_current_data)

    @classmethod
    def with_csv_fetcher(cls):
        return cls(Crypto.get_archived_data().__next__)

    def default_state(self):
        (
            new_price_usd,
            new_price_delta,
            new_hourly_usd_change,
            daily_sentiment_change,
        ) = self.fetch_data()

        first_state = State(
            market_price=new_price_usd,
            market_delta=new_price_delta,
            hourly_delta=new_hourly_usd_change,
            crypto_balance=0.0,
            cash_balance=STARTING_BALANCE,
            daily_sentiment_change=daily_sentiment_change,
        )
        return first_state

    def __init__(self, data_fetch_fn):
        self.fetch_data = data_fetch_fn

    def step(self, state: State, action: Action) -> tuple[State, float]:
        new_crypto_balance = state.crypto_balance
        new_cash_balance = state.cash_balance
        (
            new_price_usd,
            new_price_delta,
            new_hourly_usd_change,
            daily_sentiment_change,
        ) = self.fetch_data()

        reward = 0.0

        # Perform action
        if action == Action.buy:
            purchase_amount_usd = 0.5 * state.cash_balance

            crypto_purchased = Crypto.purchase(new_price_usd, purchase_amount_usd)

            # Finalize the transaction
            new_cash_balance -= purchase_amount_usd
            new_crypto_balance += crypto_purchased

        elif action == Action.sell:
            sale_amount_crypto = 0.5 * state.crypto_balance
            sale_return_usd = Crypto.sell(new_price_usd, sale_amount_crypto)

            new_cash_balance += sale_return_usd
            new_crypto_balance -= sale_amount_crypto

        new_state = State(
            new_price_usd,
            new_price_delta,
            new_hourly_usd_change,
            new_crypto_balance,
            new_cash_balance,
            daily_sentiment_change,
        )

        reward = 0

        if new_state.is_terminal:
            # Diagnose the issue
            if new_state.was_profitable:
                reward = 1
            else:
                reward = -1

        return new_state, reward

