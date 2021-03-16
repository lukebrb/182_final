from collections import defaultdict
import requests
import csv
from time import sleep
from datetime import datetime


class Crypto:
    base_url = "https://api.coingecko.com/api/v3/coins/ethereum"
    dataset_path = "./datasets/ETHUSD_Bitfinex_M_historical.csv"
    sentiment_path = "./datasets/twitter_sentiment.csv"

    @staticmethod
    def get_current_data() -> tuple[float, float]:

        sleep(5)  # So we don't get de-prioritized for api calls

        response = requests.get(
            url=Crypto.base_url,
            params={"localization": "false", "developer_data": "false"},
        )

        hourly_usd_change = response.json()["market_data"][
            "price_change_percentage_1h_in_currency"
        ]["usd"]

        price_usd = response.json()["market_data"]["current_price"]["usd"]

        return (price_usd, hourly_usd_change)

    @staticmethod
    def purchase(at_price: float, amount_usd: float) -> float:
        return amount_usd / at_price  # This is in ETH

    @staticmethod
    def sell(at_price: float, amount_crypto: float) -> float:
        return amount_crypto * at_price

    @staticmethod
    def get_archived_data() -> tuple[float, float]:
        """
        Uses a generator function to yield "new" results from the file. 
        """

        # Read in the sentiment data
        daily_sentiment_percent_change = defaultdict(int)
        with open(Crypto.sentiment_path, "r") as sentimentfile:
            reader = csv.DictReader(sentimentfile)
            yesterdays_sentiment = 0
            for daily_sentiment in reader:
                day = daily_sentiment["date"]
                sentiment = float(daily_sentiment["average_sentiment"])

                change = round((sentiment - yesterdays_sentiment) * 100)
                daily_sentiment_percent_change[day] = change

                yesterdays_sentiment = sentiment

        # Read in from the file
        with open(Crypto.dataset_path, "r") as datafile:
            reader = csv.DictReader(datafile)
            prior_price = 0

            all_prices = []

            hourly_change = (
                lambda: round(
                    (all_prices[-60:][-1] - all_prices[-60:][0])
                    / all_prices[-60:][0]
                    * 1000
                )
                if len(all_prices) > 0
                else 0
            )

            daily_change = (
                lambda: round(
                    (all_prices[-(60 * 24) :][-1] - all_prices[-(60 * 24) :][0])
                    / all_prices[-(60 * 24) :][0]
                    * 1000
                )
                if len(all_prices) > 0
                else 0
            )

            for minute in reader:
                """
                A row will be a dictionary with keys 
                TimeStamp,Pair,Open,High,Low,Close,Volume_USD,Volume_CCY

                Since we're trying to match `Crypto.get_current_data()`'s output,
                we must perform some calculations to get the same state.
                """

                new_price = float(minute["Open"])

                if prior_price == 0:
                    price_change_percentage = 0
                else:
                    price_change_percentage = round(
                        (new_price - prior_price) / prior_price * 1000
                    )

                minute_date = datetime.fromtimestamp(
                    float(minute["TimeStamp"])
                ).isoformat()

                all_prices.append(new_price)

                yield (
                    new_price,
                    int(daily_change()),
                    int(hourly_change()),
                    int(daily_sentiment_percent_change[minute_date]),
                )

                prior_price = new_price

