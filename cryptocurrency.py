import requests
import csv
from time import sleep


class Crypto:
    base_url = "https://api.coingecko.com/api/v3/coins/ethereum"
    dataset_path = "./datasets/ETHUSD_Bitfinex_M_historical.csv"

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
        # Read in from the file
        with open(Crypto.dataset_path, "r") as datafile:
            reader = csv.DictReader(datafile)
            prior_price = 0
            for row in reader:
                """
                A row will be a dictionary with keys 
                TimeStamp,Pair,Open,High,Low,Close,Volume_USD,Volume_CCY

                Since we're trying to match `Crypto.get_current_data()`'s output,
                we must perform some calculations to get the same state.
                """

                new_price = float(row["Open"])

                if prior_price == 0:
                    price_change_percentage = 0
                else:
                    price_change_percentage = (new_price - prior_price) / prior_price

                yield (new_price, price_change_percentage)

                prior_price = new_price
