from datetime import timedelta, date
import twint
from tqdm import tqdm

beginning = date.fromisoformat("2020-11-15")
end = date.fromisoformat("2021-02-18")
current_date = beginning

delta = timedelta(days=1.0)


def daterange(start, end) -> tuple[date, date]:
    current = start
    while current != end:
        tomorrow = current + delta
        yield (current, tomorrow)
        current = tomorrow


dates = tqdm(daterange(beginning, end))
for (current, tomorrow) in dates:

    dates.set_description(f"Getting tweets for {current.isoformat()}...")

    c = twint.Config()
    c.Search = "ethereum"
    c.Since = current_date.isoformat()
    c.Until = tomorrow.isoformat()
    c.Limit = 100
    c.Popular_tweets = True
    c.Store_csv = True
    c.Output = f"tweets/{current.isoformat()}.csv"

    twint.run.Search(c)

