from concurrent.futures import process
from multiprocessing.context import ProcessError
import twitter
import csv
from datetime import date, timedelta
import spacy
from spacy.util import minibatch, compounding
import numpy as np
import os
import random

from tqdm import trange, tqdm
from tqdm.contrib.concurrent import process_map
import twint

import multiprocessing


def load_training_data(
    data_dir: str = "aclImdb/train", split: float = 0.8, limit: int = 0
) -> tuple:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_dir = f"{data_dir}/{label}"
        for review in os.listdir(labeled_dir):
            if review.endswith(".txt"):
                with open(f"{labeled_dir}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {"pos": "pos" == label, "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if predicted_label == "neg":
                continue
            if score >= 0.5 and true_label["cats"]["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["cats"]["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["cats"]["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["cats"]["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def train_model(training_data: list, test_data: list, iterations: int = 20) -> None:
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    training_excluded_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]

    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        print("Training...")
        batch_sizes = compounding(4.0, 32.0, 1.001)

        for i in trange(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)

            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer, textcat=textcat, test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


def analyze_sentiment(of_string: str) -> float:
    nlp = spacy.load("model_artifacts")
    parsed_text = nlp(of_string)
    return parsed_text.cats["pos"] - parsed_text.cats["neg"]


def daterange(start, end) -> tuple[date, date]:
    delta = timedelta(days=1.0)
    current = start
    while current != end:
        tomorrow = current + delta
        yield (current, tomorrow)
        current = tomorrow


if __name__ == "__main__":

    if "model_artifacts" not in os.listdir():
        print("Model artifacts not found; training from scratch...")
        train, test = load_training_data(limit=2500)
        train_model(train, test)

    with open("datasets/twitter_sentiment.csv", "w") as sentiment_file:
        beginning = date.fromisoformat("2020-11-15")
        end = date.fromisoformat("2021-02-18")

        current_date = beginning

        csv_writer = csv.DictWriter(sentiment_file, ["date", "average_sentiment"])

        csv_writer.writeheader()

        for (current_date, tomorrow) in tqdm(
            daterange(beginning, end), desc=f"Date Progress:"
        ):
            tweets = []

            with open(f"tweets/{tomorrow.isoformat()}/tweets.csv") as day_tweets:
                csv_reader = csv.DictReader(day_tweets)
                for tweet in csv_reader:
                    tweets.append(tweet["tweet"])

            sentiments = process_map(analyze_sentiment, tweets)
            avg_daily_sentiment = np.average(sentiments)

            csv_writer.writerow(
                {
                    "date": current_date.isoformat(),
                    "average_sentiment": str(avg_daily_sentiment),
                }
            )

