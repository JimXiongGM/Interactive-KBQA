""" Official evaluation script for v1.0 of the ComplexWebQuestions dataset. """

import argparse
import json
import re
import unicodedata
from typing import List

import pandas as pd


def compare_span_to_answer(spans, golden_answers, question):
    """
    Compares one answers to spans, multiple matches are possible
    spans: predicted answer spans
    """
    if len(spans) == 0:
        return []

    found_answers = pd.DataFrame(columns=["span", "answer", "span_index"])
    spans_series = pd.Series(spans)
    pre_proc_golden_answers = []
    golden_answers = [answer.lower().strip() for answer in golden_answers]
    for golden_answer in golden_answers:
        proc_answer = (
            unicodedata.normalize("NFKD", golden_answer).encode("ascii", "ignore").decode(encoding="UTF-8")
        )

        # removing common endings such as "f.c."
        proc_answer = re.sub(r"\W", " ", proc_answer).lower().strip()
        # removing The, a, an from begining of answer as proposed by SQuAD dataset answer comparison
        if proc_answer.startswith("the "):
            proc_answer = proc_answer[4:]
        if proc_answer.startswith("a "):
            proc_answer = proc_answer[2:]
        if proc_answer.startswith("an "):
            proc_answer = proc_answer[3:]

        pre_proc_golden_answers.append(proc_answer)

    question = question.lower().strip()

    # exact match:
    for pre_proc_golden_answer, golden_answer in zip(pre_proc_golden_answers, golden_answers):
        if golden_answer in spans:
            exact_match_ind = spans.index(golden_answer)
            found_answers = found_answers.append(
                {"span_index": exact_match_ind, "answer": golden_answer, "span": golden_answer},
                ignore_index=True,
            )

        if pre_proc_golden_answer in spans:
            exact_match_ind = spans.index(pre_proc_golden_answer)
            found_answers = found_answers.append(
                {
                    "span_index": exact_match_ind,
                    "answer": golden_answer,
                    "span": pre_proc_golden_answer,
                },
                ignore_index=True,
            )

        # year should match year.
        if question.find("year") > -1:
            year_in_answer = re.search("([1-2][0-9]{3})", golden_answer)
            if year_in_answer is not None:
                year_in_answer = year_in_answer.group(0)

            year_spans = spans_series[spans_series == year_in_answer]
            if len(year_spans) > 0:
                found_answers = found_answers.append(
                    {
                        "span_index": year_spans.index[0],
                        "answer": golden_answer,
                        "span": year_in_answer,
                    },
                    ignore_index=True,
                )

    return found_answers.drop_duplicates()


def compute_P1(matched_answers):
    P1 = 0
    if len(matched_answers) > 0:
        P1 = 100

    return P1


def evaluate(dataset_df: List[dict], predictions: List[dict]):
    # please predict the full file
    if len(dataset_df) != len(predictions):
        print("predictions file does not match dataset file number of examples!!!")

    P1 = 0
    for prediction in predictions:
        golden_answer_list = []
        for answer in dataset_df.loc[prediction["id"], "answers_raw"]:
            golden_answer_list.append(answer["answer"])
            golden_answer_list += answer["aliases"]

        if not None in golden_answer_list:
            matched_answers = compare_span_to_answer(
                [prediction["answers"]],
                golden_answer_list,
                dataset_df.loc[prediction["id"], "question"],
            )
            curr_P1 = compute_P1(matched_answers)

            P1 += curr_P1

    return P1 / len(dataset_df)


if __name__ == "__main__":
    # expected_version = '1.0'
    parser = argparse.ArgumentParser(description="Evaluation for ComplexWebQuestions ")
    parser.add_argument("--dataset_file", help="Dataset file", default="")
    parser.add_argument("--prediction_file", help="Prediction File", default="")
    args = parser.parse_args()

    args.dataset_file = "dataset_processed/cwq/test/comparative.json"
    args.prediction_file = "dataset_processed/cwq/test/comparative.json"

    with open(args.dataset_file) as dataset_file:
        dataset_df = pd.DataFrame(json.load(dataset_file)).set_index("id")
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset_df, predictions)))
