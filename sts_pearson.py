from scipy.stats import pearsonr
import difflib
from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
import argparse
from util import parse_sts
import warnings

warnings.filterwarnings('ignore')


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    nist_scores = []
    bleu_scores = []
    wer_scores = []
    lcs_scores = []
    ed_scores = []

    for text_pair in texts:
        # input tokenized text
        t1, t2 = text_pair
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())

        # calculate NIST for each text pair
        try:
            nist_score = sentence_nist([t1_toks, ], t2_toks)
        except ZeroDivisionError:
            nist_score = 0.0

        nist_scores.append(nist_score)

        # calculate BLEU for each text pair
        try:
            bleu_score = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=SmoothingFunction().method0)
        except ZeroDivisionError:
            bleu_score = 0.0

        bleu_scores.append(bleu_score)

        # calculate Word Error Rate for each text pair
        wer_score = edit_distance(t1_toks, t2_toks)/(len(t1_toks)+len(t2_toks))

        wer_scores.append(wer_score)

        # calculate Longest common substring for each text pair
        lcs_score = difflib.SequenceMatcher(None, t1, t2).ratio()

        lcs_scores.append(lcs_score)

        # calculate Edit Distance for each text pair
        ed_score = edit_distance(t1, t2)
        ed_scores.append(ed_score)

    # create a dictionary to save scores
    metrics_score = {'NIST': nist_scores,
                     "BLEU": bleu_scores,
                     "Word Error Rate": wer_scores,
                     "Longest common substring": lcs_scores,
                     "Edit Distance": ed_scores
                     }

    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        score = pearsonr(metrics_score[metric_name], labels)[0]
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)