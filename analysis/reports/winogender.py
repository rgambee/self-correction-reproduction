#!/usr/bin/env python3
from typing import Dict

import matplotlib.pyplot as plt

from analysis.graders.winogender import is_answer_correct
from analysis.metrics import BinomialDistribution, calculate_accuracy, error_bars
from loaders.winogender import WinogenderParameters

from . import load_results, parse_args


def plot_accuracies(accuracies: Dict[str, BinomialDistribution]) -> None:
    """Plot the accuracy for each results file using a bar chart"""
    _, axis = plt.subplots()
    yerr = error_bars(list(accuracies.values()))
    axis.bar(
        x=range(len(accuracies)),
        height=[acc.proportion * 100.0 for acc in accuracies.values()],
        yerr=yerr,
    )
    axis.set_xticks(
        range(len(accuracies)),
        labels=accuracies.keys(),
    )
    axis.set_ylim(0, 100)
    axis.set_xlabel("Results file")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Accuracy for Winogender Dataset")
    plt.show()


def main() -> None:
    """Report metrics for results from the Winogender dataset"""
    user_args = parse_args()
    accuracies: Dict[str, BinomialDistribution] = {}
    for path in user_args.result_paths:
        assessments = tuple(
            is_answer_correct(result)
            for result in load_results(path, WinogenderParameters)
        )
        accuracy = calculate_accuracy(
            assessments,
            confidence_level=user_args.confidence_level,
        )
        print(f"{accuracy!r} accuracy for results {path.name}")
        accuracies[path.name] = accuracy

    if user_args.plot:
        plot_accuracies(accuracies)


if __name__ == "__main__":
    main()
