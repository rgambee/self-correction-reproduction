#!/usr/bin/env python3
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from analysis.graders.winogender import is_answer_correct
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.binomial import BinomialDistribution, error_bars
from analysis.reports import load_results, parse_args
from loaders.winogender import WinogenderParameters


def plot_accuracies(
    accuracies: Dict[str, BinomialDistribution],
    axes: Optional[Axes] = None,
) -> Axes:
    """Plot the accuracy for each results file using a bar chart"""
    if axes is None:
        _, axes = plt.subplots()
    yerr = error_bars(list(accuracies.values()))
    axes.bar(
        x=range(len(accuracies)),
        height=[acc.proportion * 100.0 for acc in accuracies.values()],
        yerr=yerr,
    )
    axes.set_xticks(
        range(len(accuracies)),
        labels=accuracies.keys(),
    )
    axes.set_ylim(0, 100)
    axes.set_xlabel("Results file")
    axes.set_ylabel("Accuracy (%)")
    axes.set_title("Accuracy for Winogender Dataset")
    return axes


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
        print("Results for file", path.name)
        print(f"{accuracy!r} overall accuracy")
        accuracies[path.name] = accuracy

    if user_args.plot:
        plot_accuracies(accuracies)
        plt.show()


if __name__ == "__main__":
    main()
