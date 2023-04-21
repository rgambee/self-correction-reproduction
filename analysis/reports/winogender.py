#!/usr/bin/env python3
from typing import Dict

import matplotlib.pyplot as plt

from analysis.graders.winogender import is_answer_correct
from analysis.metrics import ValueWithConfidence, calculate_accuracy
from loaders.winogender import WinogenderParameters

from . import load_results, parse_args


def plot_accuracies(accuracies: Dict[str, ValueWithConfidence]) -> None:
    """Plot the accuracy for each results file using a bar chart"""
    _, axis = plt.subplots()
    axis.bar(range(len(accuracies)), [acc.value * 100.0 for acc in accuracies.values()])
    axis.set_xticks(
        range(len(accuracies)),
        labels=accuracies.keys(),
    )
    axis.set_xlabel("Results file")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Accuracy for Winogender Dataset")
    plt.show()


def main() -> None:
    """Report metrics for results from the Winogender dataset"""
    user_args = parse_args()
    accuracies: Dict[str, ValueWithConfidence] = {}
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
