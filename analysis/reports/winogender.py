#!/usr/bin/env python3
from typing import List, Tuple

import matplotlib.pyplot as plt

from analysis.graders.winogender import is_answer_correct
from analysis.metrics import calculate_accuracy, calculate_accuracy_ci
from loaders.winogender import WinogenderParameters

from . import load_results, parse_args


def main() -> None:
    """Report metrics for results from the Winogender dataset"""
    user_args = parse_args()
    accuracies: List[Tuple[str, float]] = []
    for path in user_args.result_paths:
        assessments = tuple(
            is_answer_correct(result)
            for result in load_results(path, WinogenderParameters)
        )
        accuracy = calculate_accuracy(assessments)
        acc_low, acc_high = calculate_accuracy_ci(
            assessments, confidence=user_args.confidence_level
        )
        print(
            f"{accuracy:6.1%} accuracy",
            f"({user_args.confidence_level:3.0%} CI: {acc_low:6.1%} - {acc_high:6.1%})",
            f"for results {path.name}",
        )
        accuracies.append((path.name, accuracy * 100.0))

    if user_args.plot:
        _, axis = plt.subplots()
        axis.bar(range(len(accuracies)), [accuracy for _, accuracy in accuracies])
        axis.set_xticks(
            range(len(accuracies)),
            labels=[name for name, _ in accuracies],
        )
        axis.set_xlabel("Results file")
        axis.set_ylabel("Accuracy (%)")
        axis.set_title("Accuracy for Winogender Dataset")
        plt.show()


if __name__ == "__main__":
    main()
