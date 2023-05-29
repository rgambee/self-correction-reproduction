#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from analysis.metrics.binomial import BinomialDistribution, error_bars
from analysis.metrics.winogender import (
    calculate_accuracy_for_pronoun,
    calculate_correlation,
)
from analysis.reports import load_results, parse_args
from eval.result import Result
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
    logger = logging.getLogger(__name__)
    user_args = parse_args()
    for path in user_args.result_paths:
        results = load_results(path, WinogenderParameters)
        grouped_by_id: Dict[int, List[Result[WinogenderParameters]]] = {}
        for res in results:
            grouped_by_id.setdefault(res.sample.id, []).append(res)

        proportion_female_model: List[BinomialDistribution] = []
        proportion_female_bls: List[float] = []
        for sample_id, results_for_id in grouped_by_id.items():
            try:
                proportion_female_model.append(
                    calculate_accuracy_for_pronoun(
                        results=results_for_id,
                        pronoun_index=1,  # 0: neutral, 1: female, 2: male
                        confidence_level=user_args.confidence_level,
                    )
                )
            except ValueError:
                logger.warning(
                    "Cannot determine pronoun proportion for sample id %d", sample_id
                )
            else:
                proportion_female_bls.append(
                    results_for_id[0].sample.parameters.proportion_female
                )
        correlation_coeff = calculate_correlation(
            proportion_female_model, proportion_female_bls, user_args.confidence_level
        )

        print("Results for file", path.name)
        print(f"{correlation_coeff!r} Pearson correlation coefficient")


if __name__ == "__main__":
    main()
