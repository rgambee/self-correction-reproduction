#!/usr/bin/env python3
from typing import Dict, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from analysis.graders.bbq import is_answer_correct
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.bbq import calculate_bias_ambiguous, calculate_bias_disambiguated
from analysis.metrics.binomial import ValueCI, error_bars
from analysis.reports import load_results, parse_args
from loaders.bbq import BBQParameters


def plot_bias_scores(
    bias_scores: Dict[str, ValueCI],
    axes: Optional[Axes] = None,
) -> Axes:
    """Plot the BBQ bias score for each results file using a bar chart"""
    if axes is None:
        _, axes = plt.subplots()
    yerr = error_bars(list(bias_scores.values()))
    axes.bar(
        x=range(len(bias_scores)),
        height=[score.value for score in bias_scores.values()],
        yerr=yerr,
    )
    axes.set_xticks(
        range(len(bias_scores)),
        labels=bias_scores.keys(),
    )
    axes.set_ylim(-1.0, 1.0)
    axes.set_xlabel("Results file")
    axes.set_ylabel("Bias Score")
    axes.set_title("Bias Score for BBQ Dataset")
    return axes


def main() -> None:
    """Report metrics for results from the BBQ dataset"""
    user_args = parse_args()
    bias_scores_ambig: Dict[str, ValueCI] = {}
    for path in user_args.result_paths:
        results = list(load_results(path, BBQParameters))
        assessments = (is_answer_correct(res) for res in results)
        accuracy = calculate_accuracy(
            assessments, confidence_level=user_args.confidence_level
        )
        bias_disambig = calculate_bias_disambiguated(
            results, confidence_level=user_args.confidence_level
        )
        bias_ambig = calculate_bias_ambiguous(
            results,
            confidence_level=user_args.confidence_level,
            bias_disambig=bias_disambig,
        )
        bias_scores_ambig[str(path)] = bias_ambig

        print("Results for file", path.name)
        print(f"{accuracy!r} accuracy overall")
        print(f"{bias_disambig!r} bias score in disambiguated contexts")
        print(f"{bias_ambig!r} bias score in ambiguous contexts")

    if user_args.plot:
        plot_bias_scores(bias_scores_ambig)
        plt.show()


if __name__ == "__main__":
    main()
