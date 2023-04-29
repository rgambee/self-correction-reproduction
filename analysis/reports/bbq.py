#!/usr/bin/env python3
from analysis.graders.bbq import is_answer_correct
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.bbq import calculate_bias_ambiguous, calculate_bias_disambiguated
from analysis.reports import load_results, parse_args
from loaders.bbq import BBQParameters


def main() -> None:
    """Report metrics for results from the BBQ dataset"""
    user_args = parse_args()
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

        print("Results for file", path.name)
        print(f"{accuracy!r} accuracy overall")
        print(f"{bias_disambig!r} bias score in disambiguated contexts")
        print(f"{bias_ambig!r} bias score in ambiguous contexts")


if __name__ == "__main__":
    main()
