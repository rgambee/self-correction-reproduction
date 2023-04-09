from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, Iterator

import jsonlines

from loaders import DatasetLoader, Question


@unique
class BBQPolarity(Enum):
    POSITIVE = "pos"
    NEGATIVE = "neg"


@unique
class BBQContextCondition(Enum):
    AMBIGUOUS = "ambig"
    DISAMBIGUOUS = "disambig"


@dataclass
class BBQParameters:
    """Parameters for a single question from the BBQ dataset

    The fields are as follows:
    * context: A sentence describing the context of the question
    * context_condition: "ambig" or "disambig", depending on the detail of the context
    * polarity: "pos" or "neg", depending on the stereotype the context describes
    """

    context: str
    context_condition: BBQContextCondition
    polarity: BBQPolarity

    def __post_init__(self) -> None:
        self.context_condition = BBQContextCondition(self.context_condition)
        self.polarity = BBQPolarity(self.polarity)


class BBQQuestion(Question[BBQParameters]):  # pylint: disable=too-few-public-methods
    pass


class BBQLoader(DatasetLoader[BBQParameters]):  # pylint: disable=too-few-public-methods
    """Loader for the Bias Benchmark for QA (BBQ) dataset

    The BBQ dataset is saved as a series of JSONL files, one for each category.
    """

    dataset = "bbq"

    def _entry_to_question(self, entry: Dict[str, Any]) -> BBQQuestion:
        """Transform a line from the BBQ dataset into a Question"""
        parameters = BBQParameters(
            context=entry["context"],
            context_condition=entry["context_condition"],
            polarity=entry["question_polarity"],
        )
        return BBQQuestion(
            dataset=self.dataset,
            category=entry["category"].lower(),
            id=entry["example_id"],
            parameters=parameters,
            answers=[
                entry["ans0"],
                entry["ans1"],
                entry["ans2"],
            ],
            correct_answer=entry["label"],
        )

    def _iter_entries(self, path: Path) -> Iterator[BBQQuestion]:
        """Loop over the lines of a JSONL file and yield each as a question"""
        with jsonlines.open(path) as reader:
            for entry in reader:
                yield self._entry_to_question(entry)
