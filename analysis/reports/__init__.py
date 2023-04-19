import logging
from pathlib import Path
from typing import Iterator, List, Type

import jsonlines

from eval.result import Completion, Reply, Result
from loaders import P, Sample
from prompts.message import Message


# This function uses PascalCase for one of its parameters since it refers to a class,
# rather than a particular instance.
# pylint: disable-next=invalid-name
def load_results(path: Path, ParameterClass: Type[P]) -> Iterator[Result[P]]:
    """Load results from a JSONL file

    ParameterClass should be the dataclass that corresponds to the samples of this
    dataset, e.g. BBQParameters if these results were generated with the BBQ dataset.
    """
    logger = logging.getLogger(__name__)
    with jsonlines.open(path) as reader:
        for entry in reader:
            try:
                entry["sample"]["parameters"] = ParameterClass(
                    **entry["sample"]["parameters"]
                )
                sample = Sample[P](**entry["sample"])
                prompt_messages = [
                    Message(**message) for message in entry["prompt_messages"]
                ]
                completions: List[Completion] = []
                for compl in entry["reply"]["choices"]:
                    compl["message"] = Message(**compl["message"])
                    completions.append(Completion(**compl))
                entry["reply"]["choices"] = completions
                reply = Reply(**entry["reply"])
                yield Result(
                    sample=sample, prompt_messages=prompt_messages, reply=reply
                )
            except KeyError as err:
                logger.debug("Skipping entry due to error: %r", err)
