import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import jsonlines
import openai

from loaders import P, Sample


@dataclass
class Completion:
    """A single completion from the OpenAI API"""

    text: str
    index: int
    finish_reason: str
    logprobs: Optional[Dict[str, float]] = None


@dataclass
class Reply:
    """A reply from the OpenAI API, possibly containing several completions"""

    # pylint doesn't like two letter names, claiming they don't conform to the
    # snake_case convention
    id: str  # pylint: disable=invalid-name
    object: str
    created: int
    model: str
    choices: Sequence[Completion]

    def __post_init__(self) -> None:
        self.choices = [
            Completion(**choice) for choice in self.choices  # type: ignore[arg-type]
        ]


@dataclass
class RequestParameters:
    model: str
    max_tokens: int
    temperature: float
    logprobs: int


@dataclass
class Request:
    """A request to the OpenAI API"""

    parameters: RequestParameters
    prompt: str

    def submit(self) -> Reply:
        """Submit this request to the OpenAI API and return the reply"""
        resp = openai.Completion.create(  # type: ignore[no-untyped-call]
            prompt=self.prompt, **asdict(self.parameters)
        )
        return Reply(**resp)


def evaluate_dataset(
    samples: Iterable[Sample[P]],
    prompt_func: Callable[[Sample[P]], str],
    results_file: Path,
    parameters: RequestParameters,
) -> None:
    """Evaluate each sample of a dataset using the OpenAI API

    Results will be appended to the given path.

    This function will skip any samples that have already been evaluated by examining
    the results file.
    """
    # Check the results file to see if we've already evaluated some of the samples
    last_sample = find_most_recent_sample(results_file)

    with jsonlines.open(
        results_file,
        mode="a",
        dumps=partial(json.dumps, default=to_json_serializable_type),
    ) as output:
        for sample in samples:
            # If we've already evaluated this sample, skip it
            if last_sample is not None and sample.id <= last_sample:
                continue

            prompt = prompt_func(sample)
            request = Request(parameters=parameters, prompt=prompt)
            reply = request.submit()
            output.write({"sample": sample, "reply": reply})


def find_most_recent_sample(results_file: Path) -> Optional[int]:
    """Check the results file to get the id of the sample that was last evaluated

    This is helpful for resuming a long evaluation after it's been interrupted.

    If the results file doesn't exist or doesn't contain any valid results, return None.

    This function assumes the results are sorted by sample id.
    """
    if not results_file.exists():
        return None

    last_id: Optional[int] = None
    with jsonlines.open(results_file) as reader:
        for line in reader.iter(type=dict, skip_invalid=True):
            try:
                last_id = int(line.get("sample", {}).get("id", None))
            except (AttributeError, TypeError, ValueError):
                pass
    return last_id


def to_json_serializable_type(value: Any) -> Any:
    """Convert a value to a JSON-serializable type"""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
