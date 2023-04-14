from dataclasses import asdict, dataclass
from typing import Dict, Generic, Optional, Sequence

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
    """A reply from the OpenAI API for a single sample

    The reply could contain several completions.
    """

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
    timeout: Optional[float] = None


@dataclass
class Request(Generic[P]):
    """A request to the OpenAI API for a single sample"""

    parameters: RequestParameters
    prompt: str
    sample: Sample[P]

    async def submit(self) -> Reply:
        """Submit this request to the OpenAI API and return the reply"""
        resp = await openai.Completion.acreate(  # type: ignore[no-untyped-call]
            prompt=self.prompt, **asdict(self.parameters)
        )
        return Reply(**resp)


@dataclass
class Result(Generic[P]):
    """A combined sample and reply"""

    sample: Sample[P]
    reply: Reply
