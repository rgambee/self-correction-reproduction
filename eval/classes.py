from collections import abc
from dataclasses import asdict, dataclass
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

import openai

from loaders import P, Sample
from prompts import Message, Messages


@dataclass
class Completion:
    """A single completion from the OpenAI /chat/completions endpoint"""

    message: Message
    index: int
    finish_reason: str

    def __post_init__(self) -> None:
        self.message = dataclass_from_mapping_or_iterable(Message, self.message)


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
            dataclass_from_mapping_or_iterable(Completion, chc) for chc in self.choices
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
    """A request to the OpenAI API for a single sample

    This uses the /chat/completions endpoint, so the specified model must support chat
    completions. For instance, gpt-3.5-turbo or gpt-4.
    """

    parameters: RequestParameters
    messages: Messages
    sample: Sample[P]

    async def submit(self) -> Reply:
        """Submit this request to the OpenAI API and return the reply"""
        resp = await openai.ChatCompletion.acreate(  # type: ignore[no-untyped-call]
            messages=[asdict(msg) for msg in self.messages],
            **asdict(self.parameters),
        )
        return Reply(**resp)


@dataclass
class Result(Generic[P]):
    """A combined sample and reply"""

    sample: Sample[P]
    prompt_messages: Messages
    reply: Reply


T = TypeVar("T")


def dataclass_from_mapping_or_iterable(cls: Type[T], value: Any) -> T:
    """Convert a mapping or iterable to an instance of a dataclass"""
    if isinstance(value, cls):
        return value
    if isinstance(value, abc.Mapping):
        return cls(**value)
    try:
        iter(value)
    except TypeError as err:
        raise TypeError(f"Can't convert {value} to {cls}") from err
    return cls(*value)
