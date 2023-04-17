from dataclasses import asdict, dataclass
from typing import Generic, Optional, Sequence

import openai

from loaders import P, Sample


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Completion:
    """A single completion from the OpenAI /chat/completions endpoint"""

    message: Message
    index: int
    finish_reason: str

    def __post_init__(self) -> None:
        # pylint: disable-next=not-a-mapping
        self.message = Message(**self.message)  # type: ignore[arg-type]


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
    """A request to the OpenAI API for a single sample

    This uses the /chat/completions endpoint, so the specified model must support chat
    completions. For instance, gpt-3.5-turbo or gpt-4.
    """

    parameters: RequestParameters
    prompt: str
    sample: Sample[P]

    async def submit(self) -> Reply:
        """Submit this request to the OpenAI API and return the reply"""
        resp = await openai.ChatCompletion.acreate(  # type: ignore[no-untyped-call]
            messages=[{"role": "user", "content": self.prompt}],
            **asdict(self.parameters),
        )
        return Reply(**resp)


@dataclass
class Result(Generic[P]):
    """A combined sample and reply"""

    sample: Sample[P]
    prompt: str
    reply: Reply
