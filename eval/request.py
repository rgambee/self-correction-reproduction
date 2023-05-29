from dataclasses import asdict, dataclass
from typing import Generic, Optional

import openai

from eval.result import Reply
from loaders import P, Sample
from prompts.message import Messages


@dataclass
class RequestParameters:
    model: str
    max_tokens: int
    temperature: float
    timeout: Optional[float] = None
    # OpenAI's API uses `n` for the number of completions to return for each request.
    # Not the clearest parameter name, as Pylint notices.
    # pylint: disable-next=invalid-name
    n: int = 1


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
        return Reply.from_dict(resp)
