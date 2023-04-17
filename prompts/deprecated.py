# mypy: disable-error-code="type-arg"
from dataclasses import asdict
from functools import singledispatch

from loaders import Sample
from loaders.bbq import BBQSample
from loaders.law import LawSample
from loaders.winogender import WinogenderSample

from . import bbq, law, winogender
from .message import Message, Messages


@singledispatch
def format_preamble(sample: Sample) -> Messages:
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@format_preamble.register
def _format_preamble_bbq(sample: BBQSample) -> Messages:
    kwargs = asdict(sample)
    kwargs.update(asdict(sample.parameters))
    return [
        Message.from_format_string(role="user", content=bbq.PREAMBLE, **kwargs),
    ]


@format_preamble.register
def _format_preamble_law(sample: LawSample) -> Messages:
    kwargs = asdict(sample.parameters)
    return [
        Message.from_format_string(role="user", content=law.PREAMBLE, **kwargs),
    ]


@format_preamble.register
def _format_preamble_winogender(sample: WinogenderSample) -> Messages:
    kwargs = asdict(sample.parameters)
    return [
        Message.from_format_string(role="user", content=winogender.PREAMBLE, **kwargs),
    ]


def prompt_question(sample: Sample, postamble: str) -> Messages:
    messages = format_preamble(sample)
    messages.append(
        Message(role="user", content=postamble),
    )
    return messages


def prompt_instruction_following(
    sample: Sample,
    debias_instructions: str,
    postamble: str,
) -> Messages:
    messages = format_preamble(sample)
    messages.append(Message(role="user", content=debias_instructions))
    messages.append(Message(role="user", content=postamble))
    return messages


def prompt_chain_of_thought_a(sample: Sample, chain_of_thought: str) -> Messages:
    messages = format_preamble(sample)
    messages.append(Message(role="assistant", content=chain_of_thought))
    return messages


def prompt_chain_of_thought_b(
    sample: Sample,
    chain_of_thought: str,
    model_reasoning: str,
    postamble: str,
) -> Messages:
    messages = format_preamble(sample)
    messages.append(Message(role="assistant", content=chain_of_thought))
    messages.append(Message(role="assistant", content=model_reasoning))
    messages.append(Message(role="user", content=postamble))
    return messages
