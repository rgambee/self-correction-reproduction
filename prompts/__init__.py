# mypy: disable-error-code="type-arg"
"""Prompt functions suitable for OpenAI's /chat/completions endpoint"""
from dataclasses import asdict, dataclass
from functools import singledispatch
from typing import Any, List, Literal

from loaders import Sample
from loaders.bbq import BBQSample
from loaders.law import LawSample
from loaders.winogender import WinogenderSample

from . import bbq, law, winogender

Role = Literal["assistant", "system", "user"]


@dataclass
class Message:
    role: Role
    content: str

    @classmethod
    def from_format_string(cls, role: Role, content: str, **kwargs: Any) -> "Message":
        return cls(role, content.format(**kwargs))


Messages = List[Message]


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


@singledispatch
def prompt_question(sample: Sample) -> Messages:
    """Generate a prompt following the basic question format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_question.register
def _prompt_question_bbq(sample: BBQSample) -> Messages:
    return _prompt_question(sample, bbq.POSTAMBLE)


@prompt_question.register
def _prompt_question_law(sample: LawSample) -> Messages:
    return _prompt_question(sample, law.POSTAMBLE)


@prompt_question.register
def _prompt_question_winogender(sample: WinogenderSample) -> Messages:
    kwargs = asdict(sample.parameters)
    return _prompt_question(sample, winogender.POSTAMBLE.format(**kwargs))


def _prompt_question(sample: Sample, postamble: str) -> Messages:
    messages = format_preamble(sample)
    messages.append(
        Message(role="user", content=postamble),
    )
    return messages


@singledispatch
def prompt_instruction_following(sample: Sample) -> Messages:
    """Generate a prompt following the Question + Instruction Following format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_instruction_following.register
def _prompt_instruction_following_bbq(sample: BBQSample) -> Messages:
    return _prompt_instruction_following(
        sample=sample,
        debias_instructions=bbq.DEBIAS_INSTRUCTIONS,
        postamble=bbq.POSTAMBLE,
    )


@prompt_instruction_following.register
def _prompt_instruction_following_law(sample: LawSample) -> Messages:
    return _prompt_instruction_following(
        sample=sample,
        debias_instructions=law.DEBIAS_INSTRUCTIONS,
        postamble=law.POSTAMBLE,
    )


@prompt_instruction_following.register
def _prompt_instruction_following_winogender(sample: WinogenderSample) -> Messages:
    kwargs = asdict(sample.parameters)
    return _prompt_instruction_following(
        sample=sample,
        debias_instructions=winogender.DEBIAS_INSTRUCTIONS,
        postamble=winogender.POSTAMBLE.format(**kwargs),
    )


def _prompt_instruction_following(
    sample: Sample,
    debias_instructions: str,
    postamble: str,
) -> Messages:
    messages = format_preamble(sample)
    messages.append(Message(role="user", content=debias_instructions))
    messages.append(Message(role="user", content=postamble))
    return messages


@singledispatch
def prompt_chain_of_thought_a(sample: Sample) -> Messages:
    """Generate a prompt to elicit chain-of-thought reasoning from the model"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_bbq(sample: BBQSample) -> Messages:
    return _prompt_chain_of_thought_a(sample, bbq.CHAIN_OF_THOUGHT)


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_law(sample: LawSample) -> Messages:
    return _prompt_chain_of_thought_a(sample, law.CHAIN_OF_THOUGHT)


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_winogender(sample: WinogenderSample) -> Messages:
    return _prompt_chain_of_thought_a(sample, winogender.CHAIN_OF_THOUGHT)


def _prompt_chain_of_thought_a(sample: Sample, chain_of_thought: str) -> Messages:
    messages = format_preamble(sample)
    messages.append(Message(role="assistant", content=chain_of_thought))
    return messages


@singledispatch
def prompt_chain_of_thought_b(sample: Sample, model_reasoning: str) -> Messages:
    """Generate a prompt that incorporates the model's chain-of-thought reasoning"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_bbq(sample: BBQSample, model_reasoning: str) -> Messages:
    return _prompt_chain_of_thought_b(
        sample,
        bbq.CHAIN_OF_THOUGHT,
        model_reasoning,
        bbq.POSTAMBLE_COT,
    )


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_law(sample: LawSample, model_reasoning: str) -> Messages:
    return _prompt_chain_of_thought_b(
        sample,
        law.CHAIN_OF_THOUGHT,
        model_reasoning,
        law.POSTAMBLE_COT,
    )


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_winogender(
    sample: WinogenderSample, model_reasoning: str
) -> Messages:
    kwargs = asdict(sample.parameters)
    return _prompt_chain_of_thought_b(
        sample,
        winogender.CHAIN_OF_THOUGHT,
        model_reasoning,
        winogender.POSTAMBLE_COT.format(**kwargs),
    )


def _prompt_chain_of_thought_b(
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
