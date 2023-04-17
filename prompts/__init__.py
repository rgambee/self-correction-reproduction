# mypy: disable-error-code="type-arg"
"""Prompt functions suitable for OpenAI's /chat/completions endpoint"""
from functools import singledispatch
from types import ModuleType

from loaders import Sample
from loaders.bbq import BBQSample
from loaders.law import LawSample
from loaders.winogender import WinogenderSample

from . import bbq, law, winogender
from .message import Message, Messages, format_messages


@singledispatch
def prompt_question(sample: Sample) -> Messages:
    """Generate a prompt following the basic question format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_question.register
def _prompt_question_bbq(sample: BBQSample) -> Messages:
    return _prompt_question(sample, bbq)


@prompt_question.register
def _prompt_question_law(sample: LawSample) -> Messages:
    return _prompt_question(sample, law)


@prompt_question.register
def _prompt_question_winogender(sample: WinogenderSample) -> Messages:
    return _prompt_question(sample, winogender)


def _prompt_question(
    sample: Sample,
    module: ModuleType,
) -> Messages:
    return format_messages(
        messages=(
            Message(role="user", content=module.PREAMBLE),
            Message(role="user", content=module.POSTAMBLE),
        ),
        sample=sample,
    )


@singledispatch
def prompt_instruction_following(sample: Sample) -> Messages:
    """Generate a prompt following the Question + Instruction Following format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_instruction_following.register
def _prompt_instruction_following_bbq(sample: BBQSample) -> Messages:
    return _prompt_instruction_following(sample, bbq)


@prompt_instruction_following.register
def _prompt_instruction_following_law(sample: LawSample) -> Messages:
    return _prompt_instruction_following(sample, law)


@prompt_instruction_following.register
def _prompt_instruction_following_winogender(sample: WinogenderSample) -> Messages:
    return _prompt_instruction_following(sample, winogender)


def _prompt_instruction_following(
    sample: Sample,
    module: ModuleType,
) -> Messages:
    return format_messages(
        messages=[
            Message(role="user", content=module.PREAMBLE),
            Message(role="user", content=module.DEBIAS_INSTRUCTIONS),
            Message(role="user", content=module.POSTAMBLE),
        ],
        sample=sample,
    )


@singledispatch
def prompt_chain_of_thought_a(sample: Sample) -> Messages:
    """Generate a prompt to elicit chain-of-thought reasoning from the model"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_bbq(sample: BBQSample) -> Messages:
    return _prompt_chain_of_thought_a(sample, bbq)


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_law(sample: LawSample) -> Messages:
    return _prompt_chain_of_thought_a(sample, law)


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_winogender(sample: WinogenderSample) -> Messages:
    return _prompt_chain_of_thought_a(sample, winogender)


def _prompt_chain_of_thought_a(
    sample: Sample,
    module: ModuleType,
) -> Messages:
    return format_messages(
        messages=[
            Message(role="user", content=module.PREAMBLE),
            Message(role="assistant", content=module.CHAIN_OF_THOUGHT),
        ],
        sample=sample,
    )


@singledispatch
def prompt_chain_of_thought_b(
    sample: Sample,
    model_reasoning: str,
) -> Messages:
    """Generate a prompt that incorporates the model's chain-of-thought reasoning"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_bbq(
    sample: BBQSample,
    model_reasoning: str,
) -> Messages:
    return _prompt_chain_of_thought_b(sample, model_reasoning, bbq)


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_law(
    sample: LawSample,
    model_reasoning: str,
) -> Messages:
    return _prompt_chain_of_thought_b(sample, model_reasoning, law)


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_winogender(
    sample: WinogenderSample,
    model_reasoning: str,
) -> Messages:
    return _prompt_chain_of_thought_b(sample, model_reasoning, winogender)


def _prompt_chain_of_thought_b(
    sample: Sample,
    model_reasoning: str,
    module: ModuleType,
) -> Messages:
    return format_messages(
        messages=[
            Message(role="user", content=module.PREAMBLE),
            Message(role="assistant", content=module.CHAIN_OF_THOUGHT),
            Message(role="assistant", content=model_reasoning),
            Message(role="user", content=module.POSTAMBLE_COT),
        ],
        sample=sample,
    )
