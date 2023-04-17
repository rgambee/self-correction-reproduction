# mypy: disable-error-code="type-arg"
"""Prompt functions suitable for OpenAI's /chat/completions endpoint"""
from functools import singledispatch

from loaders import Sample
from loaders.bbq import BBQSample
from loaders.law import LawSample
from loaders.winogender import WinogenderSample

from . import bbq, law, winogender
from .message import Messages


@singledispatch
def prompt_question(sample: Sample) -> Messages:
    """Generate a prompt following the basic question format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


prompt_question.register(BBQSample, bbq.prompt_question)
prompt_question.register(LawSample, law.prompt_question)
prompt_question.register(WinogenderSample, winogender.prompt_question)


@singledispatch
def prompt_instruction_following(sample: Sample) -> Messages:
    """Generate a prompt following the Question + Instruction Following format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


prompt_instruction_following.register(BBQSample, bbq.prompt_instruction_following)
prompt_instruction_following.register(LawSample, law.prompt_instruction_following)
prompt_instruction_following.register(
    WinogenderSample, winogender.prompt_instruction_following
)


@singledispatch
def prompt_chain_of_thought_a(sample: Sample) -> Messages:
    """Generate a prompt to elicit chain-of-thought reasoning from the model"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


prompt_chain_of_thought_a.register(BBQSample, bbq.prompt_chain_of_thought_a)
prompt_chain_of_thought_a.register(LawSample, law.prompt_chain_of_thought_a)
prompt_chain_of_thought_a.register(
    WinogenderSample, winogender.prompt_chain_of_thought_a
)


@singledispatch
def prompt_chain_of_thought_b(sample: Sample, model_reasoning: str) -> Messages:
    """Generate a prompt that incorporates the model's chain-of-thought reasoning"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


prompt_chain_of_thought_b.register(BBQSample, bbq.prompt_chain_of_thought_b)
prompt_chain_of_thought_b.register(LawSample, law.prompt_chain_of_thought_b)
prompt_chain_of_thought_b.register(
    WinogenderSample, winogender.prompt_chain_of_thought_b
)
