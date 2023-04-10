# mypy: disable-error-code="type-arg"
from dataclasses import asdict
from functools import singledispatch

from loaders import Sample
from loaders.bbq import BBQSample
from loaders.law import LawSample

from . import bbq, law


@singledispatch
def format_preamble(sample: Sample) -> str:
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@format_preamble.register
def _format_preamble_bbq(sample: BBQSample) -> str:
    kwargs = asdict(sample)
    kwargs.update(asdict(sample.parameters))
    return bbq.PREAMBLE.format(**kwargs)


@format_preamble.register
def _format_preamble_law(sample: LawSample) -> str:
    return law.PREAMBLE.format(**asdict(sample.parameters))


@singledispatch
def prompt_question(sample: Sample) -> str:
    """Generate a prompt following the basic question format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_question.register
def _prompt_question_bbq(sample: BBQSample) -> str:
    return _prompt_question(sample, bbq.POSTAMBLE)


@prompt_question.register
def _prompt_question_law(sample: LawSample) -> str:
    return _prompt_question(sample, law.POSTAMBLE)


def _prompt_question(sample: Sample, postamble: str) -> str:
    return "\n".join(
        (
            "",
            format_preamble(sample),
            "",
            "",
            postamble,
        ),
    )


@singledispatch
def prompt_instruction_following(sample: Sample) -> str:
    """Generate a prompt following the Question + Instruction Following format"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_instruction_following.register
def _prompt_instruction_following_bbq(sample: BBQSample) -> str:
    return _prompt_instruction_following(
        sample,
        bbq.DEBIAS_INSTRUCTIONS,
        bbq.POSTAMBLE,
    )


@prompt_instruction_following.register
def _prompt_instruction_following_law(sample: LawSample) -> str:
    return _prompt_instruction_following(
        sample,
        law.DEBIAS_INSTRUCTIONS,
        law.POSTAMBLE,
    )


def _prompt_instruction_following(
    sample: Sample,
    debias_instructions: str,
    postamble: str,
) -> str:
    return "\n".join(
        (
            "",
            format_preamble(sample),
            "",
            "",
            debias_instructions,
            "",
            "",
            postamble,
        ),
    )


@singledispatch
def prompt_chain_of_thought_a(sample: Sample) -> str:
    """Generate a prompt to elicit chain-of-thought reasoning from the model"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_bbq(sample: BBQSample) -> str:
    return _prompt_chain_of_thought_a(sample, bbq.CHAIN_OF_THOUGHT)


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_law(sample: LawSample) -> str:
    return _prompt_chain_of_thought_a(sample, law.CHAIN_OF_THOUGHT)


def _prompt_chain_of_thought_a(sample: Sample, chain_of_thought: str) -> str:
    return "\n".join(
        (
            "",
            format_preamble(sample),
            "",
            "",
            chain_of_thought,
        ),
    )


@singledispatch
def prompt_chain_of_thought_b(sample: Sample, model_reasoning: str) -> str:
    """Generate a prompt that incorporates the model's chain-of-thought reasoning"""
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_bbq(sample: BBQSample, model_reasoning: str) -> str:
    return _prompt_chain_of_thought_b(
        sample,
        bbq.CHAIN_OF_THOUGHT,
        model_reasoning,
        bbq.POSTAMBLE_COT,
    )


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_law(sample: LawSample, model_reasoning: str) -> str:
    return _prompt_chain_of_thought_b(
        sample,
        law.CHAIN_OF_THOUGHT,
        model_reasoning,
        law.POSTAMBLE_COT,
    )


def _prompt_chain_of_thought_b(
    sample: Sample,
    chain_of_thought: str,
    model_reasoning: str,
    postamble: str,
) -> str:
    return "\n".join(
        (
            "",
            format_preamble(sample),
            "",
            "",
            chain_of_thought + model_reasoning.rstrip(),
            "",
            "",
            postamble,
        ),
    )
