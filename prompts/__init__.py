# mypy: disable-error-code="type-arg"
from dataclasses import asdict
from functools import singledispatch

from loaders import Question
from loaders.law import LawQuestion

from . import law


@singledispatch
def format_preamble(question: Question) -> str:
    raise TypeError(f"Unsupported question type: {type(question)}")


@format_preamble.register
def _format_preamble_law(question: LawQuestion) -> str:
    return law.PREAMBLE.format(**asdict(question.parameters))


@singledispatch
def prompt_question(question: Question) -> str:
    """Generate a prompt following the basic question format"""
    raise TypeError(f"Unsupported question type: {type(question)}")


@prompt_question.register
def _prompt_question_law(question: LawQuestion) -> str:
    return _prompt_question(question, law.POSTAMBLE)


def _prompt_question(question: Question, postamble: str) -> str:
    return "\n".join(
        (
            "",
            format_preamble(question),
            "",
            "",
            postamble,
        ),
    )


@singledispatch
def prompt_instruction_following(question: Question) -> str:
    """Generate a prompt following the Question + Instruction Following format"""
    raise TypeError(f"Unsupported question type: {type(question)}")


@prompt_instruction_following.register
def _prompt_instruction_following_law(question: LawQuestion) -> str:
    return _prompt_instruction_following(
        question,
        law.DEBIAS_INSTRUCTIONS,
        law.POSTAMBLE,
    )


def _prompt_instruction_following(
    question: Question,
    debias_instructions: str,
    postamble: str,
) -> str:
    return "\n".join(
        (
            "",
            format_preamble(question),
            "",
            "",
            debias_instructions,
            "",
            "",
            postamble,
        ),
    )


@singledispatch
def prompt_chain_of_thought_a(question: Question) -> str:
    """Generate a prompt to elicit chain-of-thought reasoning from the model"""
    raise TypeError(f"Unsupported question type: {type(question)}")


@prompt_chain_of_thought_a.register
def _prompt_chain_of_thought_a_law(question: LawQuestion) -> str:
    return _prompt_chain_of_thought_a(question, law.CHAIN_OF_THOUGHT)


def _prompt_chain_of_thought_a(question: Question, chain_of_thought: str) -> str:
    return "\n".join(
        (
            "",
            format_preamble(question),
            "",
            "",
            chain_of_thought,
        ),
    )


@singledispatch
def prompt_chain_of_thought_b(question: Question, model_reasoning: str) -> str:
    """Generate a prompt that incorporates the model's chain-of-thought reasoning"""
    raise TypeError(f"Unsupported question type: {type(question)}")


@prompt_chain_of_thought_b.register
def _prompt_chain_of_thought_b_law(question: LawQuestion, model_reasoning: str) -> str:
    return _prompt_chain_of_thought_b(
        question,
        law.CHAIN_OF_THOUGHT,
        model_reasoning,
        law.POSTAMBLE_COT,
    )


def _prompt_chain_of_thought_b(
    question: Question,
    chain_of_thought: str,
    model_reasoning: str,
    postamble: str,
) -> str:
    return "\n".join(
        (
            "",
            format_preamble(question),
            "",
            "",
            chain_of_thought + model_reasoning.rstrip(),
            "",
            "",
            postamble,
        ),
    )
