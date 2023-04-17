import string
import unittest
from dataclasses import asdict
from typing import cast

import prompts
from loaders import Sample
from prompts import bbq, law, winogender
from prompts.deprecated import format_preamble
from prompts.message import Message, Messages
from tests.utils import BBQ_SAMPLE, LAW_SAMPLE, WINOGENDER_SAMPLE


class TestBBQPrompts(unittest.TestCase):
    PROMPT_MODULE = bbq
    SAMPLE: Sample = BBQ_SAMPLE  # type: ignore[type-arg]

    @property
    def preamble(self) -> str:
        format_string = cast(str, getattr(self.PROMPT_MODULE, "PREAMBLE"))
        kwargs = asdict(self.SAMPLE)
        kwargs.update(asdict(self.SAMPLE.parameters))
        return format_string.format(**kwargs)

    @property
    def debias_instructions(self) -> str:
        return cast(str, getattr(self.PROMPT_MODULE, "DEBIAS_INSTRUCTIONS"))

    @property
    def chain_of_thought(self) -> str:
        return cast(str, getattr(self.PROMPT_MODULE, "CHAIN_OF_THOUGHT"))

    @property
    def postamble(self) -> str:
        return cast(str, getattr(self.PROMPT_MODULE, "POSTAMBLE"))

    @property
    def postamble_cot(self) -> str:
        return cast(str, getattr(self.PROMPT_MODULE, "POSTAMBLE_COT"))

    def check_whitespace(self, messages: Messages) -> None:
        """Check that messages don't start or end with whitespace"""
        for msg in messages:
            self.assertNotIn(msg.content[0], string.whitespace)
            self.assertNotIn(msg.content[-1], string.whitespace)

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(self.SAMPLE.parameters.context, preamble.content)
        self.assertIn(self.SAMPLE.parameters.question, preamble.content)

    def search_messages(self, messages: Messages, text: str) -> bool:
        for msg in messages:
            if text in msg.content:
                return True
        return False

    def test_preamble(self) -> None:
        """Test that the preamble is formatted correctly"""
        messages = format_preamble(self.SAMPLE)
        self.check_whitespace(messages)
        self.check_preamble_contents(messages[0])

    def test_question(self) -> None:
        """Test that the plain question prompt contains the expected text"""
        messages = prompts.prompt_question(self.SAMPLE)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertFalse(self.search_messages(messages, self.debias_instructions))
        self.assertFalse(self.search_messages(messages, self.chain_of_thought))
        self.assertTrue(self.search_messages(messages, self.postamble))
        self.assertFalse(self.search_messages(messages, self.postamble_cot))

    def test_instruction_following(self) -> None:
        """Test that the instruction-following prompt contains the expected text"""
        messages = prompts.prompt_instruction_following(self.SAMPLE)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertTrue(self.search_messages(messages, self.debias_instructions))
        self.assertFalse(self.search_messages(messages, self.chain_of_thought))
        self.assertTrue(self.search_messages(messages, self.postamble))
        self.assertFalse(self.search_messages(messages, self.postamble_cot))

    def test_chain_of_thought(self) -> None:
        """Test that the chain-of-thought prompt contains the expected text"""
        messages = prompts.prompt_chain_of_thought_a(self.SAMPLE)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertFalse(self.search_messages(messages, self.debias_instructions))
        self.assertTrue(self.search_messages(messages, self.chain_of_thought))
        self.assertFalse(self.search_messages(messages, self.postamble))
        self.assertFalse(self.search_messages(messages, self.postamble_cot))

        reasoning = "I should answer this question correctly."
        messages = prompts.prompt_chain_of_thought_b(self.SAMPLE, reasoning)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertFalse(self.search_messages(messages, self.debias_instructions))
        self.assertTrue(self.search_messages(messages, self.chain_of_thought))
        self.assertFalse(self.search_messages(messages, self.postamble))
        self.assertTrue(self.search_messages(messages, self.postamble_cot))
        self.assertTrue(self.search_messages(messages, reasoning))


class TestLawPrompts(TestBBQPrompts):
    PROMPT_MODULE = law
    SAMPLE = LAW_SAMPLE

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(
            (
                "The first student is a Other female with a raw LSAT score of 39.0 "
                "and an undergraduate GPA of 3.1."
            ),
            # Condense to one line to make is easier to search for a substring
            preamble.content.replace("\n", " "),
        )


class TestWinogenderPrompts(TestBBQPrompts):
    PROMPT_MODULE = winogender
    SAMPLE = WINOGENDER_SAMPLE

    @property
    def postamble(self) -> str:
        format_string = cast(str, getattr(self.PROMPT_MODULE, "POSTAMBLE"))
        return format_string.format(**asdict(self.SAMPLE.parameters))

    @property
    def postamble_cot(self) -> str:
        format_string = cast(str, getattr(self.PROMPT_MODULE, "POSTAMBLE_COT"))
        return format_string.format(**asdict(self.SAMPLE.parameters))

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(
            "The technician told the customer that _ had completed the repair.",
            preamble.content,
        )

    def test_question(self) -> None:
        """Test that the plain question prompt contains the expected text"""
        messages = prompts.prompt_question(self.SAMPLE)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertFalse(self.search_messages(messages, self.debias_instructions))
        self.assertFalse(self.search_messages(messages, self.chain_of_thought))
        self.assertTrue(self.search_messages(messages, self.postamble))

    def test_instruction_following(self) -> None:
        """Test that the instruction-following prompt contains the expected text"""
        messages = prompts.prompt_instruction_following(self.SAMPLE)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertTrue(self.search_messages(messages, self.debias_instructions))
        self.assertFalse(self.search_messages(messages, self.chain_of_thought))
        self.assertTrue(self.search_messages(messages, self.postamble))

    def test_chain_of_thought(self) -> None:
        """Test that the chain-of-thought prompt contains the expected text"""
        messages = prompts.prompt_chain_of_thought_a(self.SAMPLE)
        self.check_whitespace(messages)
        self.assertTrue(self.search_messages(messages, self.preamble))
        self.assertFalse(self.search_messages(messages, self.debias_instructions))
        self.assertTrue(self.search_messages(messages, self.chain_of_thought))
        self.assertTrue(self.search_messages(messages, self.postamble_cot))


if __name__ == "__main__":
    unittest.main()
