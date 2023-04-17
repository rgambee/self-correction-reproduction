import string
import unittest

import prompts
from loaders import Sample
from prompts import bbq, law, winogender
from prompts.message import Message, Messages, format_messages
from tests.utils import BBQ_SAMPLE, LAW_SAMPLE, WINOGENDER_SAMPLE


class TestBBQPrompts(unittest.TestCase):
    PROMPT_MODULE = bbq
    SAMPLE: Sample = BBQ_SAMPLE  # type: ignore[type-arg]

    @property
    def preamble(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.PREAMBLE, self.SAMPLE)

    @property
    def debias_instructions(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.DEBIAS_INSTRUCTIONS, self.SAMPLE)

    @property
    def chain_of_thought(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.CHAIN_OF_THOUGHT, self.SAMPLE)

    @property
    def postamble(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.POSTAMBLE, self.SAMPLE)

    @property
    def postamble_cot(self) -> Messages:
        return format_messages(self.PROMPT_MODULE.POSTAMBLE_COT, self.SAMPLE)

    def check_whitespace(self, messages: Messages) -> None:
        for msg in messages:
            # Check that messages don't start or end with whitespace
            self.assertNotIn(msg.content[0], string.whitespace)
            self.assertNotIn(msg.content[-1], string.whitespace)
            # Check that there aren't multiple consecutive spaces, which would be
            # indicative of an indentation issue
            self.assertNotIn("  ", msg.content)

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(self.SAMPLE.parameters.context, preamble.content)
        self.assertIn(self.SAMPLE.parameters.question, preamble.content)

    def assert_all_in(self, subsequence: Messages, parent_sequence: Messages) -> None:
        for message in subsequence:
            self.assertIn(message, parent_sequence)

    def assert_none_in(self, subsequence: Messages, parent_sequence: Messages) -> None:
        for message in subsequence:
            self.assertNotIn(message, parent_sequence)

    def test_preamble(self) -> None:
        """Test that the preamble is formatted correctly"""
        messages = self.preamble
        self.check_whitespace(messages)
        self.check_preamble_contents(messages[0])

    def test_question(self) -> None:
        """Test that the plain question prompt contains the expected text"""
        messages = prompts.prompt_question(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_none_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble, messages)

    def test_instruction_following(self) -> None:
        """Test that the instruction-following prompt contains the expected text"""
        messages = prompts.prompt_instruction_following(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_all_in(self.debias_instructions, messages)
        self.assert_none_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble, messages)

    def test_chain_of_thought(self) -> None:
        """Test that the chain-of-thought prompt contains the expected text"""
        messages = prompts.prompt_chain_of_thought_a(self.SAMPLE)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_all_in(self.chain_of_thought, messages)
        self.assert_none_in(self.postamble, messages)
        self.assert_none_in(self.postamble_cot, messages)

        reasoning = "I should answer this question correctly."
        messages = prompts.prompt_chain_of_thought_b(self.SAMPLE, reasoning)
        self.check_whitespace(messages)
        self.assert_all_in(self.preamble, messages)
        self.assert_none_in(self.debias_instructions, messages)
        self.assert_all_in(self.chain_of_thought, messages)
        self.assert_all_in(self.postamble_cot, messages)
        self.assertTrue(any(reasoning in msg.content for msg in messages))


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

    def check_preamble_contents(self, preamble: Message) -> None:
        self.assertIn(
            "The technician told the customer that _ had completed the repair.",
            preamble.content,
        )


if __name__ == "__main__":
    unittest.main()
