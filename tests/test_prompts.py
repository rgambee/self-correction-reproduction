import string
import unittest

import prompts
from loaders.law import LawParameters, LawQuestion
from prompts import law


class TestLawPrompts(unittest.TestCase):
    SAMPLE_QUESTION = LawQuestion(
        dataset="law",
        category="",
        id=0,
        parameters=LawParameters(
            race="Other",
            sex="1",
            LSAT=39.0,
            UGPA=3.1,
            region_first="GL",
            ZFYA=-0.98,
            sander_index=0.782738095238,
            first_pf=1.0,
        ),
        answers=["no", "yes"],
        correct_answer=1,
    )

    def test_preamble(self) -> None:
        """Test that the preamble is formatted correctly"""
        preamble = prompts.format_preamble(self.SAMPLE_QUESTION)
        # The preamble shouldn't have any leading or trailing whitespace
        self.assertEqual(preamble, preamble.strip())
        # Condense to one line to make is easier to search for a substring
        preamble = preamble.replace("\n", " ")
        self.assertIn(
            (
                "The first student is a Other female with a raw LSAT score of 39.0 "
                "and an undergraduate GPA of 3.1."
            ),
            preamble,
        )

    def check_whitespace(self, text: str) -> None:
        # The first character should be a newline,
        # and the last character should not be whitespace.
        self.assertEqual(text[0], "\n")
        self.assertNotIn(text[-1], string.whitespace)

    def test_question(self) -> None:
        """Test that the plain question prompt contains the expected text"""
        prompt = prompts.prompt_question(self.SAMPLE_QUESTION)
        self.check_whitespace(prompt)
        self.assertNotIn(law.DEBIAS_INSTRUCTIONS, prompt)
        self.assertNotIn(law.CHAIN_OF_THOUGHT, prompt)
        self.assertIn(law.POSTAMBLE, prompt)
        self.assertNotIn(law.POSTAMBLE_COT, prompt)

    def test_instruction_following(self) -> None:
        """Test that the instruction-following prompt contains the expected text"""
        prompt = prompts.prompt_instruction_following(self.SAMPLE_QUESTION)
        self.check_whitespace(prompt)
        self.assertIn(law.DEBIAS_INSTRUCTIONS, prompt)
        self.assertNotIn(law.CHAIN_OF_THOUGHT, prompt)
        self.assertIn(law.POSTAMBLE, prompt)
        self.assertNotIn(law.POSTAMBLE_COT, prompt)

    def test_chain_of_thought(self) -> None:
        """Test that the chain-of-thought prompt contains the expected text"""
        prompt = prompts.prompt_chain_of_thought_a(self.SAMPLE_QUESTION)
        self.check_whitespace(prompt)
        self.assertNotIn(law.DEBIAS_INSTRUCTIONS, prompt)
        self.assertIn(law.CHAIN_OF_THOUGHT, prompt)
        self.assertNotIn(law.POSTAMBLE, prompt)
        self.assertNotIn(law.POSTAMBLE_COT, prompt)

        reasoning = "I should answer this question correctly."
        prompt = prompts.prompt_chain_of_thought_b(self.SAMPLE_QUESTION, reasoning)
        self.check_whitespace(prompt)
        self.assertNotIn(law.DEBIAS_INSTRUCTIONS, prompt)
        self.assertIn(law.CHAIN_OF_THOUGHT, prompt)
        self.assertNotIn(law.POSTAMBLE, prompt)
        self.assertIn(law.POSTAMBLE_COT, prompt)
        self.assertIn(reasoning, prompt, prompt)


if __name__ == "__main__":
    unittest.main()
