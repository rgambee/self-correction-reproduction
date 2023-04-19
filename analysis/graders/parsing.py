import logging
import string
from typing import Any, Optional

from eval.result import Result


def result_to_answer(result: Result[Any]) -> Optional[str]:
    logger = logging.getLogger(__name__)
    try:
        answer = result.reply.choices[0].message.content
    except IndexError as err:
        logger.debug("Could not find answer in result due to %r", err)
        return None

    model_answer = answer.strip(string.whitespace + string.punctuation).lower()
    return model_answer
