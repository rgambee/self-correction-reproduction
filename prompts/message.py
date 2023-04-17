from dataclasses import asdict, dataclass, replace
from typing import Any, List, Literal

from loaders import Sample

Role = Literal["assistant", "system", "user"]


@dataclass
class Message:
    role: Role
    content: str

    @classmethod
    def from_format_string(cls, role: Role, content: str, **kwargs: Any) -> "Message":
        return cls(role, content.format(**kwargs))


Messages = List[Message]


def format_messages(
    messages: Messages,
    sample: Sample,  # type: ignore[type-arg]
) -> Messages:
    kwargs = asdict(sample)
    kwargs.update(asdict(sample.parameters))
    formatted: Messages = []
    for msg in messages:
        formatted.append(replace(msg, content=msg.content.format(**kwargs)))
    return formatted
