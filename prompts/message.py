from dataclasses import dataclass
from typing import Any, List, Literal

Role = Literal["assistant", "system", "user"]


@dataclass
class Message:
    role: Role
    content: str

    @classmethod
    def from_format_string(cls, role: Role, content: str, **kwargs: Any) -> "Message":
        return cls(role, content.format(**kwargs))


Messages = List[Message]
