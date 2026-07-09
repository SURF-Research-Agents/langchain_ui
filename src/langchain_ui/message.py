from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """A simple chat message.

    Attributes:
        role: Either "user" or "assistant" indicating the message source.
        content: The textual content of the message.
    """

    role: Literal["user", "assistant"]
    content: str


class OpenAIRequest(BaseModel):
    """A simple Request message

    Attributes:
        model: the name of the model
        user:
        stream:
        message:
    """

    model: str
    user: str
    stream: bool
    messages: list[Message]
