from typing import Literal
from pydantic import BaseModel

class Message(BaseModel):
    """A chat message model.

    Attributes:
        role: Either ``"user"`` or ``"assistant"`` indicating who sent the message.
        content: The textual content of the message.
    """
    role: Literal["user", "assistant"]
    content: str

class OpenAIRequest(BaseModel):
    """A OpenAI request formatter

    Attributes:
        model:
        user:
        stream:
        messages:
    """
    model: str
    user: str
    stream: bool
    messages: list[Message]