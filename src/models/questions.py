from pydantic import BaseModel, Field
import uuid
from .minimalSource import MinimalSource


class UnansweredQuestion(BaseModel):
    """Represent a question without reference answer annotations.

    Attributes:
        question_id: Unique identifier for the question.
        question: Raw question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question annotated with sources and expected answer.

    Attributes:
        sources: Reference source spans supporting the answer.
        answer: Expected answer text.
    """

    sources: list[MinimalSource]
    answer: str
