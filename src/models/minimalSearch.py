import uuid
from pydantic import BaseModel, Field
from .minimalSource import MinimalSource


class MinimalSearchResults(BaseModel):
    """Container for retrieval results associated with a single question.

    Attributes:
        question_id: Unique identifier of the question.
        question: Raw question text.
        retrieved_sources: Retrieved source spans supporting the question.
    """

    question_id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval result enriched with an optional generated answer.

    Attributes:
        answer: Generated answer text, when available.
    """

    answer: str
