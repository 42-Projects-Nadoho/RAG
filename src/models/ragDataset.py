from pydantic import BaseModel
from .questions import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    """Container for RAG questions used by indexing and evaluation.

    Attributes:
        rag_questions: List of answered or unanswered question entries.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
