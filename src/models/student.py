from pydantic import BaseModel
from .minimalSearch import MinimalSearchResults, MinimalAnswer


class StudentSearchResults(BaseModel):
    """Represent retrieval outputs for a batch of questions.

    Attributes:
        search_results: Retrieval results for each processed question.
        k: Number of top retrieved sources stored per question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent retrieval outputs augmented with generated answers.

    Attributes:
        k: Number of top retrieved sources used per question.
        search_results: Results containing retrieved sources and answers.
    """

    k: int
    search_results: list[MinimalAnswer]
