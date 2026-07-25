import pytest
import pickle
from pathlib import Path
from src.retriever import CodebaseRetriever
from src.indexer import BM25Index
from src.chunker import Chunk


def test_search_invalid_k() -> None:
    """Must return an empty list if k <= 0."""
    retriever = CodebaseRetriever()
    retriever.index = BM25Index(
        chunks=[],
        doc_lengths=[],
        avg_doc_len=0.0,
        df={},
        idf={},
        doc_term_freqs=[]
    )

    assert retriever.search("query", k=0) == []
    assert retriever.search("query", k=-5) == []


def test_search_empty_query() -> None:
    """
    Must return an empty list if the query is empty or only contains spaces.
    """
    retriever = CodebaseRetriever()
    retriever.index = BM25Index(
            chunks=[],
            doc_lengths=[],
            avg_doc_len=0.0,
            df={},
            idf={},
            doc_term_freqs=[]
        )

    assert retriever.search("", k=5) == []
    assert retriever.search("    ", k=5) == []


def test_search_without_index() -> None:
    """
    Must raise a ValueError if search is called before load_index.
    """
    retriever = CodebaseRetriever()

    with pytest.raises(ValueError, match="Index is not loaded"):
        retriever.search("test query", k=5)


def test_load_index_corrupted_or_missing() -> None:
    """
    Must raise a RuntimeError if the .pkl file does not exist.
    """
    retriever = CodebaseRetriever()

    with pytest.raises(RuntimeError, match="not found or corrupted"):
        retriever.load_index("imaginary_folder/fake_index.pkl")


def test_valid_search(tmp_path: Path) -> None:
    """
    Test a valid search returning the expected top-k sources.
    """
    dummy_chunk = Chunk(
        file_path="dummy_file.py",
        content="This is a test chunk about machine learning.",
        first_character_index=0,
        last_character_index=44
    )
    dummy_index = BM25Index(
        chunks=[dummy_chunk],
        doc_lengths=[8],
        avg_doc_len=8.0,
        df={"machine": 1, "learning": 1},
        idf={"machine": 1.5, "learning": 1.5},
        doc_term_freqs=[{"machine": 1, "learning": 1}]
    )

    index_path = tmp_path / "valid_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(dummy_index.model_dump(), f)

    retriever = CodebaseRetriever()
    retriever.load_index(str(index_path))

    results = retriever.search("machine learning", k=1)

    assert len(results) == 1, "Should have found exactly 1 result."
    assert results[0].file_path == "dummy_file.py"
    assert results[0].first_character_index == 0
