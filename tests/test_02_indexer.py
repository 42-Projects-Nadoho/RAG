import pytest
from pathlib import Path
from src.chunker import Chunk
from src.indexer import CodebaseIndexer, BM25Index


def test_tokenize_clean_text() -> None:
    """
    Verify that the tokenizer converts to lowercase and removes punctuation.
    """
    indexer = CodebaseIndexer()
    text = "Hello World! This is a TEST. 123_abc"
    tokens = indexer._tokenize(text)

    assert tokens == ["hello", "world", "this", "is", "a", "test", "123_abc"]


def test_tokenize_special_chars() -> None:
    """
    Verify that special characters (emojis, symbols) do not crash
    the tokenizer.
    """
    indexer = CodebaseIndexer()
    text = "RAG 🚀 & Machine Learning @ 2026!!!"
    tokens = indexer._tokenize(text)

    assert tokens == ["rag", "machine", "learning", "2026"]


def test_build_index_with_temp_files(tmp_path: Path) -> None:
    """
    Test the mathematical creation of the index on a mock directory.
    """
    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir()

    py_file = raw_dir / "test_script.py"
    py_file.write_text(
        "def dummy_function():\n    print('hello RAG')", encoding="utf-8"
    )

    md_file = raw_dir / "readme.md"
    md_file.write_text("# Doc\nThis is a RAG test.", encoding="utf-8")

    indexer = CodebaseIndexer(max_chunk_size=500)
    index = indexer.build_index(str(raw_dir))

    assert index is not None, "The index should not be None."
    assert len(index.chunks) > 0, "Chunks should have been created."
    assert index.avg_doc_len > 0, "Average document length must be > 0."

    assert "rag" in index.df, (
                               "The word 'rag' should be in the "
                               "document frequency dictionary."
                               )
    assert "dummy_function" in index.df, ("The function name should be"
                                          " indexed."
                                          )


def test_build_index_invalid_dir() -> None:
    """
    Verify that building an index on a non-existent directory
    raises an OSError.
    """
    indexer = CodebaseIndexer()

    with pytest.raises(OSError):
        indexer.build_index("this_directory_does_not_exist_12345")


def test_save_index(tmp_path: Path) -> None:
    """
    Test the serialization (saving) of the index into a binary pickle file.
    """
    indexer = CodebaseIndexer()

    dummy_chunk = Chunk(
        file_path="dummy.py",
        content="dummy content",
        first_character_index=0,
        last_character_index=13
    )
    dummy_index = BM25Index(
        chunks=[dummy_chunk],
        doc_lengths=[2],
        avg_doc_len=2.0,
        df={"dummy": 1, "content": 1},
        idf={"dummy": 0.5, "content": 0.5},
        doc_term_freqs=[{"dummy": 1, "content": 1}]
    )

    output_file = tmp_path / "fake_index.pkl"

    indexer.save_index(dummy_index, str(output_file))

    assert output_file.exists(), "The pickle file was not created."
    assert output_file.stat().st_size > 0, "The pickle file is empty."
