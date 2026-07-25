"""
Test module for the pipeline orchestrator.
"""
from unittest.mock import patch, MagicMock
from src.pipeline import RagPipeline


@patch("src.pipeline.CodebaseIndexer")
def test_pipeline_index_valid(mock_indexer: MagicMock) -> None:
    """
    Verify that the index method correctly initializes and saves the index.
    """
    pipeline = RagPipeline()
    pipeline.index(max_chunk_size=500)
    mock_indexer.return_value.build_index.assert_called_once()
    mock_indexer.return_value.save_index.assert_called_once()


@patch("src.pipeline.TerminalColors.error")
def test_pipeline_index_invalid_chunk(mock_error: MagicMock) -> None:
    """
    Verify that an invalid chunk size triggers an error and prevents indexing.
    """
    pipeline = RagPipeline()
    pipeline.index(max_chunk_size=-10)
    pipeline.index(max_chunk_size=0)
    pipeline.index(max_chunk_size=5000)
    assert mock_error.call_count == 3


@patch("src.pipeline.CodebaseIndexer")
@patch("src.pipeline.TerminalColors.error")
def test_pipeline_index_permission_error(
    mock_error: MagicMock,
    mock_indexer: MagicMock
) -> None:
    """
    Verify handling of PermissionError during index saving.
    """
    mock_indexer.return_value.save_index.side_effect = \
        PermissionError("Denied")
    pipeline = RagPipeline()
    pipeline.index(max_chunk_size=500)
    mock_error.assert_called()


@patch("src.pipeline.CodebaseRetriever")
def test_pipeline_search_valid(mock_retriever: MagicMock) -> None:
    """
    Verify that a valid search loads the index and executes the query.
    """
    pipeline = RagPipeline()
    mock_instance = mock_retriever.return_value
    mock_instance.search.return_value = []

    pipeline.search(query="test", k=5)
    mock_instance.load_index.assert_called_once()


@patch("src.pipeline.TerminalColors.error")
def test_pipeline_search_invalid_k(mock_error: MagicMock) -> None:
    """
    Verify that searching with an invalid k value triggers an error.
    """
    pipeline = RagPipeline()
    pipeline.search(query="test", k=-1)
    pipeline.search(query="test", k=0)
    assert mock_error.call_count == 2


@patch("src.pipeline.CodebaseRetriever")
@patch("src.pipeline.TerminalColors.error")
def test_pipeline_search_load_failure(
    mock_error: MagicMock, mock_retriever: MagicMock
) -> None:
    """
    Verify handling of index loading failure during search.
    """
    mock_retriever.return_value.load_index.side_effect = Exception("Corrupted")
    pipeline = RagPipeline()
    pipeline.search(query="test", k=5)
    mock_error.assert_called_once()


@patch("src.pipeline.TerminalColors.error")
def test_pipeline_search_dataset_missing(mock_error: MagicMock) -> None:
    """
    Verify that attempting to search a missing dataset file gracefully fails.
    """
    pipeline = RagPipeline()
    with patch("src.pipeline.CodebaseRetriever"):
        pipeline.search_dataset("ghost.json", 5, "out")
    mock_error.assert_called()


@patch("builtins.open")
@patch("src.pipeline.TerminalColors.error")
def test_pipeline_search_dataset_invalid_json(
    mock_error: MagicMock, mock_open: MagicMock
) -> None:
    """
    Verify handling of JSONDecodeError when reading a corrupted dataset.
    """
    mock_file = MagicMock()
    mock_file.read.return_value = "{bad_json: "
    mock_open.return_value.__enter__.return_value = mock_file

    pipeline = RagPipeline()
    with patch("src.pipeline.CodebaseRetriever"):
        pipeline.search_dataset("fake.json", 5, "out")

    mock_error.assert_called()


@patch("src.pipeline.RagGenerator")
@patch("src.pipeline.TerminalColors.warning")
def test_pipeline_answer_invalid_k(
    mock_warning: MagicMock, mock_generator: MagicMock
) -> None:
    """
    Verify that answering a query with an invalid k value triggers an warning.
    """
    pipeline = RagPipeline()
    pipeline.answer(query="test", k=0)
    pipeline.answer(query="test", k=-5)
    assert mock_warning.call_count == 2


@patch("src.pipeline.RagGenerator")
@patch("src.pipeline.CodebaseRetriever")
def test_pipeline_answer_valid(
    mock_retriever: MagicMock, mock_generator: MagicMock
) -> None:
    """
    Verify that the answer method correctly orchestrates retrieval
    and generation.
    """
    pipeline = RagPipeline()

    mock_ret_instance = mock_retriever.return_value
    mock_ret_instance.search.return_value = []

    mock_gen_instance = mock_generator.return_value
    mock_gen_instance.answer.return_value = "Fake answer"

    pipeline.answer(query="test", k=3)

    mock_gen_instance.load_model.assert_called_once()
    mock_gen_instance.answer.assert_called_once()


@patch("src.pipeline.RagGenerator")
@patch("src.pipeline.CodebaseRetriever")
@patch("src.pipeline.TerminalColors.error")
def test_pipeline_answer_model_failure(
    mock_error: MagicMock,
    mock_retriever: MagicMock,
    mock_generator: MagicMock
) -> None:
    """
    Verify handling of a model loading failure.
    """
    mock_retriever.return_value.search.return_value = []
    mock_generator.return_value.load_model.side_effect = Exception("OOM")

    pipeline = RagPipeline()
    pipeline.answer(query="test", k=3)

    mock_error.assert_called()


@patch("src.pipeline.TerminalColors.error")
def test_pipeline_answer_dataset_missing(mock_error: MagicMock) -> None:
    """
    Verify that attempting to answer a missing dataset file gracefully fails.
    """
    pipeline = RagPipeline()
    pipeline.answer_dataset("ghost.json", "out")
    mock_error.assert_called_once()


@patch("src.pipeline.RagEvaluator.evaluate")
def test_pipeline_evaluate(mock_evaluate: MagicMock) -> None:
    """
    Verify that the evaluate method correctly delegates to the RagEvaluator.
    """
    pipeline = RagPipeline()
    pipeline.evaluate("student.json", "dataset.json")
    mock_evaluate.assert_called_once_with("student.json", "dataset.json")
