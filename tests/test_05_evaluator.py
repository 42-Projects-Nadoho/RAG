import json
from pathlib import Path
from src.evaluation import RagEvaluator
from unittest.mock import patch, MagicMock


def test_evaluate_perfect_match(tmp_path: Path) -> None:
    student_file = tmp_path / "student.json"
    dataset_file = tmp_path / "dataset.json"

    student_data = {
        "k": 1,
        "search_results": [
            {
                "question_id": 1,
                "question": "test?",
                "retrieved_sources": [
                    {
                        "file_path": "test.py",
                        "first_character_index": 0,
                        "last_character_index": 10
                    }
                ]
            }
        ]
    }

    dataset_data = {
        "rag_questions": [
            {
                "question_id": 1,
                "question": "test?",
                "sources": [
                    {
                        "file_path": "test.py",
                        "first_character_index": 0,
                        "last_character_index": 10
                    }
                ]
            }
        ]
    }

    student_file.write_text(json.dumps(student_data), encoding="utf-8")
    dataset_file.write_text(json.dumps(dataset_data), encoding="utf-8")

    RagEvaluator.evaluate(str(student_file), str(dataset_file))


@patch("src.evaluation.TerminalColors.error")
def test_evaluate_missing_file(mock_error: MagicMock) -> None:
    RagEvaluator.evaluate("ghost_student.json", "ghost_dataset.json")
    mock_error.assert_called()
