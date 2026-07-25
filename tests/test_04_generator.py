from unittest.mock import patch, MagicMock
from src.generator import RagGenerator
from src.models.minimalSource import MinimalSource


def test_generator_no_sources() -> None:
    """
    Must return the default fallback string if no sources are provided.
    """
    generator = RagGenerator()
    generator.tokenizer = MagicMock()
    generator.model = MagicMock()
    generator.tokenizer.decode.return_value = "I don't know."

    answer = generator.answer("What is vLLM?", [])

    assert answer == "I don't know."


def test_build_prompt_truncation() -> None:
    """
    Must truncate the context if it exceeds max_context_length.
    """
    generator = RagGenerator(max_context_length=20)
    generator.tokenizer = MagicMock()
    generator.tokenizer.apply_chat_template.side_effect = \
        lambda conv, **kwargs: str(conv)

    long_source = MinimalSource(
        file_path="dummy.py",
        first_character_index=0,
        last_character_index=100
    )

    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "A" * 100
        mock_open.return_value.__enter__.return_value = mock_file

        prompt = generator.build_prompt("Test?", [long_source])

        assert "A" * 20 in prompt
        assert "A" * 21 not in prompt


def test_generator_answer_with_mocked_model() -> None:
    """
    Must successfully generate an answer without loading the real model.
    """
    generator = RagGenerator()
    generator.tokenizer = MagicMock()
    generator.model = MagicMock()

    generator.model.generate.return_value = [[1, 2, 3]]
    generator.tokenizer.decode.return_value = "This is a mocked answer."

    dummy_source = MinimalSource(
        file_path="dummy.py",
        first_character_index=0,
        last_character_index=17
    )

    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = "def dummy(): pass"
        mock_open.return_value.__enter__.return_value = mock_file

        answer = generator.answer("How does it work?", [dummy_source])

        assert answer is not None
        assert len(answer) > 0
