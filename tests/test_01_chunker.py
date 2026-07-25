from pathlib import Path
from src.chunker import chunk_markdown, chunk_python, Chunk


def test_chunk_wrong_extensions() -> None:
    """
    Must return an empty list if the file extension is incorrect.
    """
    assert chunk_markdown("script.py") == []
    assert chunk_markdown("document.txt") == []
    assert chunk_python("readme.md") == []
    assert chunk_python("script.js") == []


def test_chunk_missing_files() -> None:
    """
    Must return an empty list without raising an exception if the file
    does not exist.
    """
    assert chunk_markdown("ghost_file.md") == []
    assert chunk_python("ghost_file.py") == []


def test_chunk_empty_files(tmp_path: Path) -> None:
    """
    Must return an empty list if the file exists but is empty.
    """
    empty_md = tmp_path / "empty.md"
    empty_md.touch()

    empty_py = tmp_path / "empty.py"
    empty_py.touch()

    assert chunk_markdown(str(empty_md)) == []
    assert chunk_python(str(empty_py)) == []


def test_chunk_markdown_valid(tmp_path: Path) -> None:
    """
    Test the chunking of a real Markdown file with sections.
    """
    md_file = tmp_path / "doc.md"
    content = (
        "# Main Title\n"
        "This is an introduction.\n\n"
        "## Subtitle\n"
        "Detailed explanations of the system.\n"
    )
    md_file.write_text(content, encoding="utf-8")

    chunks = chunk_markdown(str(md_file), max_chunk_size=500)

    assert len(chunks) > 0, "The chunker should have extracted chunks."
    assert all(isinstance(c, Chunk) for c in chunks), "Must be a Chunk object."

    first_chunk = chunks[0]
    assert first_chunk.first_character_index >= 0
    assert first_chunk.last_character_index <= len(content)

    assert "Main Title" in first_chunk.content


def test_chunk_python_valid(tmp_path: Path) -> None:
    """
    Test AST extraction on valid Python code.
    """
    py_file = tmp_path / "script.py"
    content = (
        "class Agent:\n"
        "    def __init__(self) -> None:\n"
        "        self.name = 'Smith'\n\n"
        "    def act(self) -> None:\n"
        "        pass\n\n"
        "def standalone_func() -> None:\n"
        "    return True\n"
    )
    py_file.write_text(content, encoding="utf-8")

    chunks = chunk_python(str(py_file), max_chunk_size=500)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)

    combined_content = "".join([c.content for c in chunks])
    assert "class Agent:" in combined_content
    assert "def standalone_func() -> None:" in combined_content


def test_chunk_python_syntax_error(tmp_path: Path) -> None:
    """
    Verify that the AST parser catches syntax errors without crashing the app.
    """
    py_file = tmp_path / "bad_syntax.py"
    py_file.write_text(
        "class Def invalid(:\n    print 'error'", encoding="utf-8"
    )

    chunks = chunk_python(str(py_file), max_chunk_size=500)

    assert chunks == [], ("A file with invalid syntax must"
                          " return an empty list."
                          )
