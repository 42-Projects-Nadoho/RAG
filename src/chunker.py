import ast
from typing import List
from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a chunk of text from a file with strict character offsets.
    """

    file_path: str
    content: str
    first_character_index: int
    last_character_index: int


def _split_oversized(file_path: str,
                     content: str,
                     start: int,
                     end: int,
                     max_chunk_size: int,
                     ) -> List[Chunk]:
    """
    Broke a too large block to smal-chunks with a slice window
    """
    chunks: List[Chunk] = []
    size = end - start

    if size <= max_chunk_size:
        return [Chunk(
            file_path=file_path,
            content=content[start:end],
            first_character_index=start,
            last_character_index=end
        )]

    overlap = max_chunk_size // 10
    step = max_chunk_size - overlap
    pos = start

    while pos < end:
        sub_end = min(pos + max_chunk_size, end)
        chunks.append(Chunk(
            file_path=file_path,
            content=content[pos:sub_end],
            first_character_index=pos,
            last_character_index=sub_end
        ))
        if sub_end >= end:
            break
        pos += step

    return chunks


def chunk_markdown(file_path: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """
    Chunk a markdown file by sections.

    Args:
        file_path: Path to the markdown file.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        List of chunks with their positions.
    """
    try:
        if not file_path.endswith(".md"):
            raise ValueError("[ERROR] : not the adapted file format")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except (FileNotFoundError, UnicodeDecodeError, OSError, ValueError) as e:
        print(f"Error handling file {file_path}: {e}")
        return []

    if not content:
        return []

    lines = content.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    section_starts: List[int] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            section_starts.append(line_offsets[i])

    if not section_starts or section_starts[0] != 0:
        section_starts.insert(0, 0)

    boundaries = section_starts + [len(content)]
    chunks: List[Chunk] = []

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if content[start:end].strip():
            chunks.extend(
                _split_oversized(
                    file_path,
                    content,
                    start,
                    end,
                    max_chunk_size
                )
            )

    return chunks


def chunk_python(file_path: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Chunk a Python file by functions and classes.

    Args:
        file_path: Path to the Python file.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        List of chunks with their positions.
    """
    try:
        if not file_path.endswith(".py"):
            raise ValueError("[ERROR] : not the adapted file format")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except (
        FileNotFoundError, UnicodeDecodeError, SyntaxError, OSError, ValueError
    ) as e:
        print(f"Error handling file {file_path}: {e}")
        return []

    lines = content.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    top_level = [
        node for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
    ]

    chunks: List[Chunk] = []
    cursor = 0

    for node in top_level:
        if node.end_lineno is None:
            continue

        start = line_starts[node.lineno - 1]
        end = line_starts[node.end_lineno]

        if start > cursor:
            if content[cursor:start].strip():
                chunks.extend(
                    _split_oversized(
                            file_path, content, cursor, start, max_chunk_size
                        )
                    )

        chunks.extend(_split_oversized(
            file_path,
            content, start,
            end, max_chunk_size
            )
        )
        cursor = end

    if cursor < len(content):
        if content[cursor:].strip():
            chunks.extend(_split_oversized(
                    file_path,
                    content,
                    cursor,
                    len(content),
                    max_chunk_size
                )
            )

    if not chunks and content.strip():
        chunks = _split_oversized(
            file_path,
            content,
            0,
            len(content),
            max_chunk_size
        )

    return chunks
