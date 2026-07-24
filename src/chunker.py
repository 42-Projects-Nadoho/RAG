import ast
from typing import List
from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a contiguous segment of text extracted from a source file.

    This model maintains strict character-level positional tracking to allow
    precise mapping of the extracted content back to the original document.

    Attributes:
        file_path (str): The relative or absolute path to the original
            source file.
        content (str): The actual text content contained within this chunk.
        first_character_index (int): The starting character offset of the chunk
            in the original file (inclusive).
        last_character_index (int): The ending character offset of the chunk
            in the original file (exclusive).
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
    pos = start

    while pos < end:
        sub_end = min(pos + max_chunk_size, end)

        if sub_end < end:
            last_newline = content.rfind('\n', pos, sub_end)
            last_space = content.rfind(' ', pos, sub_end)

            break_point = last_newline if last_newline != -1 else last_space

            if break_point > pos + (max_chunk_size // 2):
                sub_end = break_point

        chunks.append(Chunk(
            file_path=file_path,
            content=content[pos:sub_end],
            first_character_index=pos,
            last_character_index=sub_end
        ))
        if sub_end >= end:
            break

        pos = sub_end - overlap

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

    target_nodes: List[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            target_nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.end_lineno is None:
                continue

            class_start = line_starts[node.lineno - 1]
            class_end = line_starts[node.end_lineno]

            if (class_end - class_start) <= max_chunk_size:
                target_nodes.append(node)
            else:
                for sub_node in node.body:
                    if isinstance(
                            sub_node, (
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                                ast.ClassDef
                                )
                            ):
                        target_nodes.append(sub_node)

    target_nodes.sort(key=lambda x: x.lineno)
    chunks: List[Chunk] = []
    cursor = 0

    for node in target_nodes:
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
