import ast
from typing import List
from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a chunk of text from a file."""

    file_path: str
    content: str
    first_character_index: int
    last_character_index: int


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
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except UnicodeDecodeError:
        print(f"Could not decode file: {file_path}")
        return []
    except OSError as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    except ValueError as e:
        print(e)
        return []

    chunks: List[Chunk] = []
    current_start = 0
    current_text = ""

    for line in content.splitlines(keepends=True):
        if line.startswith("#") and current_text:
            new_chunk = Chunk(
                file_path=file_path,
                content=current_text,
                first_character_index=current_start,
                last_character_index=current_start + len(current_text)
            )
            if len(current_text) > max_chunk_size:
                chunks.extend(_split_large_chunk(new_chunk, max_chunk_size))
            else:
                chunks.append(new_chunk)

            current_start += len(current_text)
            current_text = line
        else:
            current_text += line

    if current_text:
        new_chunk = Chunk(
            file_path=file_path,
            content=current_text,
            first_character_index=current_start,
            last_character_index=current_start + len(current_text)
        )
        if len(current_text) > max_chunk_size:
            chunks.extend(_split_large_chunk(new_chunk, max_chunk_size))
        else:
            chunks.append(new_chunk)

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
    except ValueError as e:
        print(e)
        return []
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except UnicodeDecodeError:
        print(f"Could not decode file: {file_path}")
        return []
    except SyntaxError as e:
        print(f"Syntax error in file {file_path}: {e}")
        return []
    except OSError as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    lines = content.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    chunks: List[Chunk] = []
    for node in tree.body:
        if isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
            )
        ):
            if node.end_lineno is None:
                continue
            start = line_starts[node.lineno - 1]
            end = line_starts[node.end_lineno]
            chunk_content = content[start:end]

            new_chunk = Chunk(
                file_path=file_path,
                content=chunk_content,
                first_character_index=start,
                last_character_index=end
            )
            if len(chunk_content) > max_chunk_size:
                chunks.extend(_split_large_chunk(new_chunk, max_chunk_size))
            else:
                chunks.append(new_chunk)
    return chunks


def _split_large_chunk(chunk: Chunk, max_chunk_size: int) -> List[Chunk]:
    """Split a chunk that exceeds max_chunk_size into smaller pieces.

    Args:
        chunk: The chunk to split.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        List of smaller chunks with overlap.
    """
    chunks: List[Chunk] = []
    lines = chunk.content.splitlines(keepends=True)

    current_text = ""
    current_start = chunk.first_character_index

    for line in lines:
        if len(current_text) + len(line) > max_chunk_size and current_text:
            chunks.append(Chunk(
                file_path=chunk.file_path,
                content=current_text,
                first_character_index=current_start,
                last_character_index=current_start + len(current_text)
            ))
            current_start += len(current_text)
            current_text = line
        else:
            current_text += line

    # On n'oublie pas le dernier morceau
    if current_text:
        chunks.append(Chunk(
            file_path=chunk.file_path,
            content=current_text,
            first_character_index=current_start,
            last_character_index=current_start + len(current_text)
        ))

    return chunks
