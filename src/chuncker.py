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
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
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

    chunks = []
    current_start = 0
    current_text = ""

    for line in content.splitlines(keepends=True):
        if line.startswith("#") and current_text:
            chunks.append(Chunk(
                file_path=file_path,
                content=current_text,
                first_character_index=current_start,
                last_character_index=current_start + len(current_text)
            ))
            current_start += len(current_text)
            current_text = line
        else:
            current_text += line

    if current_text:
        chunks.append(Chunk(
            file_path=file_path,
            content=current_text,
            first_character_index=current_start,
            last_character_index=current_start + len(current_text)
        ))

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

    # calcul des positions de début de chaque ligne
    lines = content.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
            )
        ):
            if node.end_lineno is None:
                continue
            start = line_starts[node.lineno - 1]
            end = line_starts[node.end_lineno]
            chunk_content = content[start:end]
            chunks.append(Chunk(
                file_path=file_path,
                content=chunk_content,
                first_character_index=start,
                last_character_index=end
            ))

    return chunks
