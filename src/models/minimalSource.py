from pydantic import BaseModel


class MinimalSource(BaseModel):
    """Represent a character-span source extracted from a file.

    Attributes:
        file_path: Absolute or relative path to the source file.
        first_character_index: Inclusive start index of the extracted span.
        last_character_index: Exclusive end index of the extracted span.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
