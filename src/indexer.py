import re
import math
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel
from src.chunker import chunk_markdown, chunk_python, Chunk


class BM25Index(BaseModel):
    """
    Persistence model containing all the data structures for the BM25 index.

    This Pydantic model encapsulates the pre-computed Okapi BM25 statistics
    and the chunk metadata. It serves as the single source of truth for
    the retrieval phase and is easily serializable for disk storage.

    Attributes:
        chunks (List[Chunk]): The list of all chunked text segments extracted
            from the codebase.
        doc_lengths (List[int]): The token count for each document (chunk),
            indexed in the exact same order as `chunks`.
        avg_doc_len (float): The average document length across the entire
            corpus.
        df (Dict[str, int]): The Document Frequency (number of documents
            containing a specific term) for every unique token in the corpus.
        idf (Dict[str, float]): The pre-calculated Inverse Document Frequency
            weight for every unique token.
        doc_term_freqs (List[Dict[str, int]]): The Term Frequency mapping for
            each individual document/chunk.
    """

    chunks: List[Chunk]
    doc_lengths: List[int]
    avg_doc_len: float
    df: Dict[str, int]
    idf: Dict[str, float]
    doc_term_freqs: List[Dict[str, int]]


class CodebaseIndexer(BaseModel):
    """
    Pydantic-based indexer to ingest the codebase and build a BM25 index.

    Orchestrates the ingestion pipeline by walking the target directory,
    routing files to the appropriate chunking strategies based on their
    extensions, and computing the Okapi BM25 lexical statistics for the
    resulting corpus.

    Attributes:
        max_chunk_size (int): The hard limit for chunk length in characters.
        k1 (float): The BM25 term frequency saturation parameter.
            Controls how quickly the score plateaus for repeated terms.
        b (float): The BM25 document length normalization parameter.
            Controls how much long documents are penalized compared to
            short ones.
    """
    max_chunk_size: int = 2000
    k1: float = 1.5
    b: float = 0.75

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text for lexical analysis using basic word splitting.

        Strips punctuation and splits the text into lowercase alphanumeric
        tokens. This normalization ensures consistent term matching between
        the indexed codebase and the user's queries.

        Args:
            text (str): The raw string content to process.

        Returns:
            List[str]: A list of lowercased alphanumeric tokens.
        """
        return re.findall(r'\w+', text.lower())

    def build_index(self, raw_data_dir: str) -> BM25Index | None:
        """
        Scan the raw directory, chunk discovered files, and compute BM25
        statistics.

        Recursively walks the provided directory, applies AST-based chunking
        for Python files and section-based chunking for Markdown,
        then tokenizes the chunks. Artificially augments each chunk's token
        list with heavily weighted filename tokens to improve retrieval
        accuracy for identifier-based queries.

        Args:
            raw_data_dir (str): Path to the root of the raw source code
                directory.

        Returns:
            BM25Index | None: A fully populated BM25Index instance containing
                all chunks and computed metrics.

        Raises:
            OSError: If the specified raw data directory does not exist.
        """
        all_chunks: List[Chunk] = []
        raw_path = Path(raw_data_dir)

        if not raw_path.exists():
            raise OSError(f"The directory {raw_path}does not exist.")

        file_list = list(raw_path.rglob("*"))

        for p in tqdm(file_list, desc="Chunking", unit="file"):
            if p.is_file():
                if p.suffix == ".py":
                    all_chunks.extend(
                        chunk_python(str(p), self.max_chunk_size)
                    )
                elif p.suffix == ".md":
                    all_chunks.extend(
                        chunk_markdown(str(p), self.max_chunk_size)
                    )

        doc_term_freqs: List[Dict[str, int]] = []
        doc_lengths: List[int] = []
        df: Dict[str, int] = {}
        total_docs = len(all_chunks)

        for chunk in tqdm(all_chunks, desc="Tokenizing", unit="chunk"):
            tokens = self._tokenize(chunk.content)
            file_stem = Path(chunk.file_path).stem.replace('_', ' ')
            file_stem = file_stem.replace('-', ' ')
            filename_tokens = self._tokenize(file_stem)
            tokens.extend(filename_tokens * 5)
            doc_lengths.append(len(tokens))

            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            doc_term_freqs.append(tf)

            for token in tf.keys():
                df[token] = df.get(token, 0) + 1

        avg_doc_len = sum(doc_lengths) / total_docs if total_docs > 0 else 0.0

        idf: Dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log(
                1 + (total_docs - freq + 0.5) / (freq + 0.5)
            )

        return BM25Index(
            chunks=all_chunks,
            doc_lengths=doc_lengths,
            avg_doc_len=avg_doc_len,
            df=df,
            idf=idf,
            doc_term_freqs=doc_term_freqs
        )

    def save_index(self, index: BM25Index, output_path: str) -> None:
        """
        Serialize the complete index structure to a binary file using pickle.

        Uses Python's pickle module for ultra-fast deserialization during the
        search phase, converting the Pydantic model to a standard dictionary
        before dumping to minimize load overhead.

        Args:
            index (BM25Index): The populated index instance to serialize.
            output_path (str): Target path (including filename) to save the
                binary payload.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            pickle.dump(index.dict(), f)
