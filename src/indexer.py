import re
import math
import pickle
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel
from src.chunker import chunk_markdown, chunk_python, Chunk


class BM25Index(BaseModel):
    """
    Persistence model containing all the data structures for the BM25 index.
    """

    chunks: List[Chunk]
    doc_lengths: List[int]
    avg_doc_len: float
    df: Dict[str, int]
    idf: Dict[str, float]
    doc_term_freqs: List[Dict[str, int]]


class CodebaseIndexer(BaseModel):
    """Pydantic-based indexer to ingest the codebase and build a BM25 index."""

    max_chunk_size: int = 2000
    k1: float = 1.5
    b: float = 0.75

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text for lexical analysis using basic splitting.

        Args:
            text: The raw string content to process.

        Returns:
            A list of lowercased tokens.
        """
        return re.findall(r'\w+', text.lower())

    def build_index(self, raw_data_dir: str) -> BM25Index:
        """Scan the raw directory, chunk discovered files,
        and compute BM25 statistics.

        Args:
            raw_data_dir: Path to the raw source code directory.

        Returns:
            A populated BM25Index instance.
        """
        all_chunks: List[Chunk] = []
        raw_path = Path(raw_data_dir)

        if not raw_path.exists():
            print(f"Error: The directory {raw_data_dir} does not exist.")
            return BM25Index(
                chunks=[], doc_lengths=[], avg_doc_len=0.0,
                df={}, idf={}, doc_term_freqs=[]
            )

        file_list = list(raw_path.rglob("*"))
        print(f"Ingesting files from {raw_data_dir}...")

        for p in file_list:
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

        for chunk in all_chunks:
            tokens = self._tokenize(chunk.content)
            file_stem = Path(chunk.file_path).stem.replace('_', ' ').replace('-', ' ')
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

        Args:
            index: The BM25Index instance to serialize.
            output_path: Target path to save the binary payload.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            pickle.dump(index.dict(), f)
        print(f"Index successfully saved to: {output_path}")
