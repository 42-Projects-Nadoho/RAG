import re
import pickle
from typing import List, Optional
from pydantic import BaseModel
from src.indexer import BM25Index
from src.models.minimalSource import MinimalSource
from src.models.utils import TerminalColors


class CodebaseRetriever(BaseModel):
    """
    Retriever that loads a BM25 index and queries it.
    """
    k1: float = 1.5
    b: float = 0.75
    index: Optional[BM25Index] = None

    def load_index(self, index_path: str) -> None:
        """Load the serialized BM25 index from disk."""
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
                self.index = BM25Index(**data)
            TerminalColors.success(
                f"Index successfully loaded from {index_path}"
            )
        except (
                FileNotFoundError,
                EOFError,
                pickle.UnpicklingError) as e:
            raise RuntimeError(
                f"File '{index_path}' not found or corrupted."
            ) from e

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """
        Score all chunks against the query and return the top-k sources.
        """
        if not query or not query.strip():
            TerminalColors.warning(
                "[WARNING] Empty query received. Returning no sources."
            )
            return []
        if k <= 0:
            TerminalColors.warning(
                f"[WARNING] Requested k={k} is invalid. Returning no sources."
            )
            return []
        if self.index is None:
            raise ValueError(
                "Index is not loaded. Please call load_index() first."
            )

        query_tokens = re.findall(r'\w+', query.lower())

        scores = []

        for i, chunk_tf in enumerate(self.index.doc_term_freqs):
            score = 0.0
            doc_len = self.index.doc_lengths[i]

            for token in query_tokens:
                if token not in self.index.df:
                    continue

                tf = chunk_tf.get(token, 0)
                if tf == 0:
                    continue

                idf = self.index.idf[token]
                numerator = tf * (self.k1 + 1)
                doc_base = doc_len / self.index.avg_doc_len
                doc_factor = 1 - self.b + self.b * doc_base
                denominator = tf + self.k1 * doc_factor

                score += idf * (numerator / denominator)

            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        top_sources = []
        for score, idx in scores[:k]:
            chunk = self.index.chunks[idx]
            top_sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index
                )
            )

        return top_sources
