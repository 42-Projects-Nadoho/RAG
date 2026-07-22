"""
Pipeline module for orchestrating the RAG against the machine processes.
"""

import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import List
from pydantic import ValidationError
from src.generator import RagGenerator
from src.indexer import CodebaseIndexer
from src.retriever import CodebaseRetriever
from src.evaluation import RagEvaluator
from src.models.utils import TerminalColors
from src.models.ragDataset import RagDataset
from src.models.minimalSearch import (
    MinimalAnswer,
    MinimalSearchResults
)
from src.models.student import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class RagPipeline:
    """
    Core orchestration class for the RAG against the machine project.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest raw data and build the BM25 index."""
        raw_dir = "data/raw/vllm-0.10.1"
        processed_dir = "data/processed"
        index_path = os.path.join(processed_dir, "bm25_index.pkl")

        TerminalColors.info(
            f"Indexing {raw_dir} with chunk size {max_chunk_size}..."
        )

        try:
            indexer = CodebaseIndexer(max_chunk_size=max_chunk_size)
            bm25_index = indexer.build_index(raw_dir)
        except Exception as e:
            TerminalColors.error(
                f"A critical error occurred during indexing: {e}"
            )
            return

        try:
            indexer.save_index(bm25_index, index_path)
            TerminalColors.success(
                "Ingestion complete! Indices saved under data/processed/"
            )
        except PermissionError:
            TerminalColors.error(
                f"Permission denied: Cannot write to '{index_path}'."
            )
            TerminalColors.warning(
                "Tip: Check folder permissions (chown/chmod)."
            )
        except OSError as e:
            TerminalColors.error(
                f"System error while saving the index: {e}"
            )

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        index_path = "data/processed/bm25_index.pkl"
        TerminalColors.info(
            f"Searching for top {k} sources for query: '{query}'"
        )
        retriever = CodebaseRetriever()

        try:
            retriever.load_index(index_path)
        except Exception as e:
            TerminalColors.error(f"Failed to load index: {e}")
            TerminalColors.warning(
                "Tip: Run 'uv run python -m src index' "
                "to build the index first."
            )
            return

        try:
            results = retriever.search(query, k)
        except ValueError as e:
            TerminalColors.error(f"Search failed: {e}")
            return

        for src in results:
            print(
                f"{src.file_path} ["
                f"{src.first_character_index}: {src.last_character_index}]"
            )

    def search_dataset(self,
                       dataset_path: str,
                       k: int,
                       save_directory: str) -> None:
        """Run search over a dataset and write a StudentSearchResults JSON."""
        index_path = "data/processed/bm25_index.pkl"
        TerminalColors.info(f"Loading index from {index_path}...")
        retriever = CodebaseRetriever()

        try:
            retriever.load_index(index_path)
        except Exception as e:
            TerminalColors.error(f"Failed to load index: {e}")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
        except FileNotFoundError:
            TerminalColors.error(f"Dataset file not found: '{dataset_path}'")
            return
        except ValidationError as e:
            TerminalColors.error("Invalid dataset format.")
            bad_indices = set()
            for error in e.errors():
                if len(error['loc']) >= 2:
                    bad_indices.add(error['loc'][1])

            for idx in sorted(bad_indices):
                TerminalColors.warning(
                    f"  - Malformed item at [rag_questions -> index {idx}]. "
                    "Check required fields."
                )
            return
        except json.JSONDecodeError:
            TerminalColors.error(f"File '{dataset_path}' is not a valid JSON.")
            return
        except Exception as e:
            TerminalColors.error(f"Failed to load dataset: {e}")
            return

        search_results_list: List[MinimalSearchResults] = []
        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            try:
                sources = retriever.search(item.question, k)
            except ValueError as e:
                TerminalColors.error(f"\nSearch operation aborted: {e}")
                TerminalColors.warning(
                    "Tip: Run 'uv run python -m src index' "
                    "to build the index first."
                )
                return

            search_results_list.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=sources
                )
            )

        final_output = StudentSearchResults(
            search_results=search_results_list, k=k
        )

        try:
            out_path = Path(save_directory)
            out_path.mkdir(parents=True, exist_ok=True)
            full_out_path = out_path / Path(dataset_path).name

            with open(full_out_path, "w", encoding="utf-8") as f:
                f.write(final_output.model_dump_json(indent=2))
            TerminalColors.success(
                f"Saved student_search_results to {full_out_path}"
            )
        except PermissionError:
            TerminalColors.error(
                f"Permission denied: Cannot save results to '{full_out_path}'."
            )
        except OSError as e:
            TerminalColors.error(f"System error while saving results: {e}")

    def answer(self, query: str, k: int = 5) -> None:
        """Test the full pipeline for a single query."""
        index_path = "data/processed/bm25_index.pkl"
        retriever = CodebaseRetriever()

        try:
            retriever.load_index(index_path)
        except Exception as e:
            TerminalColors.error(f"Failed to load index: {e}")
            TerminalColors.warning(
                "Tip: Run 'uv run python -m src index' "
                "to build the index first."
            )
            return

        try:
            sources = retriever.search(query, k)
        except ValueError as e:
            TerminalColors.error(f"Search failed: {e}")
            return

        try:
            generator = RagGenerator()
            generator.load_model()
            answer_text = generator.answer(query, sources)
        except Exception as e:
            TerminalColors.error(f"Model generation failed: {e}")
            return

        print("\n" + "="*50)
        TerminalColors.info(f"QUESTION: {query}")
        print("="*50)
        TerminalColors.success(f"RÉPONSE GÉNÉRÉE:\n{answer_text}")
        print("="*50)

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers from a StudentSearchResults JSON."""
        try:
            with open(student_search_results_path,
                      "r", encoding="utf-8") as f:
                data = json.load(f)
            search_data = StudentSearchResults(**data)
        except FileNotFoundError:
            TerminalColors.error(
                f"Results file not found: '{student_search_results_path}'"
            )
            return
        except ValidationError as e:
            TerminalColors.error("Invalid search results format.")
            bad_indices = set()
            for error in e.errors():
                if len(error['loc']) >= 2:
                    bad_indices.add(error['loc'][1])

            for idx in sorted(bad_indices):
                TerminalColors.warning(
                    f"  - Malformed item at [rag_questions -> index {idx}]. "
                    "Check required fields."
                )
            return
        except json.JSONDecodeError:
            TerminalColors.error(
                f"File '{student_search_results_path}' is not a valid JSON."
            )
            return
        except Exception as e:
            TerminalColors.error(f"Failed to load search results: {e}")
            return

        try:
            generator = RagGenerator()
            generator.load_model()
        except Exception as e:
            TerminalColors.error(f"Failed to load the generation model: {e}")
            return

        answers_list: List[MinimalAnswer] = []

        for item in tqdm(search_data.search_results,
                         desc="Generating answers"):
            try:
                answer_text = generator.answer(
                    item.question, item.retrieved_sources
                )
            except Exception as e:
                TerminalColors.error(
                    "\nFailed to generate answer for query "
                    f"'{item.question_id}': {e}"
                )
                answer_text = "Generation error."

            answers_list.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer_text
                )
            )

        final_answers = StudentSearchResultsAndAnswer(
            k=search_data.k,
            search_results=answers_list
        )

        try:
            out_path = Path(save_directory)
            out_path.mkdir(parents=True, exist_ok=True)
            full_out_path = out_path / Path(student_search_results_path).name

            with open(full_out_path, "w", encoding="utf-8") as f:
                f.write(final_answers.model_dump_json(indent=2))
            TerminalColors.success(f"Saved student_answers to {full_out_path}")
        except PermissionError:
            TerminalColors.error(
                f"Permission denied: Cannot save answers to '{full_out_path}'."
            )
        except OSError as e:
            TerminalColors.error(f"System error while saving answers: {e}")

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        """Delegate evaluation to the RagEvaluator module."""
        try:
            RagEvaluator.evaluate(student_search_results_path, dataset_path)
        except Exception as e:
            TerminalColors.error(f"Evaluation process failed: {e}")
