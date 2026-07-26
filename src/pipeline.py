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


DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
STUDENT_SEARCH_RESULT = "data/output/search_results/dataset_docs_public.json"
SAVE_DIRECTORY = "data/output/search_results_and_answer"
ANSWER_PATH = "data/datasets/AnsweredQuestions/dataset_docs_public.json"


class RagPipeline:
    """
    Core orchestration class for the RAG against the machine project.

    This class acts as the main entry point for the Command Line Interface
        (CLI).
    It connects the distinct modules of the pipeline (Indexing, Retrieval,
    Generation, and Evaluation), handles file I/O, catches critical
        exceptions,
    and validates incoming/outgoing data using strict Pydantic schemas.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest raw data and build the BM25 index.

        Initializes the CodebaseIndexer to parse the raw vLLM source code,
        chunks the files according to the specified size constraint, computes
        the BM25 statistics, and serializes the result to a pickle file.

        Args:
            max_chunk_size (int, optional): The maximum allowed length
                (in characters) for each extracted chunk. Defaults to 2000.
        """

        if max_chunk_size > 2000 or max_chunk_size <= 0:
            TerminalColors.error(f"Invalid chunk_size : {max_chunk_size}")
            return

        raw_dir = "data/raw/vllm-0.10.1"
        processed_dir = "data/processed"
        index_path = os.path.join(processed_dir, "bm25_index.pkl")

        TerminalColors.info(
            f"Indexing {raw_dir} with chunk size {max_chunk_size}..."
        )

        try:
            indexer = CodebaseIndexer(max_chunk_size=max_chunk_size)
            bm25_index = indexer.build_index(raw_dir)

            if bm25_index is None:
                TerminalColors.error("Index building failed (returned None).")
                return
        except Exception as e:
            TerminalColors.error(
                f"A critical error occurred during indexing: {e}"
            )
            return

        try:
            indexer.save_index(bm25_index, index_path)
            TerminalColors.success(
                f"Ingestion complete! Indexed {len(bm25_index.chunks)} "
                f"chunks under {processed_dir}/"
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
        """
        Return and print the top-k sources for a single query.

        Loads the pre-computed BM25 index from disk and queries it. This
        method is primarily used for quick debugging and testing the retrieval
        accuracy in the terminal without needing a full dataset.

        Args:
            query (str): The search query provided by the user.
            k (int): The number of top-scoring sources to retrieve.
        """
        index_path = "data/processed/bm25_index.pkl"
        if not query.strip():
            TerminalColors.warning("The search query cannot be empty.")
            return

        if k <= 0:
            TerminalColors.error(f"Value of k : {k} is invalid")
            return

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
                       dataset_path: str = DATASET_PATH,
                       k: int = 5,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """
        Run search over a dataset and write a StudentSearchResults JSON.

        Loads a raw dataset, iterates over all questions, retrieves the top-k
        sources for each, and exports the results to a cleanly formatted JSON
        file that complies with the moulinette's expected schema.

        Args:
            dataset_path (str): Path to the input JSON dataset
                (e.g., UnansweredQuestions).
            k (int): The number of top sources to retrieve per question.
            save_directory (str): The output directory where the results
                JSON will be saved.
        """
        if k <= 0:
            TerminalColors.error(f"Value of k : {k} is invalid")
            return

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
        """
        Test the full end-to-end pipeline for a single query.

        Performs both the retrieval phase (finding top-k sources) and the
        generation phase (prompting the LLM) for a single input query,
        printing the final generated answer directly to the terminal.

        Args:
            query (str): The user's question to answer.
            k (int, optional): The number of sources to retrieve.
                Defaults to 5.
        """
        index_path = "data/processed/bm25_index.pkl"
        retriever = CodebaseRetriever()

        if not query.strip():
            TerminalColors.warning("The question query cannot be empty.")
            return

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
        TerminalColors.success(f"SUCESSFUL GENERATION:\n{answer_text}")
        print("="*50)

    def answer_dataset(
            self,
            student_search_results_path: str = STUDENT_SEARCH_RESULT,
            save_directory: str = SAVE_DIRECTORY) -> None:
        """
        Generate answers from a StudentSearchResults JSON.

        Takes the output file produced by `search_dataset`, loads the local
        LLM, and sequentially generates a short answer for every question
        based on its previously retrieved sources. Saves the final output
        as a new JSON file.

        Args:
            student_search_results_path (str): Path to the intermediate JSON
                file containing the questions and their retrieved sources.
            save_directory (str): The output directory where the final JSON
                with answers will be saved.
        """
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

        answer_texts = []
        for item in tqdm(
                search_data.search_results,
                desc="Génération des réponses", unit="q"):
            try:

                answer_text = generator.answer(
                    item.question,
                    item.retrieved_sources
                )
                answer_texts.append(answer_text)
            except Exception as item_error:
                TerminalColors.error(
                    "\nFailed to generate answer for query "
                    f"'{item.question_id}': {item_error}"
                )
                answer_texts.append("Generation error.")

        answers_list: List[MinimalAnswer] = [
            MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                answer=answer_text
            )
            for item, answer_text in zip(
                search_data.search_results, answer_texts)
        ]

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
                 student_search_results_path: str = STUDENT_SEARCH_RESULT,
                 dataset_path: str = ANSWER_PATH) -> None:
        """
        Delegate evaluation to the RagEvaluator module.

        Compares the student's retrieved sources against a ground-truth
        dataset to compute and print the Recall@K metric.

        Args:
            student_search_results_path (str): Path to the student's search
                results JSON.
            dataset_path (str): Path to the ground-truth JSON dataset
                (AnsweredQuestions).
        """
        try:
            RagEvaluator.evaluate(student_search_results_path, dataset_path)
        except Exception as e:
            TerminalColors.error(f"Evaluation process failed: {e}")
