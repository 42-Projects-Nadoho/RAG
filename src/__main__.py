"""
Main entry point for the RAG against the machine project.

This module defines the CLI application using Python Fire, exposing
the required commands for indexing, searching, and generating answers.
"""

import os
import json
import fire
from tqdm import tqdm
from pathlib import Path
from src.generator import RagGenerator
from src.indexer import CodebaseIndexer
from src.retriever import CodebaseRetriever
from src.models.utils import TerminalColors
from src.models.ragDataset import RagDataset
from src.models.minimalSearch import MinimalSearchResults, MinimalAnswer
from src.models.student import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class RagPipeline:
    """
    CLI application for the RAG against the machine project.

    This class exposes methods that are transformed into command-line
    interfaces by the python-fire library. It covers the full pipeline
    from document indexing to answer generation.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build the BM25 index under data/processed/.

        Args:
            max_chunk_size (int, optional): The maximum length of characters
                for each text chunk. Defaults to 2000.
        """
        raw_dir = "data/raw/vllm-0.10.1"
        processed_dir = "data/processed"
        index_path = os.path.join(processed_dir, "bm25_index.pkl")

        TerminalColors.info(
            f"Indexing {raw_dir} with chunk size {max_chunk_size}..."
        )
        indexer = CodebaseIndexer(max_chunk_size=max_chunk_size)
        bm25_index = indexer.build_index(raw_dir)
        indexer.save_index(bm25_index, index_path)
        TerminalColors.success(
            "Ingestion complete! Indices saved under data/processed/"
        )

    def search(self, query: str, k: int) -> None:
        """
        Return the top-k sources for a single query.

        Args:
            query (str): The question to search sources for.
            k (int): The number of top sources to retrieve.
        """
        index_path = "data/processed/bm25_index.pkl"

        TerminalColors.info(
            f"Searching for top {k} sources for query: '{query}'"
        )
        retriever = CodebaseRetriever()
        retriever.load_index(index_path)

        results = retriever.search(query, k)

        output = MinimalSearchResults(
            question_id="test-id",
            question=query,
            question_str=query,
            retrieved_sources=results
        )
        print(output.model_dump_json(indent=2))

    def search_dataset(self,
                       dataset_path: str,
                       k: int,
                       save_directory: str) -> None:
        """
        Run search over a whole dataset and write a StudentSearchResults
        JSON file.

        Args:
            dataset_path (str): Path to the JSON dataset containing questions.
            k (int): Number of sources to retrieve per question.
            save_directory (str): Directory where the output JSON will be
                saved.
        """
        index_path = "data/processed/bm25_index.pkl"

        TerminalColors.info(f"Loading index from {index_path}...")
        retriever = CodebaseRetriever()
        retriever.load_index(index_path)

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
        except Exception as e:
            TerminalColors.error(f"Failed to load dataset: {e}")
            return

        search_results_list = []

        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            sources = retriever.search(item.question, k)
            search_results_list.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    question_str=item.question,
                    retrieved_sources=sources
                )
            )

        final_output = StudentSearchResults(
            search_results=search_results_list,
            k=k
        )

        out_path = Path(save_directory)
        out_path.mkdir(parents=True, exist_ok=True)
        file_name = Path(dataset_path).name
        full_out_path = out_path / file_name

        with open(full_out_path, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))

        TerminalColors.success(
            f"Saved student_search_results to {full_out_path}"
        )

    def answer(self, query: str, k: int = 5) -> None:
        """
        Test the full pipeline (Search + Generation) for a single query.

        Args:
            query (str): The question to answer.
            k (int, optional): Number of sources to retrieve. Defaults to 5.
        """
        index_path = "data/processed/bm25_index.pkl"

        TerminalColors.info(f"Searching top {k} sources...")
        retriever = CodebaseRetriever()
        retriever.load_index(index_path)
        sources = retriever.search(query, k)

        TerminalColors.info("Generating answer...")
        generator = RagGenerator()
        generator.load_model()
        answer_text = generator.answer(query, sources)

        print("\n" + "="*50)
        TerminalColors.info(f"QUESTION: {query}")
        print("="*50)
        TerminalColors.success(f"RÉPONSE GÉNÉRÉE:\n{answer_text}")
        print("="*50)

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Read a StudentSearchResults JSON, generate answers, and save results.

        Outputs a StudentSearchResultsAndAnswer JSON file formatted
            for evaluation.

        Args:
            student_search_results_path (str): Path to the retrieval
                results JSON.
            save_directory (str): Directory where the output JSON
                will be saved.
        """
        try:
            with open(student_search_results_path,
                      "r", encoding="utf-8") as f:
                data = json.load(f)
            search_data = StudentSearchResults(**data)
        except Exception as e:
            TerminalColors.error(f"Failed to load search results: {e}")
            return

        generator = RagGenerator()
        generator.load_model()

        answers_list = []

        for item in tqdm(search_data.search_results,
                         desc="Generating answers"):
            answer_text = generator.answer(
                item.question, item.retrieved_sources
            )

            answers_list.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    question_str=item.question_str,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer_text
                )
            )

        final_answers = StudentSearchResultsAndAnswer(
            k=search_data.k,
            search_results=answers_list
        )

        out_path = Path(save_directory)
        out_path.mkdir(parents=True, exist_ok=True)
        file_name = Path(student_search_results_path).name
        full_out_path = out_path / file_name

        with open(full_out_path, "w", encoding="utf-8") as f:
            f.write(final_answers.model_dump_json(indent=2))

        TerminalColors.success(f"Saved student_answers to {full_out_path}")


def main() -> None:
    """
    Entry point for the fire CLI application.
    """
    fire.Fire(RagPipeline)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        TerminalColors.error("[ERROR] Keybord intrruption was detected")
