"""
Main entry point for the RAG against the machine project.

This module defines the CLI application using Python Fire, exposing
the required commands for indexing, searching, evaluating,
and generating answers.
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
    from document indexing to answer generation and evaluation.
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
            f"[INFO] Indexing {raw_dir} with chunk size {max_chunk_size}..."
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

        try:
            retriever.load_index(index_path)
        except Exception as e:
            TerminalColors.error(f"[ERROR] Failed to load index: {e}")
            return

        results = retriever.search(query, k)

        output = MinimalSearchResults(
            question_id="test-id",
            question=query,
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
        """
        index_path = "data/processed/bm25_index.pkl"

        TerminalColors.info(f"Loading index from {index_path}...")
        retriever = CodebaseRetriever()

        try:
            retriever.load_index(index_path)
        except Exception as e:
            TerminalColors.error(f"[ERROR] Failed to load index: {e}")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
        except Exception as e:
            TerminalColors.error(f"[ERROR] Failed to load dataset: {e}")
            return

        search_results_list = []

        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            sources = retriever.search(item.question, k)
            search_results_list.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
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
        """
        index_path = "data/processed/bm25_index.pkl"

        TerminalColors.info(f"Searching top {k} sources...")
        retriever = CodebaseRetriever()

        try:
            retriever.load_index(index_path)
        except Exception as e:
            TerminalColors.error(f"[ERROR] Failed to load index: {e}")
            return

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

    def _calculate_iou(self,
                       start1: int,
                       end1: int,
                       start2: int,
                       end2: int) -> float:
        """
        Calculate the Intersection over Union (IoU) of two character spans.
        """
        intersection = max(0, min(end1, end2) - max(start1, start2))
        length1 = end1 - start1
        length2 = end2 - start2
        union = length1 + length2 - intersection
        return intersection / union if union > 0 else 0.0

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Report recall@k against a ground-truth dataset, using the k
        stored in the student results file.
        """
        TerminalColors.info(
            "Evaluating search results against ground truth..."
            )

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_data = json.load(f)
            student_results = StudentSearchResults(**student_data)

            k = student_results.k
            TerminalColors.info(
                f"Using k={k} from student results for evaluation."
            )

            with open(dataset_path, "r", encoding="utf-8") as f:
                truth_data = json.load(f)
            truth_dataset = RagDataset(**truth_data)
        except Exception as e:
            TerminalColors.error(f"Failed to load files for evaluation: {e}")
            return

        truth_dict = {
            q.question_id: getattr(q, 'sources', [])
            for q in truth_dataset.rag_questions
        }

        total_recall = 0.0
        valid_questions = 0

        for student_q in student_results.search_results:
            q_id = student_q.question_id
            if q_id not in truth_dict or not truth_dict[q_id]:
                continue

            true_sources = truth_dict[q_id]
            hits = 0
            top_k_retrieved = student_q.retrieved_sources[:k]

            for expected_src in true_sources:
                found = False
                for retrieved_src in top_k_retrieved:
                    if retrieved_src.file_path == expected_src.file_path:
                        iou = self._calculate_iou(
                            retrieved_src.first_character_index,
                            retrieved_src.last_character_index,
                            expected_src.first_character_index,
                            expected_src.last_character_index
                        )
                        if iou >= 0.05:
                            found = True
                            break
                if found:
                    hits += 1

            if len(true_sources) > 0:
                question_recall = hits / len(true_sources)
                total_recall += question_recall
                valid_questions += 1

        if valid_questions == 0:
            TerminalColors.warning(
                "No matching questions found between results and ground truth."
            )
            return

        average_recall = total_recall / valid_questions

        print("\n" + "="*40)
        TerminalColors.success("EVALUATION RESULTS")
        print("="*40)
        TerminalColors.info(f"Questions evaluated : {valid_questions}")
        TerminalColors.info(f"Recall@{k} : {average_recall:.3f}")
        print("="*40 + "\n")


def main() -> None:
    """
    Entry point for the fire CLI application.
    """
    fire.Fire(RagPipeline)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        TerminalColors.error(
            "\n[ERROR] Keyboard interruption was detected."
        )
