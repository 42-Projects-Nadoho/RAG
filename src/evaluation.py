"""
Evaluation module for computing search metrics against ground truth.
"""

import json
from src.models.ragDataset import RagDataset
from src.models.student import StudentSearchResults
from src.models.utils import TerminalColors


class RagEvaluator:
    """
    Utility class for evaluating the RAG pipeline's retrieval performance.

    This class compares the sources retrieved by the search pipeline against
    a ground-truth dataset. It calculates the Recall@K metric by measuring
    the Intersection over Union (IoU) of the character spans, determining
    if a retrieved source correctly matches an expected source.
    """

    @staticmethod
    def _calculate_iou(
        start1: int, end1: int, start2: int, end2: int
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) of two character spans.

        This metric determines how much two text segments overlap relative to
        their combined total length. It is used to verify if a retrieved
        chunk accurately covers the expected ground-truth context.

        Args:
            start1 (int): The starting character index of the first span.
            end1 (int): The ending character index of the first span.
            start2 (int): The starting character index of the second span.
            end2 (int): The ending character index of the second span.

        Returns:
            float: The IoU ratio, ranging from 0.0 (no overlap) to 1.0
                (exact match).
        """
        intersection = max(0, min(end1, end2) - max(start1, start2))
        length1 = end1 - start1
        length2 = end2 - start2
        union = length1 + length2 - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def evaluate(cls,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Report Recall@K against a ground-truth dataset.

        Loads the student's search results and the expected ground-truth
        answers, calculates the hit rate based on an IoU threshold (>= 0.05),
        and prints the final average recall score to the terminal.

        Args:
            student_search_results_path (str): Path to the JSON file containing
                the retrieved sources to evaluate.
            dataset_path (str): Path to the ground-truth JSON dataset.
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
            q.question_id: getattr(q, "sources", [])
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
                        iou = cls._calculate_iou(
                            retrieved_src.first_character_index,
                            retrieved_src.last_character_index,
                            expected_src.first_character_index,
                            expected_src.last_character_index,
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

        print("\n" + "=" * 40)
        TerminalColors.success("EVALUATION RESULTS")
        print("=" * 40)
        TerminalColors.info(f"Questions evaluated : {valid_questions}")
        TerminalColors.info(f"Recall@{k} : {average_recall:.3f}")
        print("=" * 40 + "\n")
