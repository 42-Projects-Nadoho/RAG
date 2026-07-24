"""
Evaluation module for computing search metrics against ground truth.
"""

import json
from src.models.ragDataset import RagDataset
from src.models.student import StudentSearchResults
from src.models.utils import TerminalColors


class RagEvaluator:
    """Evaluates student search results against ground-truth datasets."""

    @staticmethod
    def _calculate_iou(
        start1: int, end1: int, start2: int, end2: int
    ) -> float:
        """
        Calculate the Intersection over Union (IoU) of two character spans.
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
        """Report recall@k against a ground-truth dataset."""
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
