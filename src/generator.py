"""
Generation module for processing retrieved context into natural language.

This module provides the RAGGenerator class, which loads a local Large
Language Model (LLM) to produce answers grounded solely in provided sources.
It satisfies the 'Answer generation' requirement of the project subject.
"""

import torch
from typing import List, cast, Any
from pydantic import BaseModel
from src.models.utils import TerminalColors
from src.models.minimalSource import MinimalSource
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


class RagGenerator(BaseModel):
    """
    Local LLM manager for context-grounded answer generation.

    Requires the model and tokenizer to be loaded into memory before
    generating answers to avoid initialization overhead on every call.

    Attributes:
        model_name (str): The Hugging Face hub identifier for the model
            (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
        model_config (dict): Pydantic configuration allowing arbitrary
            types for PyTorch/Transformers objects.
        tokenizer (AutoTokenizer | None): The loaded tokenizer instance.
        model (AutoModelForCausalLM | None): The loaded model instance
            mapped to the available device.
    """
    model_name: str = "Qwen/Qwen3-0.6B"
    model_config = {"arbitrary_types_allowed": True}
    tokenizer: AutoTokenizer | None = None
    model: AutoModelForCausalLM | None = None

    @staticmethod
    def create_bnb_config() -> Any:
        bnb_config_class: Any = BitsAndBytesConfig
        return bnb_config_class(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )

    def load_model(self) -> None:
        """
        Load the LLM and tokenizer into memory.

        Initializes the Hugging Face model and tokenizer using the specified
        model_name. It leverages automatic device mapping and FP16 precision
        to optimize VRAM usage.
        """
        TerminalColors.info(f"Loading the LLM : {self.model_name}...")

        self.tokenizer = cast(
            AutoTokenizer,
            AutoTokenizer.from_pretrained(self.model_name)
        )
        quantization_config = self.create_bnb_config()
        self.model = cast(
            AutoModelForCausalLM,
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
        )

        TerminalColors.success("Modèle chargé avec succès.")

    def _extract_text_from_source(self, source: MinimalSource) -> str:
        """
        Read and extract a specific text span from a local file.

        Args:
            source (MinimalSource): The source object containing the file
            path and the character indices defining the exact span to extract.

        Returns:
            str: The extracted text span, or an empty string if the file
                cannot be read.
        """
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content[
                    source.first_character_index: source.last_character_index
                ]
        except Exception as e:
            TerminalColors.warning(
                f"Impossible de lire {source.file_path} : {e}"
            )
            return ""

    def build_prompt(self,
                     question: str,
                     sources: List[MinimalSource]) -> str:
        """
        Construct a strict, chat-formatted prompt for the LLM.

        Combines the user question with the extracted context chunks and wraps
        them in the model's specific chat template, enforcing a strict
        technical assistant persona.

        Args:
            question (str): The raw question to be answered.
            sources (List[MinimalSource]): The list of retrieved sources
                providing the context for the answer.

        Returns:
            str: The fully formatted prompt ready for tokenization.

        Raises:
            ValueError: If the tokenizer has not been loaded before calling.
        """
        if not self.tokenizer:
            raise ValueError("Le tokenizer n'est pas chargé.")

        context_chunks = [
            self._extract_text_from_source(src) for src in sources
        ]
        context_text = "\n---\n".join([c for c in context_chunks if c])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict technical assistant. Answer the "
                    "question using ONLY the provided context. "
                    "Be extremely concise (1 or 2 sentences maximum). "
                    "If the context does not contain the answer, "
                    "reply EXACTLY with 'I don't know'."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}"
            }
        ]

        tokenizer = cast(Any, self.tokenizer)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt += """
<think>\nContext analyzed. Generating final concise answer.\n</think>\n\n
        """

        return str(prompt)

    def answer(self,
               question: str,
               sources: List[MinimalSource],
               max_tokens: int = 512) -> str:
        """
        Generate a natural language answer based on the provided sources.

        Uses the loaded LLM to generate an answer. The generation is
        constrained by greedy decoding (do_sample=False) to prevent
        hallucinations and ensure grounding in the provided context.

        Args:
            question (str): The raw question to answer.
            sources (List[MinimalSource]): The list of retrieved sources.
            max_tokens (int, optional): The maximum number of new tokens
                to generate. Defaults to 512.

        Returns:
            str: The generated text answer.

        Raises:
            ValueError: If the model or tokenizer has not been loaded before
                calling.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError(
                "Model and Tokenizer must be loaded before usage."
            )

        tokenizer = cast(Any, self.tokenizer)
        model = cast(Any, self.model)
        prompt = self.build_prompt(question, sources)

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False
            )

        input_length: int = int(inputs.input_ids.shape[1])
        generated_tokens = outputs[0][input_length:]

        answer_text: str = str(tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ))
        import re
        answer_text = re.sub(
            r'<think>.*?</think>', '', answer_text, flags=re.DOTALL
        ).strip()
        answer_text = answer_text.replace(
            '<think>',
            ''
        ).replace('</think>', '').strip()

        if not answer_text:
            return "I don't know."

        return answer_text
