"""
Generation module for processing retrieved context into natural language.

This module provides the RAGGenerator class, which loads a local Large
Language Model (LLM) to produce answers grounded solely in provided sources.
It satisfies the 'Answer generation' requirement of the project subject.
"""
import os
import torch
from pathlib import Path
from typing import List, cast, Any, Dict
from pydantic import BaseModel, PrivateAttr
from src.models.utils import TerminalColors
from src.models.minimalSource import MinimalSource
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    max_context_length: int = 2000
    model_config = {"arbitrary_types_allowed": True}
    tokenizer: AutoTokenizer | None = None
    model: AutoModelForCausalLM | None = None

    # Attributs privés : non sérialisés, non traités comme des champs Pydantic
    _file_cache: Dict[str, str] = PrivateAttr(default_factory=dict)
    _project_root: Path = PrivateAttr(default_factory=Path.cwd)

    def load_model(self) -> None:
        """
        Load the LLM and tokenizer into memory.

        Initializes the Hugging Face model and tokenizer using the specified
        model_name. Uses fp16 on GPU (if available) or fp32 on CPU, since
        fp16 offers no speed benefit without hardware acceleration and
        4-bit quantization was found to degrade precision-critical answers
        on this small model.
        """
        # torch.set_num_threads(os.cpu_count() or 1)
        torch.set_num_threads(10)
        TerminalColors.info(f"Loading the LLM : {self.model_name}...")

        self.tokenizer = cast(
            AutoTokenizer,
            AutoTokenizer.from_pretrained(self.model_name)
        )

        use_cuda = torch.cuda.is_available()
        dtype: Any = torch.float16 if use_cuda else torch.float32

        self.model = cast(
            AutoModelForCausalLM,
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                low_cpu_mem_usage=True
            )
        )
        if not use_cuda:
            self.model.to("cpu")

        TerminalColors.success("Modèle chargé avec succès.")

    def _resolve_path(self, file_path: str) -> Path:
        """
        Resolve a source file path relative to the project root.

        Prevents silent read failures when the process is launched from
        a different working directory than the project root.

        Args:
            file_path (str): Relative or absolute path as stored in the
                source metadata.

        Returns:
            Path: An absolute path safe to open regardless of cwd.
        """
        p = Path(file_path)
        if p.is_absolute():
            return p
        return self._project_root / p

    def _read_file_cached(self, file_path: str) -> str:
        """
        Read a file's full content once and cache it in memory.

        Avoids re-reading the same file from disk multiple times when
        several questions reference overlapping sources.

        Args:
            file_path (str): The source file path as stored in metadata.

        Returns:
            str: The file's full content, or an empty string if it
                could not be read.
        """
        if file_path not in self._file_cache:
            resolved_path = self._resolve_path(file_path)
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    self._file_cache[file_path] = f.read()
            except Exception as e:
                TerminalColors.warning(
                    f"Impossible de lire {resolved_path} : {e}"
                )
                self._file_cache[file_path] = ""
        return self._file_cache[file_path]

    def _extract_text_from_source(self, source: MinimalSource) -> str:
        """
        Extract a specific text span from a (cached) source file.

        Args:
            source (MinimalSource): The source object containing the file
            path and the character indices defining the exact span to extract.

        Returns:
            str: The extracted text span, or an empty string if the file
                could not be read.
        """
        content = self._read_file_cached(source.file_path)
        if not content:
            return ""
        return content[
            source.first_character_index: source.last_character_index
        ]

    def build_prompt(self, question: str, sources: List[MinimalSource]) -> str:
        """
        Construct a minimalist prompt adapted for Micro-LLMs (< 1B parameters).
        """
        if not self.tokenizer:
            raise ValueError("Le tokenizer n'est pas chargé.")

        context_chunks = []
        for i, src in enumerate(sources, 1):
            raw_text = self._extract_text_from_source(src).strip()
            if raw_text:
                if len(raw_text) > self.max_context_length:
                    raw_text = raw_text[: self.max_context_length]
                context_chunks.append(f"Document {i}:\n{raw_text}")

        context_text = "\n\n".join(context_chunks)

        # Consigne ultra-basique, les petits modèles ne comprennent pas les règles complexes
        system_content = (
            "You are a strict technical assistant. "
            "Answer the question using ONLY the provided documents. "
            "Your answer MUST be a single, short sentence. "
            "If the answer is not in the documents, reply EXACTLY with: I don't know."
        )

        user_content = f"Documents:\n{context_text}\n\nQuestion: {question}\nShort Answer:"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        return str(prompt)

    def answer(self,
               question: str,
               sources: List[MinimalSource],
               max_tokens: int = 35) -> str:
        """
        Generate a natural language answer with strict length constraints.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and Tokenizer must be loaded before usage.")

        prompt = self.build_prompt(question, sources)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length: int = int(inputs.input_ids.shape[1])
        generated_tokens = outputs[0][input_length:]

        full_answer: str = str(self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )).strip()

        # Post-processing agressif : on ne garde que la première ligne ou première vraie phrase générée
        final_answer = full_answer.split('\n')[0].strip()

        # Si le modèle essaie de dire qu'il ne sait pas avec d'autres mots, on normalise
        if not final_answer or "don't know" in final_answer.lower() or "do not know" in final_answer.lower():
            return "I don't know."

        return final_answer