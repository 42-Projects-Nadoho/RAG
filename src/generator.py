"""
Generation module for processing retrieved context into natural language.

This module provides the RAGGenerator class, which loads a local Large
Language Model (LLM) to produce answers grounded solely in provided sources.
It satisfies the 'Answer generation' requirement of the project subject.
"""
import torch
from pathlib import Path
from typing import List, Any, Dict
from pydantic import BaseModel, PrivateAttr
from src.models.utils import TerminalColors
from src.models.minimalSource import MinimalSource
from transformers import AutoModelForCausalLM, AutoTokenizer


class RagGenerator(BaseModel):
    """
    Local LLM manager for context-grounded answer generation.

    This class orchestrates the loading of a small causal language model (LLM)
    and handles the prompt construction and inference steps.
    It uses an internal file cache to avoid redundant disk I/O
    when multiple sources reference the same file.

    Attributes:
        model_name (str): The Hugging Face hub identifier for the model.
        max_context_length (int): The maximum character length allowed per
            source chunk to prevent exceeding the model's context window.
        max_sources_for_generation (int): The maximum number of top-ranked
            sourcesto inject into the prompt (avoids "lost in the middle"
            syndrome).
        tokenizer (Any): The loaded Hugging Face tokenizer instance.
        model (Any): The loaded Hugging Face causal language model.
    """
    model_name: str = "Qwen/Qwen3-0.6B"
    max_context_length: int = 1500
    max_sources_for_generation: int = 3
    model_config = {"arbitrary_types_allowed": True}
    tokenizer: Any = None
    model: Any = None

    _file_cache: Dict[str, str] = PrivateAttr(default_factory=dict)
    _project_root: Path = PrivateAttr(default_factory=Path.cwd)

    def load_model(self) -> None:
        """
        Load the LLM and tokenizer into memory with optimized CPU/GPU settings.

        Initializes the model with Scaled Dot-Product Attention (SDPA)
        for faster inference. Automatically falls back to float32 precision
        on CPU or uses float16 if a CUDA-compatible GPU is available.
        """
        TerminalColors.info(f"Loading the LLM : {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_cuda = torch.cuda.is_available()
        dtype: Any = torch.float16 if use_cuda else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True
        )
        if not use_cuda:
            self.model.to("cpu")

        TerminalColors.success("Modèle chargé avec succès.")

    def _resolve_path(self, file_path: str) -> Path:
        """
        Resolve a source file path relative to the project root.

        Args:
            file_path (str): The relative or absolute path to resolve.

        Returns:
            Path: An absolute Path object safe for file I/O operations.
        """
        p = Path(file_path)
        if p.is_absolute():
            return p
        return self._project_root / p

    def _read_file_cached(self, file_path: str) -> str:
        """
        Read a file's content and cache it in memory.

        This prevents redundant disk reads when multiple retrieved sources
        overlap or originate from the same source file.

        Args:
            file_path (str): The path to the file to read.

        Returns:
            str: The full content of the file, or an empty string if it fails.
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
        Extract the exact text span from a source file using character offsets.

        Args:
            source (MinimalSource): The source object containing the file path
                and the strict character boundaries to extract.

        Returns:
            str: The extracted substring representing the chunk.
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

        Extracts the text from the top-ranked sources, truncates them if
        necessary, and formats them into a strict instructional prompt
        using the model's specific chat template.

        Args:
            question (str): The user's query to answer.
            sources (List[MinimalSource]): The retrieved sources to use
                as context.

        Returns:
            str: The fully formatted prompt ready for tokenization.

        Raises:
            ValueError: If the tokenizer has not been loaded prior to calling.
        """
        if not self.tokenizer:
            raise ValueError("Le tokenizer n'est pas chargé.")

        top_sources = sources[:self.max_sources_for_generation]

        context_chunks = []
        for i, src in enumerate(top_sources, 1):
            raw_text = self._extract_text_from_source(src).strip()
            if raw_text:
                if len(raw_text) > self.max_context_length:
                    raw_text = raw_text[: self.max_context_length]
                context_chunks.append(f"Doc {i}:\n{raw_text}")

        context_text = "\n\n".join(context_chunks)

        system_content = (
            "You are a strict technical assistant. "
            "Answer the question using ONLY the provided documents. "
            "Your answer MUST be a single, short sentence. "
            "If the answer is not in the documents, "
            "reply EXACTLY with: I don't know."
        )

        user_content = (
            f"Documents:\n{context_text}\n\n"
            f"Question: {question}\nShort Answer:"
        )

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

    def _postprocess(self, full_answer: str) -> str:
        """
        Clean the raw generated string into a finalized short answer.

        Strips unnecessary line breaks and normalizes variants of
            "I don't know"
        to ensure consistency with the grading constraints.

        Args:
            full_answer (str): The raw text output generated by the LLM.

        Returns:
            str: The cleaned, single-sentence answer.
        """
        final_answer = full_answer.split('\n')[0].strip()
        if not final_answer or "don't know" in final_answer.lower()\
                or "do not know" in final_answer.lower():
            return "I don't know."
        return final_answer

    def answer(self,
               question: str,
               sources: List[MinimalSource],
               max_tokens: int = 35) -> str:
        """
        Generate a natural language answer based solely on the provided
        sources.

        Runs inference using `torch.inference_mode()` for optimal performance.
        Only the newly generated tokens are decoded and post-processed.

        Args:
            question (str): The user's question.
            sources (List[MinimalSource]): The retrieved sources for context.
            max_tokens (int, optional): The maximum number of new tokens
                to generate.
                Defaults to 35 to enforce concise answers.

        Returns:
            str: The generated and post-processed short answer.

        Raises:
            ValueError: If the model or tokenizer are not initialized.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError(
                "Model and Tokenizer must be loaded before usage."
            )

        prompt = self.build_prompt(question, sources)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = int(inputs.input_ids.shape[1])
        generated_tokens = outputs[0][input_length:]
        full_answer = str(self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True)
        ).strip()
        return self._postprocess(full_answer)
