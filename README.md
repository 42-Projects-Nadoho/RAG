*This project has been created as part of the 42 curriculum by nadoho.*

# RAG Against the Machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) pipeline
from scratch in Python. Instead of relying on a language model's frozen
training-time knowledge, the system indexes an external codebase (the
vLLM 0.10.1 source tree), retrieves the snippets most relevant to a
question at answer time, and generates a grounded answer with a small
local model (`Qwen/Qwen3-0.6B`).

The pipeline covers the four classic RAG stages:

1. **Indexing** — chunk the codebase and build a searchable index.
2. **Retrieval** — match a question against the index and return the
   top-k most relevant source locations.
3. **Augmenting** — feed those sources into the model's context window.
4. **Generating** — produce a natural-language answer grounded in the
   retrieved context.

Retrieval quality is measured with recall@k against reference datasets
of documentation and code questions.

## System Architecture

The project is organized as a `src/` Python module exposing a CLI built
with [Python Fire](https://github.com/google/python-fire). It is
orchestrated by a single `RagPipeline` class that wires together four
components:

| Component            | Responsibility                                            |
|-----------------------|------------------------------------------------------------|
| `CodebaseIndexer`     | Walks `data/raw/`, chunks files, builds a BM25 index        |
| `CodebaseRetriever`   | Loads the persisted index and answers top-k search queries  |
| `RagGenerator`        | Loads `Qwen/Qwen3-0.6B` and generates answers from context   |
| `RagEvaluator`        | Computes recall@k against a ground-truth dataset            |

Data flows strictly in one direction: `data/raw/` -> BM25 index
(`data/processed/`) -> search results (`data/output/search_results/`) ->
generated answers (`data/output/search_results_and_answer/`). All
exchanged structures (`MinimalSource`, `MinimalSearchResults`,
`MinimalAnswer`, `RagDataset`, ...) are validated with Pydantic v2
models, so a malformed dataset or a malformed intermediate file fails
fast with a readable error instead of crashing deep in the pipeline.

## Chunking Strategy

Python files and Markdown files are split differently, since code and
prose don't break apart the same way:

- **Python chunking (AST-based):** Instead of blind slicing, the parser uses Python's built-in `ast` module to extract structural nodes (Functions, Classes). If an isolated node exceeds the `--max_chunk_size`, it is recursively split or hard-sliced as a fallback. This guarantees that logical blocks of code remain intact in the context window.
- **Markdown chunking:** Files are split logically by Markdown headings (`#`, `##`), then by paragraphs if a section still exceeds the limit.

Every chunk is capped at `--max_chunk_size` characters (2000 by
default), since the grading moulinette rejects any retrieved source
wider than its `max_context_length`. Smaller chunk sizes were also
tested; see the Performance Analysis section for their effect on
recall@k.

## Retrieval Method

Retrieval uses **BM25**, a classic lexical ranking function built on
top of term frequency, inverse document frequency, and document-length
normalization (`k1=1.5`, `b=0.75` by default). At indexing time, each
chunk's tokens are augmented with tokens extracted from its file name
(repeated to boost their weight), so that identifier-like queries can
still match a chunk even when the identifier itself doesn't appear
verbatim in the body of the chunk.

At query time, the query is tokenized the same way, scored against
every chunk's BM25 score, and the top-k highest-scoring chunks are
returned as ranked source locations (`file_path`,
`first_character_index`, `last_character_index`).

## Performance Analysis

### 1. Retrieval Accuracy (Recall@K)
*Evaluated with default parameters: max_chunk_size=2000, BM25 (k1=1.5, b=0.75).*

| Metric                     | Docs questions | Code questions | Target |
|-----------------------------|-----------------|-----------------|-----------------
| Recall@1                   | `<TODO>`        | `<TODO>`        | `None`        |
| Recall@3                   | `<TODO>`        | `<TODO>`        | `None`        |
| Recall@5                   | `<TODO>`        | `<TODO>`        | **Docs: >80% / Code: >50%**       |
| Recall@10                  | `<TODO>`        | `<TODO>`        | `None`        |

### 2. Execution & Resource Metrics
*Tested on `<TODO: e.g., Apple M1 / 16GB RAM>`.*

| Metric | Measurement | Target |
|---|---|---|
| **Indexing time (Full corpus)** | `<TODO>s` | < 300s (5 min) |
| **Retrieval time (200 queries)** | `<TODO>s` | < 90s |
| **Generation time (per answer)** | `<TODO>s` | - |
| **Index File Size on Disk** | `<TODO> MB` | - |
| **Peak RAM Usage (Indexing)** | `<TODO> MB` | - |

### 3. Ablation Study: Impact of Chunk Size
*Comparing how chunk size affects Recall@5.*

| Chunk Size | Docs Recall@5 | Code Recall@5 | Indexing Time | Note |
|---|---|---|---|---|
| **500 chars** | `<TODO>%` | `<TODO>%` | `<TODO>s` | *Too fragmented for full code blocks.* |
| **1000 chars**| `<TODO>%` | `<TODO>%` | `<TODO>s` | *Good balance, but misses long doc sections.* |
| **2000 chars**| `<TODO>%` | `<TODO>%` | `<TODO>s` | *Optimal for this moulinette constraint.* |

## Design Decisions

- **BM25 over TF-IDF:** BM25 handles term-frequency saturation better than TF-IDF. In codebases, keywords (like `request`, `tensor`, `async`) repeat heavily within a single file. TF-IDF would over-reward these repetitive chunks, whereas BM25 dampens the score after the first few occurrences, leading to more relevant matching.
- **Pydantic for Data Flow:** Using Pydantic models for every intermediate step prevents silent downstream failures. RAG pipelines are notoriously difficult to debug when text artifacts get misaligned; strict schemas guarantee data integrity between the indexing, searching, and generation steps.
- **Index Serialization:** The BM25 index is persisted via `pickle`. While JSON is more portable, `pickle` allows near-instantaneous loading of complex Python dictionaries holding term frequencies, significantly speeding up the CLI response time for single queries.

## Challenges Faced

- **Respecting the 2000-character Hard Cap:** Code logic is easily destroyed if sliced arbitrarily. Implementing the AST-based chunking required handling edge cases where a single function body was wider than 2000 characters, requiring a graceful fallback mechanism.
- **Model Constraints:** Running `Qwen/Qwen3-0.6B` locally without a dedicated GPU required optimizing the context injection. Feeding too many retrieved chunks (large K) caused generation times to spike and the model to hallucinate or lose focus (the "lost in the middle" phenomenon).
- **Data Layout Artifacts:** Decompressing the datasets often left nested wrapper folders (e.g., `public/`). A custom `_flatten_single_subdir` logic had to be implemented to dynamically unwrap datasets without breaking the corpus structure expected by the grading moulinette.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) as package/dependency manager

### Installation

```bash
uv sync
```

### Data setup

Place the corpus and datasets archives at the repository root, then
run the extraction helper (see `setup_data.py`), which unpacks them
into the expected layout:

```bash
python setup_data.py <corpus_archive> <datasets_archive>
```

This populates `data/raw/` (the vLLM source tree) and
`data/datasets/` (the `UnansweredQuestions/` and `AnsweredQuestions/`
question sets).

### Running the pipeline

```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 2000

# 2. Search a dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 3. Score with the moulinette
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

# 4. Generate answers
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Single-query commands are also available for debugging:

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "How to configure the OpenAI server?" --k 5
uv run python -m src evaluate \
    --student_search_results_path <path> \
    --dataset_path <path>
```

### Makefile

```bash
make install      # install dependencies
make run          # run the main script
make debug        # run under pdb
make clean        # remove caches / temp files
make lint         # flake8 + mypy
make lint-strict  # flake8 + mypy --strict
```

## Example Usage

```
$ uv run python -m src index --max_chunk_size 2000
Chunking:    100%|##########| 1965/1965 [00:00<00:00, 16710 file/s]
Tokenizing:  100%|##########| 13466/13466 [00:01<00:00, 10668 chunk/s]
Ingestion complete! Indexed 13466 chunks under data/processed/

$ uv run python -m src search "How does the scheduler batch requests?" --k 3
data/raw/vllm-0.10.1/vllm/core/scheduler.py [1200: 2400]
data/raw/vllm-0.10.1/docs/design/scheduler.md [0: 1800]
data/raw/vllm-0.10.1/vllm/engine/llm_engine.py [430: 1900]
```

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Okapi BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Python Fire documentation](https://github.com/google/python-fire)
- [uv documentation](https://docs.astral.sh/uv/)
- [vLLM documentation](https://docs.vllm.ai/)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [AST For Code RAg Models](https://medium.com/@jouryjc0409/ast-enables-code-rag-models-to-overcome-traditional-chunking-limitations-b0bc1e61bdab)


### AI Usage

AI assistance (Claude / Gemini) was used for:
- Accelerating the development of regular expressions and AST node parsing logic within the CodebaseIndexer.
- Designing strictly typed Pydantic models for robust data validation between pipeline steps.
- Generating the structural layout of this README and structuring the performance evaluation matrix.
- Troubleshooting terminal issues regarding mypy strict typing checks and path exclusions.
