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

Data flows strictly in one direction: `data/raw/` → BM25 index
(`data/processed/`) → search results (`data/output/search_results/`) →
generated answers (`data/output/search_results_and_answer/`). All
exchanged structures (`MinimalSource`, `MinimalSearchResults`,
`MinimalAnswer`, `RagDataset`, ...) are validated with Pydantic v2
models, so a malformed dataset or a malformed intermediate file fails
fast with a readable error instead of crashing deep in the pipeline.

## Chunking Strategy

Python files and Markdown files are split differently, since code and
prose don't break apart the same way:

- **Python chunking** — <TODO: describe how `chunk_python` splits
  files (e.g. by function/class boundaries via the `ast` module, with
  a fallback to fixed-size slicing), and how it respects
  `--max_chunk_size`.>
- **Markdown chunking** — <TODO: describe how `chunk_markdown` splits
  files (e.g. by heading boundaries, then by paragraph if a section
  still exceeds the limit).>

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

<TODO: fill in with real numbers once available.>

| Metric                     | Docs questions | Code questions |
|-----------------------------|-----------------|-----------------|
| Recall@1                   | `<TODO>`        | `<TODO>`        |
| Recall@3                   | `<TODO>`        | `<TODO>`        |
| Recall@5                   | `<TODO>`        | `<TODO>`        |
| Recall@10                  | `<TODO>`        | `<TODO>`        |
| Indexing time (full corpus)| `<TODO>`        | —               |
| Retrieval time (200 q.)    | `<TODO>`        | —               |

Targets to reach: **80% recall@5** on docs questions, **50% recall@5**
on code questions, indexing under 5 minutes, and retrieval under 90
seconds for 200 questions.

<TODO: discuss the effect of `--max_chunk_size` on recall@k here,
e.g. whether smaller chunks helped or hurt on docs vs. code
questions.>

## Design Decisions

- <TODO: why BM25 over TF-IDF (or vice versa)?>
- <TODO: why append weighted file-name tokens to each chunk's token
  list — what retrieval failure did this fix?>
- <TODO: any other notable trade-off, e.g. how large `k` is chosen by
  default, how the index is persisted (pickle) and why.>

## Challenges Faced

- <TODO: e.g. balancing chunk size against the 2000-character hard
  cap while keeping enough context per chunk.>
- <TODO: e.g. handling malformed datasets/JSON gracefully without
  crashing the CLI, per the project's robustness requirement.>
- <TODO: any issue specific to running `Qwen/Qwen3-0.6B` on CPU.>

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
- - (AST For Code RAg Models)[https://medium.com/@jouryjc0409/ast-enables-code-rag-models-to-overcome-traditional-chunking-limitations-b0bc1e61bdab]


### AI Usage

AI assistance (Claude) was used for:

- <TODO: be specific and honest here, e.g. "drafting the tqdm progress
  bar integration in the indexer and reviewing exception handling in
  the pipeline commands" or "discussing BM25 vs TF-IDF trade-offs
  before implementation".>
- Generating the initial structure of this README, subsequently
  reviewed and filled in with the project's actual design choices and
  results.

All AI-generated code was reviewed, tested, and understood before
being kept in the final submission, per the project's AI usage
guidelines.
