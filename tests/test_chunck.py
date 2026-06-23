from src.chuncker import chunk_python, chunk_markdown


def test_chunk_python() -> None:
    chunks = chunk_python("data/raw/vllm-0.10.1/vllm/engine/llm_engine.py")
    print(f"Number of chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"\n--- {chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}] ---")
        print(chunk.content[:200])

def test_chunk_markdown() -> None:
    chunks = chunk_markdown("data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md")
    print(f"Number of chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"\n--- {chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}] ---")
        print(chunk.content[:200])

if __name__ == "__main__":
    test_chunk_python()
    test_chunk_markdown()
