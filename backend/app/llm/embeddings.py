from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().embeddings_model)


class EmbeddingInputTooLong(ValueError):
    pass


def validate_embedding_inputs(
    embeddings: HuggingFaceEmbeddings, documents: list[Document]
) -> None:
    # LangChain keeps the underlying SentenceTransformer in its private client.
    model = embeddings._client
    max_tokens = model.max_seq_length
    tokenizer = model.tokenizer
    if not isinstance(max_tokens, int) or max_tokens <= 0 or tokenizer is None:
        raise RuntimeError("Embedding model must expose a tokenizer and input token limit")
    options = getattr(embeddings, "encode_kwargs", {})
    prompt = options.get("prompt")
    if prompt is None:
        prompt_name = options.get("prompt_name") or getattr(model, "default_prompt_name", None)
        prompt = getattr(model, "prompts", {}).get(prompt_name, "")
    for document in documents:
        text = prompt + document.page_content.replace("\n", " ")
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(tokens) > max_tokens:
            raise EmbeddingInputTooLong(
                f"A chunk has {len(tokens)} tokens, but the embedding model accepts "
                f"at most {max_tokens} including special tokens. Reduce chunk_size in "
                "app/rag/chunking.py and re-upload the document."
            )