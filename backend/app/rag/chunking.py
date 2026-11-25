from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from app.core.config import get_settings


@lru_cache
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(get_settings().embeddings_model),
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )