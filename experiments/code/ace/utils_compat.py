"""Compatibility layer for appworld.utils functions that may not exist in all versions."""

from typing import Any, List

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return len(text) // 4  # rough heuristic fallback

try:
    from appworld.common.utils import chunk_and_return as _chunk_and_return
    chunk_and_return = _chunk_and_return
except ImportError:
    # Fallback: appworld may have chunk_list but not chunk_and_return
    try:
        from appworld.common.utils import chunk_list

        def chunk_and_return(
            input_list: List[Any],
            num_chunks: int,
            chunk_index: int,
        ) -> List[Any]:
            chunks = chunk_list(input_list, num_chunks=num_chunks)
            return chunks[chunk_index]
    except ImportError:

        def chunk_and_return(
            input_list: List[Any],
            num_chunks: int,
            chunk_index: int,
        ) -> List[Any]:
            n = len(input_list)
            num_chunks = min(num_chunks, n)
            chunk_size = n // num_chunks
            remainder = n % num_chunks
            start = chunk_index * chunk_size + min(chunk_index, remainder)
            end = start + chunk_size + (1 if chunk_index < remainder else 0)
            return input_list[start:end]


# Together / OpenAI-style 400 when prompt exceeds model context window
CONTEXT_LENGTH_EXCEEDED_MARKER = "longer than the model's context length"


def is_context_length_api_exception(exc: BaseException) -> bool:
    """True if exception message indicates input exceeded model context length."""
    return CONTEXT_LENGTH_EXCEEDED_MARKER in str(exc).lower()


def text_indicates_context_length_exceeded(text: str) -> bool:
    """True if log/stderr text contains a context-length exceeded API error."""
    t = (text or "").lower()
    if CONTEXT_LENGTH_EXCEEDED_MARKER in t:
        return True
    return (
        "invalid_request_error" in t
        and "context length" in t
        and "longer than" in t
    )
