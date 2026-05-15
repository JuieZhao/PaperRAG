"""
LLM answer generation with DeepSeek (OpenAI-compatible API)
"""
import os
from collections.abc import Generator
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

_client = None


def get_client():
    """Get or create cached OpenAI-compatible client for DeepSeek."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=60.0,
        )
    return _client


SYSTEM_PROMPT = """You are an academic research assistant. Answer questions based on provided literature excerpts.

Requirements:
1. Base your answer on the provided literature content — do not fabricate
2. Cite sources by paper name when referencing them
3. If the literature does not contain relevant information, honestly state so
4. Answer in the same language as the question
5. You may supplement with your domain knowledge, but clearly distinguish what comes from the literature vs. your knowledge"""


def _build_context(retrieved_chunks: list[dict]) -> str:
    """Build formatted context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks):
        parts.append(
            f"[Source {i+1}] Paper: {chunk['paper_name']}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, context: str) -> list[dict]:
    """Build messages for the LLM call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""## Question
{query}

## Relevant Literature Excerpts
{context}

Please answer the question based on the above literature excerpts."""},
    ]


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str | None = None,
) -> str:
    """
    Generate answer from retrieved chunks.
    Returns answer string, or error message on failure.
    """
    if not retrieved_chunks:
        return "No relevant literature found to answer this question."

    try:
        client = get_client()
        model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        context = _build_context(retrieved_chunks)
        messages = _build_messages(query, context)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    except RateLimitError:
        return "⚠️ API rate limit exceeded. Please wait a moment and try again."
    except APITimeoutError:
        return "⚠️ Request timed out. The model may be overloaded. Please try again."
    except APIError as e:
        return f"⚠️ API error: {e.message if hasattr(e, 'message') else str(e)[:200]}"
    except Exception as e:
        return f"⚠️ Unexpected error generating answer: {str(e)[:200]}"


def generate_answer_stream(
    query: str,
    retrieved_chunks: list[dict],
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Stream answer token by token from retrieved chunks.
    Yields text fragments; yields error strings prefixed with ⚠️ on failure.
    """
    if not retrieved_chunks:
        yield "No relevant literature found to answer this question."
        return

    try:
        client = get_client()
        model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        context = _build_context(retrieved_chunks)
        messages = _build_messages(query, context)

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except RateLimitError:
        yield "⚠️ API rate limit exceeded. Please wait a moment and try again."
    except APITimeoutError:
        yield "⚠️ Request timed out. The model may be overloaded. Please try again."
    except APIError as e:
        yield f"⚠️ API error: {e.message if hasattr(e, 'message') else str(e)[:200]}"
    except Exception as e:
        yield f"⚠️ Unexpected error generating answer: {str(e)[:200]}"
