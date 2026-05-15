"""
LLM answer generation with DeepSeek (OpenAI-compatible API)
Supports multi-turn conversation with history.
"""
import os
from collections.abc import Generator
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

_client = None
MAX_HISTORY_TURNS = 10  # Keep last N Q&A pairs for context


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


SYSTEM_PROMPT = """You are an academic research assistant. Answer questions based on provided literature excerpts and conversation context.

Requirements:
1. Base your answer on the provided literature content — do not fabricate
2. Cite sources by paper name AND page number when referencing them (e.g. "(Smith et al., p.23)")
3. If the literature does not contain relevant information, honestly state so
4. Answer in the same language as the question
5. You may supplement with your domain knowledge, but clearly distinguish what comes from the literature vs. your knowledge
6. When the user asks follow-up questions (e.g. "explain more", "what about X"), refer to the previous conversation context"""


def _build_context(retrieved_chunks: list[dict]) -> str:
    """Build formatted context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks):
        parts.append(
            f"[Source {i+1}] Paper: {chunk['paper_name']} (p.{chunk.get('page', '?')})\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_messages(
    query: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build messages for the LLM call, including conversation history.
    history: list of {"role": "user"|"assistant", "content": "..."}
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent conversation history (last N turns)
    if history:
        recent = history[-(MAX_HISTORY_TURNS * 2):]  # Each turn = user + assistant
        messages.extend(recent)

    # Current question with retrieved context
    messages.append({
        "role": "user",
        "content": f"""## Question
{query}

## Relevant Literature Excerpts
{context}

Please answer the question based on the above literature excerpts."""
    })

    return messages


def _call_llm(
    client,
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
):
    """Unified LLM call with error handling."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except RateLimitError:
        raise RuntimeError("⚠️ API rate limit exceeded. Please wait a moment and try again.")
    except APITimeoutError:
        raise RuntimeError("⚠️ Request timed out. The model may be overloaded. Please try again.")
    except APIError as e:
        raise RuntimeError(f"⚠️ API error: {e.message if hasattr(e, 'message') else str(e)[:200]}")
    except Exception as e:
        raise RuntimeError(f"⚠️ Unexpected error generating answer: {str(e)[:200]}")


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str | None = None,
    history: list[dict] | None = None,
    temperature: float = 0.3,
) -> str:
    """
    Generate answer from retrieved chunks with optional conversation history.
    Returns answer string, or error message on failure.
    """
    if not retrieved_chunks:
        return "No relevant literature found to answer this question."

    try:
        client = get_client()
        model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        context = _build_context(retrieved_chunks)
        messages = _build_messages(query, context, history)

        response = _call_llm(client, model, messages, temperature=temperature)
        return response.choices[0].message.content

    except RuntimeError as e:
        return str(e)


def generate_answer_stream(
    query: str,
    retrieved_chunks: list[dict],
    model: str | None = None,
    history: list[dict] | None = None,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """
    Stream answer token by token from retrieved chunks.
    Accepts conversation history for multi-turn support.
    Yields text fragments; yields error strings prefixed with ⚠️ on failure.
    """
    if not retrieved_chunks:
        yield "No relevant literature found to answer this question."
        return

    try:
        client = get_client()
        model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        context = _build_context(retrieved_chunks)
        messages = _build_messages(query, context, history)

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except RuntimeError as e:
        yield str(e)
