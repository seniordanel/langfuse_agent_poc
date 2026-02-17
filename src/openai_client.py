import os
from typing import List, Dict, Optional

from openai import OpenAI


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    Returns a configured OpenAI client.

    Args:
        api_key:  OpenAI API key. Falls back to OPENAI_API_KEY env var if None.
        base_url: Custom endpoint URL. Falls back to OPENAI_BASE_URL env var,
                  then to the default OpenAI endpoint if None.

    Returns:
        An initialized OpenAI client instance.
    """
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    resolved_url = base_url or os.getenv("OPENAI_BASE_URL") or None

    return OpenAI(api_key=resolved_key, base_url=resolved_url)


def invoke(
    client: OpenAI,
    langfuse,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Calls the OpenAI chat completions API and updates Langfuse with token usage.

    Args:
        client:   An OpenAI client instance (from get_client()).
        langfuse: A Langfuse instance used to update the current generation.
        messages: A list of message dicts, e.g. [{"role": "user", "content": "..."}].
        model:    The model identifier to use.

    Returns:
        The assistant's response content as a string.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Update Langfuse with token usage
    if response.usage:
        langfuse.update_current_generation(
            usage_details={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=model,
        )

    return response.choices[0].message.content or ""
