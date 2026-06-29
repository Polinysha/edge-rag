"""
LLM client connected to OpenRouter through the OpenAI-compatible SDK.
Uses OpenRouter's Free Models Router (openrouter/free), which automatically
picks an available free-tier model for each request. Pinning to a specific
":free" model slug is fragile — OpenRouter's free model list changes often,
and specific slugs get discontinued without notice.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "openrouter/free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def call_llm(messages: list[dict], temperature: float = 0.0) -> str:
    """
    Sends a chat completion request to the LLM via OpenRouter.
    messages follows the standard OpenAI format: [{"role": ..., "content": ...}, ...]
    Returns the assistant's reply text.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    reply = call_llm([{"role": "user", "content": "Say hello in one short sentence."}])
    print("LLM reply:", reply)