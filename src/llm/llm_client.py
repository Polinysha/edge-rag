import os
import time

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"
API_KEY = os.getenv("GROQ_API_KEY")

REQUEST_TIMEOUT_S = 60.0
SDK_MAX_RETRIES = 2
CALL_MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 5.0

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=REQUEST_TIMEOUT_S,
    max_retries=SDK_MAX_RETRIES,
)

async_client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=REQUEST_TIMEOUT_S,
    max_retries=SDK_MAX_RETRIES,
)


def call_llm(messages: list[dict], temperature: float = 0.0) -> str:

    last_error: Exception | None = None
    for attempt in range(1, CALL_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            last_error = ValueError("LLM returned an empty response")
        except Exception as e:
            last_error = e
        if attempt < CALL_MAX_ATTEMPTS:
            print(f"    ! call_llm attempt {attempt}/{CALL_MAX_ATTEMPTS} failed "
                  f"({last_error}); retrying in {RETRY_BACKOFF_S}s")
            time.sleep(RETRY_BACKOFF_S)
    raise RuntimeError(
        f"call_llm failed after {CALL_MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


if __name__ == "__main__":
    reply = call_llm([{"role": "user", "content": "Say hello in one short sentence."}])
    print("LLM reply:", reply)