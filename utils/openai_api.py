# utils/openai_api.py
import os
import openai
import time

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or (os.environ.get("STREAMLIT_OPENAI_API_KEY") or "")
if not OPENAI_API_KEY:
    # allow Streamlit secrets if set
    try:
        import streamlit as _st
        OPENAI_API_KEY = _st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

openai.api_key = OPENAI_API_KEY

# simple moderation wrapper
def moderation_check(text: str) -> dict:
    try:
        resp = openai.Moderation.create(input=text)
        # resp contains 'results' with classification booleans
        return resp
    except Exception as e:
        # if moderation fails, return safe default (allow) but log
        return {"error": str(e)}

def call_gpt(prompt: str, max_tokens: int = 400, temperature: float = 0.2, model: str = "gpt-5-thinking-mini"):
    """
    Generic call: send 'prompt' as user content and return assistant text.
    Adjust model name if needed.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in env or Streamlit secrets.")

    # simple retry with exponential backoff for transient errors
    for attempt in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are X-Tutor, a concise, student-friendly educational assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except openai.error.RateLimitError as e:
            wait = (2 ** attempt) * 1.0
            time.sleep(wait)
            continue
        except Exception as e:
            # bubble up other errors
            raise
    # if we get here, try one last time and let exception propagate
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are X-Tutor, a concise, student-friendly educational assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
