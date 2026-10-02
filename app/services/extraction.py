import json ,os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.prompts.templates import (
    SYSTEM_PROMPT,
    USER_PROMPT
)
from app.schema.device import Device


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



SUSPICIOUS_PHRASES = [
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "you are now",
]


def validate_external_text(text: str) -> None:
    """
      Reject obvious prompt injection attempts before calling the LLM.
    """

    normalized = text.lower()

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in normalized:
            raise ValueError(
                "Potential prompt injection detected."
            )


def call_llm(user_prompt: str)->str:
    """
    Send the system and user messages to the LLM.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system","content": SYSTEM_PROMPT,
            },
            {
                "role": "user","content": user_prompt,
            },
        ],
        response_format={ "type": "json_object"},
    )

    return response.choices[0].message.content


def extract_device(text: str) -> Device:
    """
    Extract structured device information from messy text.

    Maximum of two retries after the initial attempt.
    """

    validate_external_text(text)

    user_prompt = USER_PROMPT.format(
        text=text
    )

    last_error: str | None = None

    max_attempts = 3

    for attempt in range(max_attempts):

        if attempt > 0:
            user_prompt = f"""
            The previous output failed Pydantic validation.

            Validation error:
            {last_error}

            Return a corrected JSON object.

            <external_data>
            {text}
            </external_data>

            The output must match:

            {{
                "brand": "string",
                "model": "string",
                "specs": {{
                    "key": "value"
                }},
                "release_year": 2024,
                "price_tier": "budget | mid-range | flagship"
            }}
"""

        raw_response = call_llm(user_prompt)

        try:
            parsed_data: Any = json.loads(raw_response)

            device = Device.model_validate(
                parsed_data
            )

            return device

        except json.JSONDecodeError as error:
            last_error = (
                f"Invalid JSON returned by the model: {error}"
            )

        except ValidationError as error:
            last_error = str(error)

    raise ValueError(
        f"Could not extract a valid device after "
        f"{max_attempts} attempts. "
        f"Last error: {last_error}"
    )