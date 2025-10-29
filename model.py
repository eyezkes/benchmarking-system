# model.py
from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from openai import OpenAI

from errors import ModelError

logger = logging.getLogger(__name__)


class Model:
    """Thin wrapper around an LLM client for text generation."""

    def __init__(
        self, 
        model_name: str, 
        api_key: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini").
            api_key: API key for the OpenAI client.
            default_params: Default parameters for model generation (e.g., temperature, max_tokens).
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.params = params or {}

    def name(self) -> str:

        return self.model_name
    
    def get_params(self) -> Dict[str, Any]:

        return dict(self.params)

    def generate(
        self, 
        prompt: str, 
        system_content: str, 
        *, 
        timeout: Optional[float] = None,
        **model_params
    ) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: User prompt text.
            system_content: System role message to guide style/behavior.
            timeout: Optional request timeout in seconds (if supported by client).
            **model_params: Additional model parameters (temperature, max_tokens, top_p, etc.)
                           These override params for this call.

        Returns:
            The model's text response (stripped).

        Raises:
            ValueError: If inputs are empty.
            ModelError: If the API call fails or returns an invalid payload.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")
        if not system_content or not isinstance(system_content, str):
            raise ValueError("system_content must be a non-empty string")

        # Merge params with call-specific params (call-specific override defaults)
        params = {**self.params, **model_params}

        try:
            logger.debug("Generating with model=%s, params=%s", self.model_name, params)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
                **params  # temperature, max_tokens, top_p, etc.
            )
        except Exception as exc:
            logger.error("Model request failed: %s", exc)
            raise ModelError(f"Model request failed: {exc}") from exc

        try:
            text = response.choices[0].message.content
        except Exception as exc:
            logger.error("Unexpected model response structure: %s", exc)
            raise ModelError(f"Unexpected model response structure: {exc}") from exc

        if not text or not isinstance(text, str):
            raise ModelError("Empty or invalid content in model response")

        text = text.strip()
        logger.debug("Generation complete (len=%d)", len(text))
        return text