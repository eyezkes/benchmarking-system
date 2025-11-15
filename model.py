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
        system_prompt: str | None = None,
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
        self.system_prompt = system_prompt or (
        "You are a knowledgeable and reliable AI assistant.\n"
        "Answer accurately, clearly, and concisely.\n"
        "Avoid unnecessary explanations unless requested."
    )


    def get_name(self) -> str:

        return self.model_name
    
    def get_params(self) -> Dict[str, Any]:

        return dict(self.params)
    
    def get_system_prompt(self) -> str:
        return self.system_prompt


    def generate(
        self,
        prompt: str,
        *,
        timeout: Optional[float] = None,
        **model_params
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt text.
            timeout: Optional request timeout in seconds.
            **model_params: Extra model parameters (temperature, max_tokens, etc.)
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        params = {**self.params, **model_params}

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
                **params
            )
        except Exception as exc:
            raise ModelError(f"Model request failed: {exc}") from exc

        text = response.choices[0].message.content.strip()
        if not text:
            raise ModelError("Empty or invalid content in model response")

        return text