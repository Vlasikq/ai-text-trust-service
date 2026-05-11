"""
Общие LLM-бэкенды для генерации текстов
Используется в generate_dataset.py и generate_adversarial.py

"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)


class LLMBackend:
    def generate(self, system: str, user: str, temperature: float) -> Optional[str]:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass --api-key")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float) -> Optional[str]:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            text = resp.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            log.error(f"OpenAI API error: {e}")
            return None


class GigaChatBackend(LLMBackend):
    def __init__(self, model: str = "GigaChat"):
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole

        creds = os.getenv("GIGACHAT_CREDENTIALS")
        if not creds:
            raise ValueError("Set GIGACHAT_CREDENTIALS")

        self._client = GigaChat(
            credentials=creds,
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true",
            model=model,
        )
        self._model = model
        self._Chat = Chat
        self._Messages = Messages
        self._MessagesRole = MessagesRole

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float) -> Optional[str]:
        try:
            chat = self._Chat(
                messages=[
                    self._Messages(role=self._MessagesRole.SYSTEM, content=system),
                    self._Messages(role=self._MessagesRole.USER, content=user),
                ],
                temperature=min(temperature, 2.0),
                max_tokens=4096,
            )
            resp = self._client.chat(chat)
            text = resp.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            log.error(f"GigaChat API error: {e}")
            return None


class YandexGPTBackend(LLMBackend):
    BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, model: str = "yandexgpt-lite"):
        import requests

        self._requests = requests
        self._api_key = os.getenv("YANDEX_API_KEY")
        self._iam_token = os.getenv("YANDEX_IAM_TOKEN")
        self._folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not self._folder_id:
            raise ValueError("Set YANDEX_FOLDER_ID")
        if not self._api_key and not self._iam_token:
            raise ValueError("Set YANDEX_API_KEY or YANDEX_IAM_TOKEN")

        model_map = {
            "yandexgpt": f"gpt://{self._folder_id}/yandexgpt/latest",
            "yandexgpt-lite": f"gpt://{self._folder_id}/yandexgpt-lite/latest",
        }
        self._model_uri = model_map.get(model, model)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Api-Key {self._api_key}"
        else:
            headers["Authorization"] = f"Bearer {self._iam_token}"
        headers["x-folder-id"] = self._folder_id

        payload = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": 4096,
            },
            "messages": [
                {"role": "system", "text": system},
                {"role": "user", "text": user},
            ],
        }

        try:
            resp = self._requests.post(self.BASE_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = data["result"]["alternatives"][0]["message"]["text"]
            return text.strip() if text else None
        except Exception as e:
            log.error(f"YandexGPT API error: {e}")
            return None


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float) -> Optional[str]:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            text = response.text
            return text.strip() if text else None
        except Exception as e:
            log.error(f"Gemini API error: {e}")
            return None


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gigachat": "GigaChat",
    "yandex": "yandexgpt-lite",
    "gemini": "gemini-2.5-flash-lite",
}


def create_backend(
    backend: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMBackend:
    model = model or DEFAULT_MODELS.get(backend, "unknown")
    if backend == "openai":
        return OpenAIBackend(model, base_url, api_key)
    elif backend == "gigachat":
        return GigaChatBackend(model)
    elif backend == "yandex":
        return YandexGPTBackend(model)
    elif backend == "gemini":
        return GeminiBackend(model)
    raise ValueError(f"Unknown backend: {backend}")
