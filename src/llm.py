"""Local open-source LLM client using Ollama's localhost REST API."""

import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class OllamaClient:
    def __init__(self, model=None, base_url=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def chat(self, messages, temperature=0.1, num_predict=None):
        return self.chat_response(messages, temperature=temperature,
                                  num_predict=num_predict)["message"]["content"].strip()

    def chat_response(self, messages, tools=None, temperature=0.1, num_predict=None):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
                   "options": {"temperature": temperature,
                               "num_predict": int(num_predict or os.getenv("OLLAMA_NUM_PREDICT", "160")),
                               "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                               "repeat_penalty": 1.15}}
        if tools:
            payload["tools"] = tools
        request = Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data
        except (URLError, HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama unavailable; start it and pull {self.model}: {exc}") from exc

class GeminiEmbeddingClient:
    """Gemini text embeddings for asymmetric retrieval (documents and queries)."""

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def embed(self, texts, model=None, task_type="RETRIEVAL_DOCUMENT"):
        model = model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing; set it in voiceops_agent/.env to use Gemini embeddings.")
        requests = [{
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "embedContentConfig": {"taskType": task_type},
        } for text in texts]
        payload = json.dumps({"requests": requests}).encode("utf-8")
        request = Request(self.endpoint.format(model=model), data=payload,
                          headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                          method="POST")
        for attempt in range(3):
            try:
                with urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                vectors = [item["values"] for item in data["embeddings"]]
                if len(vectors) != len(texts) or any(not vector for vector in vectors):
                    raise RuntimeError("Gemini returned an incomplete embedding batch.")
                return vectors
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f"Gemini embedding request failed for {model} (HTTP {exc.code}).") from exc
                try:
                    delay = min(5.0, max(0.0, float((exc.headers or {}).get("Retry-After", "0"))))
                except (TypeError, ValueError):
                    delay = 0.0
                time.sleep(delay or (0.5 * (2 ** attempt)))
            except (URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Gemini embedding request failed for {model}: {exc}") from exc
                time.sleep(0.5 * (2 ** attempt))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Gemini returned an invalid embedding response for {model}.") from exc
