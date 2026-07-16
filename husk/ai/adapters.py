import os
import json
import urllib.request
import urllib.error
import ssl
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

def get_ssl_context():
    """
    Returns an SSL context. Prefers certifi's bundle if available.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl._create_unverified_context()
        except AttributeError:
            return None

class BaseAdapter(ABC):
    """
    Abstract Base Class for LLM API Adapters.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """
        Sends a request to the LLM provider and returns the text response.
        """
        pass

class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI completions.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Fallback to environment variable
        key = api_key or os.environ.get("OPENAI_API_KEY")
        mdl = model or "gpt-4o-mini"
        super().__init__(key, mdl)

    def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            raise ValueError("OpenAI API Key is missing. Set OPENAI_API_KEY or configure it in husk.")
            
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30, context=get_ssl_context()) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP Error {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI: {e}")

class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic messages API.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        mdl = model or "claude-3-5-sonnet-20241022"
        super().__init__(key, mdl)

    def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            raise ValueError("Anthropic API Key is missing. Set ANTHROPIC_API_KEY or configure it in husk.")
            
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system:
            payload["system"] = system
            
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30, context=get_ssl_context()) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Anthropic HTTP Error {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Anthropic: {e}")

class OllamaAdapter(BaseAdapter):
    """
    Adapter for local Ollama instances.
    """
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        # host maps to api_key argument in base
        hst = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        mdl = model or "llama3"
        super().__init__(hst, mdl)

    def generate(self, prompt: str, system: str = "") -> str:
        url = f"{self.api_key}/api/generate"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
            
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=300, context=get_ssl_context()) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["response"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP Error {e.code}: {err_body}")
        except Exception as e:
            import socket
            if isinstance(e, socket.timeout) or "timed out" in str(e).lower():
                raise RuntimeError(f"Ollama request timed out: {e}")
            raise RuntimeError(f"Failed to communicate with Ollama at {self.api_key}: {e}. Ensure Ollama is running.")

def get_adapter(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> BaseAdapter:
    """
    Factory function to retrieve LLM adapter based on provider name.
    """
    prov = provider.lower()
    if prov == "openai":
        return OpenAIAdapter(api_key, model)
    elif prov == "anthropic":
        return AnthropicAdapter(api_key, model)
    elif prov in ("ollama", "local"):
        return OllamaAdapter(api_key, model) # api_key stores host URL for Ollama
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
