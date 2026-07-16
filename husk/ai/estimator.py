from typing import Dict, Tuple

# Pricing per 1,000,000 tokens (in USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Ollama / Local
    "llama3": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
}

class TokenEstimator:
    """
    Handles local token estimation and cost calculation for LLM processing.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pricing = MODEL_PRICING.get(model_name, {"input": 0.50, "output": 1.50}) # fallback default

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Estimates the token count of a given text.
        Uses tiktoken if installed, otherwise falls back to character approximation.
        """
        try:
            import tiktoken
            # cl100k_base is the standard vocabulary for GPT-4 and Claude tokens are closely aligned
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback approximation: ~4 characters per token
            return len(text) // 4

    def calculate_cost(self, total_tokens: int, estimated_completion_ratio: float = 0.25) -> Tuple[float, int, int]:
        """
        Calculates the estimated cost (in USD) for processing total_tokens.
        
        Args:
            total_tokens: Number of prompt/input tokens.
            estimated_completion_ratio: Multiplier to estimate completion output tokens.
            
        Returns:
            Tuple: (Estimated Cost USD, Input Tokens, Output Tokens)
        """
        input_tokens = total_tokens
        output_tokens = int(total_tokens * estimated_completion_ratio)
        
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        
        total_cost = input_cost + output_cost
        return total_cost, input_tokens, output_tokens

    @staticmethod
    def format_cost(cost_usd: float) -> str:
        """
        Formats cost in a user-friendly way.
        """
        if cost_usd < 0.01:
            return f"${cost_usd:.5f}"
        return f"${cost_usd:.2f}"
