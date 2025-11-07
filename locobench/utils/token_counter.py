"""
Token counting utilities for LoCoBench
Supports multiple tokenization methods for different model providers
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate OpenAI token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.debug("tiktoken not available, using approximate token counting")


def count_tokens_openai(text: str, model_name: str = "gpt-4") -> int:
    """
    Count tokens for OpenAI models using tiktoken (accurate) or approximation.
    
    Args:
        text: Text to count tokens for
        model_name: OpenAI model name (e.g., "gpt-4", "o3", "gpt-4o")
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Try to use tiktoken for accurate counting
    if TIKTOKEN_AVAILABLE:
        try:
            # Determine encoding based on model
            if model_name.startswith(("o1", "o3", "o4")):
                encoding_name = "cl100k_base"  # o-series uses GPT-4 tokenizer
            elif model_name.startswith("gpt-4"):
                encoding_name = "cl100k_base"
            elif model_name.startswith("gpt-3.5"):
                encoding_name = "cl100k_base"
            else:
                encoding_name = "cl100k_base"  # Default to GPT-4 tokenizer
            
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.debug(f"tiktoken encoding failed for {model_name}: {e}, using approximation")
    
    # Fallback: approximate counting (1 token ≈ 4 characters for English)
    # This is a rough estimate, but works reasonably well for code
    return len(text) // 4


def count_tokens_google(text: str) -> int:
    """
    Count tokens for Google models (Gemini).
    Uses approximation since Google doesn't provide a public tokenizer.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count (Gemini uses ~4 chars per token)
    """
    if not text:
        return 0
    
    # Gemini models use approximately 4 characters per token
    return len(text) // 4


def count_tokens_claude(text: str) -> int:
    """
    Count tokens for Claude models.
    Uses approximation since Anthropic doesn't provide a public tokenizer.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count (Claude uses ~4 chars per token)
    """
    if not text:
        return 0
    
    # Claude models use approximately 4 characters per token
    return len(text) // 4


def count_tokens(text: str, model_name: Optional[str] = None, provider: Optional[str] = None) -> int:
    """
    Count tokens for text based on model provider.
    
    Args:
        text: Text to count tokens for
        model_name: Model name (e.g., "gpt-4", "o3", "gemini-2.5-pro")
        provider: Provider name ("openai", "google", "claude", "custom")
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Determine provider from model name if not provided
    if provider is None and model_name:
        model_lower = model_name.lower()
        if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
            provider = "openai"
        elif "gemini" in model_lower or "google" in model_lower:
            provider = "google"
        elif "claude" in model_lower:
            provider = "claude"
        else:
            provider = "openai"  # Default
    
    # Count tokens based on provider
    if provider == "openai":
        return count_tokens_openai(text, model_name or "gpt-4")
    elif provider == "google":
        return count_tokens_google(text)
    elif provider == "claude":
        return count_tokens_claude(text)
    else:
        # Default approximation for unknown providers
        return len(text) // 4
