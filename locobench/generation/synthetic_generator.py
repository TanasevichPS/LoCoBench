"""
Synthetic Project Generator for LoCoBench

This module generates realistic multi-file software projects using LLMs.
The generated projects serve as contexts for evaluating long-context LLMs 
in software development scenarios.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random

import openai
import httpx
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, TaskID

from ..core.config import Config
from ..core.task import TaskCategory, DifficultyLevel
from ..utils.rate_limiter import APIRateLimitManager

# Set up logging
logger = logging.getLogger(__name__)

# Hugging Face support
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers/torch not available. Hugging Face models will not be available.")

def setup_generation_logging(log_file: str = None) -> logging.Logger:
    """Setup structured logging for generation process"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/generation_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    gen_logger = logging.getLogger('locobench.generation')
    gen_logger.setLevel(logging.INFO)
    gen_logger.addHandler(file_handler)
    gen_logger.addHandler(console_handler)
    
    return gen_logger


class APIError(Exception):
    """Custom exception for API errors with specific provider info"""
    def __init__(self, provider: str, error_type: str, message: str, original_error: Exception = None, should_retry: bool = True):
        self.provider = provider
        self.error_type = error_type
        self.message = message
        self.original_error = original_error
        self.should_retry = should_retry
        super().__init__(f"{provider} {error_type}: {message}")


class CriticalAuthError(Exception):
    """Critical authentication error that should stop the entire process"""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"🚨 CRITICAL AUTH FAILURE - {provider}: {message}")


async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, provider: str = "Unknown"):
    """Enhanced retry logic with exponential backoff and critical error handling"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            error_str = str(e).lower()
            
            # Critical auth errors - stop immediately, don't retry
            if any(pattern in error_str for pattern in [
                "auth", "unauthorized", "forbidden", "invalid api key", "api key", 
                "authentication failed", "credentials", "access denied", "token expired",
                "expiredtokenexception", "security token", "session token"
            ]):
                if "expired" in error_str or "expiredtoken" in error_str:
                    raise CriticalAuthError(provider, f"Authentication token expired: {str(e)}")
                else:
                    raise CriticalAuthError(provider, f"Authentication failed: {str(e)}")
            
            # Retryable errors
            elif any(pattern in error_str for pattern in [
                "rate limit", "too many requests", "connection", "timeout", 
                "network", "502", "503", "504", "internal error", "server error",
                "throttlingexception", "too many tokens"  # Claude API specific
            ]):
                if attempt < max_retries - 1:
                    # Use longer delay for throttling errors
                    if "throttlingexception" in error_str or "too many tokens" in error_str:
                        delay = min(base_delay * (3 ** attempt), max_delay * 2)  # Longer delay for throttling
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    error_type = "RATE_LIMIT" if "rate limit" in error_str or "throttling" in error_str else "CONNECTION_ERROR"
                    raise APIError(provider, error_type, f"Max retries exceeded: {str(e)}", should_retry=False)
            
            # Unknown errors - treat as non-retryable
            else:
                raise APIError(provider, "UNKNOWN_ERROR", f"Unexpected error: {str(e)}", should_retry=False)
    
    # Should never reach here
    raise APIError(provider, "RETRY_EXHAUSTED", "All retry attempts failed", should_retry=False)


class ProjectComplexity(Enum):
    """Project complexity levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ProjectDomain(Enum):
    """Software project domains with sub-categories for uniqueness"""
    # Web Applications (6 subcategories)
    WEB_ECOMMERCE = "web_ecommerce"
    WEB_SOCIAL = "web_social"
    WEB_CMS = "web_cms"
    WEB_DASHBOARD = "web_dashboard"
    WEB_BLOG = "web_blog"
    WEB_PORTFOLIO = "web_portfolio"
    
    # API Services (4 subcategories)  
    API_REST = "api_rest"
    API_GRAPHQL = "api_graphql"
    API_MICROSERVICE = "api_microservice"
    API_GATEWAY = "api_gateway"
    
    # Data Systems (5 subcategories)
    DATA_ANALYTICS = "data_analytics"
    DATA_ETL = "data_etl"
    DATA_WAREHOUSE = "data_warehouse"
    DATA_STREAMING = "data_streaming"
    DATA_LAKE = "data_lake"
    
    # ML/AI (4 subcategories)
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    ML_NLP = "ml_nlp"
    ML_COMPUTER_VISION = "ml_computer_vision"
    
    # Desktop Apps (3 subcategories)
    DESKTOP_PRODUCTIVITY = "desktop_productivity"
    DESKTOP_MEDIA = "desktop_media"
    DESKTOP_DEVELOPMENT = "desktop_development"
    
    # Mobile Apps (3 subcategories)
    MOBILE_SOCIAL = "mobile_social"
    MOBILE_UTILITY = "mobile_utility"
    MOBILE_GAME = "mobile_game"
    
    # Systems/Infrastructure (4 subcategories)
    SYSTEM_MONITORING = "system_monitoring"
    SYSTEM_AUTOMATION = "system_automation"
    SYSTEM_NETWORKING = "system_networking"
    SYSTEM_SECURITY = "system_security"
    
    # Finance/Business (3 subcategories)
    FINTECH_PAYMENT = "fintech_payment"
    FINTECH_TRADING = "fintech_trading"
    FINTECH_BANKING = "fintech_banking"
    
    # Gaming (2 subcategories)
    GAME_ENGINE = "game_engine"
    GAME_SIMULATION = "game_simulation"
    
    # Blockchain (2 subcategories)
    BLOCKCHAIN_DEFI = "blockchain_defi"
    BLOCKCHAIN_NFT = "blockchain_nft"


class ProjectArchitecture(Enum):
    """Architecture patterns for additional uniqueness"""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    CLEAN_ARCHITECTURE = "clean_architecture"
    HEXAGONAL = "hexagonal"
    MVC = "mvc"
    MVVM = "mvvm"
    COMPONENT_BASED = "component_based"


class ProjectTheme(Enum):
    """Project themes for additional variation"""
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    PRODUCTIVITY = "productivity"
    SOCIAL = "social"
    UTILITY = "utility"
    CREATIVE = "creative"


@dataclass
class ProjectSpecification:
    """Specification for a synthetic project"""
    unique_id: str                        # Guaranteed unique identifier
    name: str
    description: str
    domain: ProjectDomain
    complexity: ProjectComplexity
    language: str
    architecture: ProjectArchitecture     # Architecture pattern
    theme: ProjectTheme                   # Project theme
    target_file_count: int
    target_token_count: int
    features: List[str]
    architecture_patterns: List[str]
    dependencies: List[str]
    seed: int                            # Deterministic seed for LLM variation
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enums to strings for JSON serialization
        data['domain'] = self.domain.value
        data['complexity'] = self.complexity.value
        data['architecture'] = self.architecture.value
        data['theme'] = self.theme.value
        return data


@dataclass 
class GeneratedFile:
    """A generated source code file"""
    path: str
    content: str
    file_type: str  # 'source', 'config', 'test', 'documentation'
    dependencies: List[str]
    complexity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticProject:
    """A complete synthetic software project"""
    specification: ProjectSpecification
    files: List[GeneratedFile]
    file_structure: Dict[str, Any]  # Directory tree structure
    architecture_overview: str
    setup_instructions: str
    test_scenarios: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'specification': self.specification.to_dict(),
            'files': [f.to_dict() for f in self.files],
            'file_structure': self.file_structure,
            'architecture_overview': self.architecture_overview,
            'setup_instructions': self.setup_instructions,
            'test_scenarios': self.test_scenarios
        }


class MultiLLMGenerator:
    """Multi-LLM system for generating diverse, high-quality synthetic projects"""
    
    def __init__(self, config: Config, log_file: str = None):
        self.config = config
        
        # Setup logging
        self.logger = setup_generation_logging(log_file)
        self.logger.info("🚀 MultiLLMGenerator initialized")
        
        self.rate_limiter = APIRateLimitManager(config)
        self._openai_http_client: Optional[httpx.AsyncClient] = None
        self._custom_http_client: Optional[httpx.AsyncClient] = None
        self.custom_openai_client: Optional[openai.AsyncOpenAI] = None
        self._custom_client_signature: Optional[Tuple[str, str]] = None
        self.setup_llm_clients()
        
        # Generator specialization (using 2 Elite Models)
        # ✅ OpenAI o3: Best reasoning model | ✅ Gemini 2.5 Pro: Reliable with 1M+ token limits
        self.generators = {
            "requirements": "openai",      # OpenAI o3 - Best for structured requirements
            "architecture": "google",      # Gemini 2.5 Pro - Excellent at system design with generous limits
            "implementation": "openai",    # OpenAI o3 - Strong at code generation
            "scenarios": "google",         # Gemini 2.5 Pro - Good at realistic scenarios with large context
            "validation": "google"         # Gemini 2.5 Pro - Reliable validation
        }
    
    def setup_llm_clients(self):
        """Initialize LLM API clients (OpenAI o3 + Gemini 2.5 Pro + Claude 4 via Bearer Token + Hugging Face)"""
        # OpenAI-compatible client (supports custom base URL & timeout)
        openai_kwargs: Dict[str, Any] = {}

        if self.config.api.openai_api_key:
            openai_kwargs["api_key"] = self.config.api.openai_api_key
        if self.config.api.openai_base_url:
            openai_kwargs["base_url"] = self.config.api.openai_base_url.rstrip("/")
        if self.config.api.openai_timeout:
            openai_kwargs["timeout"] = self.config.api.openai_timeout
        if self.config.api.disable_proxy:
            if self._openai_http_client is None:
                self._openai_http_client = httpx.AsyncClient(trust_env=False)
            openai_kwargs["http_client"] = self._openai_http_client

        self.openai_client = openai.AsyncOpenAI(**openai_kwargs)
        if self.config.api.openai_base_url:
            self.logger.info(f"🔗 OpenAI base URL configured: {self.config.api.openai_base_url}")
        if self.config.api.disable_proxy:
            self.logger.info("🚫 Proxy usage disabled for OpenAI-compatible clients")
        
        # Gemini 2.5 Pro
        genai.configure(api_key=self.config.api.google_api_key)
        
        # Claude Bearer Token (no additional setup needed - used directly in API calls)
        # Verification happens in generate_with_claude method
        
        # Hugging Face models - lazy loading (loaded on first use)
        self.hf_models = {}
        self.hf_tokenizers = {}
        
        if HF_AVAILABLE:
            self.logger.info("✅ 4-Model generator initialized (OpenAI o3 + Gemini 2.5 Pro + Claude 4 + Hugging Face)")
        else:
            self.logger.info("✅ 3-Elite-Model generator initialized (OpenAI o3 + Gemini 2.5 Pro + Claude 4 Bearer Token)")
            self.logger.warning("⚠️ Hugging Face models not available (transformers/torch not installed)")
    
    async def generate_with_openai(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using OpenAI with retry logic and rate limiting"""
        
        async def _make_openai_call():
            if not self.config.api.openai_api_key:
                raise APIError("OpenAI", "AUTH_FAILED", "OpenAI API key not configured")
            
            self.logger.info(f"🤖 Making OpenAI call, model: {self.config.api.default_model_openai}")
            self.logger.info(f"📝 Prompt length: {len(prompt)} chars, System prompt: {len(system_prompt) if system_prompt else 0} chars")
            
            # Apply rate limiting
            async with await self.rate_limiter.acquire("openai"):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                request_kwargs: Dict[str, Any] = {}
                if self.config.api.openai_timeout:
                    request_kwargs["timeout"] = self.config.api.openai_timeout

                # Handle different OpenAI models with appropriate token limits
                if self.config.api.default_model_openai.startswith(("o1", "o3", "o4")):
                    self.logger.info(f"🔧 Using o-series format with max_completion_tokens=100000")
                    response = await self.openai_client.chat.completions.create(
                        model=self.config.api.default_model_openai,
                        messages=messages,
                        max_completion_tokens=100000,  # o-series supports 100K+, maximizing for comprehensive generation
                        **request_kwargs
                    )
                elif self.config.api.default_model_openai.startswith("gpt-5"):
                    self.logger.info(f"🔧 Using GPT-5 format with max_completion_tokens=50000")
                    response = await self.openai_client.chat.completions.create(
                        model=self.config.api.default_model_openai,
                        messages=messages,
                        max_completion_tokens=50000,  # GPT-5 series optimized generation limit
                        # Note: GPT-5 only supports default temperature (1.0), omitting temperature parameter
                        **request_kwargs
                    )
                elif self.config.api.default_model_openai.startswith(("custom", "gpt-oss-120b")):
                    target_model = self.config.api.default_model_openai
                    if target_model.startswith("custom:"):
                        target_model = target_model.split(":", 1)[1] or self.config.api.custom_model_name or target_model
                    elif target_model == "custom":
                        target_model = self.config.api.custom_model_name or target_model

                    # Allow fallback if config specifies dedicated custom name
                    if target_model == "gpt-oss-120b" and self.config.api.custom_model_name:
                        target_model = self.config.api.custom_model_name

                    max_tokens = self.config.api.custom_model_max_tokens or 8192
                    temperature = self.config.api.custom_model_temperature if self.config.api.custom_model_temperature is not None else 0.1
                    custom_request_kwargs = dict(request_kwargs)
                    custom_timeout = self.config.api.custom_model_timeout or self.config.api.openai_timeout
                    if custom_timeout:
                        custom_request_kwargs["timeout"] = custom_timeout

                    self.logger.info(f"🔧 Using custom format with max_tokens={max_tokens}")
                    response = await self.openai_client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **custom_request_kwargs
                    )
                elif self.config.api.default_model_openai.startswith(("gpt-4o", "gpt-4-turbo")):
                    self.logger.info(f"🔧 Using GPT-4o/turbo format with max_tokens=16384")
                    response = await self.openai_client.chat.completions.create(
                        model=self.config.api.default_model_openai,
                        messages=messages,
                        max_tokens=16384,  # GPT-4o/turbo max limit
                        temperature=0.7,
                        **request_kwargs
                    )
                elif self.config.api.default_model_openai.startswith("gpt-4"):
                    self.logger.info(f"🔧 Using GPT-4 format with max_tokens=8192")  
                    response = await self.openai_client.chat.completions.create(
                        model=self.config.api.default_model_openai,
                        messages=messages,
                        max_tokens=8192,  # GPT-4 standard limit
                        temperature=0.7,
                        **request_kwargs
                    )
                else:
                    self.logger.info(f"🔧 Using standard format with max_tokens=4096")
                    response = await self.openai_client.chat.completions.create(
                        model=self.config.api.default_model_openai,
                        messages=messages,
                        max_tokens=4096,  # Conservative default
                        temperature=0.7,
                        **request_kwargs
                    )
                
                content = response.choices[0].message.content
                self.logger.info(f"📤 OpenAI response length: {len(content) if content else 0} chars")
                if content is None:
                    self.logger.warning(f"⚠️ OpenAI returned None content")
                    raise APIError("OpenAI", "EMPTY_RESPONSE", "OpenAI returned empty content")
                elif content.strip() == "":
                    self.logger.warning(f"⚠️ OpenAI returned empty string")
                    raise APIError("OpenAI", "EMPTY_RESPONSE", "OpenAI returned empty content")
                else:
                    self.logger.info(f"✅ OpenAI returned valid content: {content[:100]}...")
                return content
        
        return await retry_with_backoff(_make_openai_call, provider="OpenAI o3")
    
    async def generate_with_custom_model(self, prompt: str, system_prompt: str = None, model_override: Optional[str] = None) -> str:
        """Generate content using a custom OpenAI-compatible endpoint"""

        target_model = model_override or self.config.api.custom_model_name
        if not target_model:
            raise APIError("Custom", "CONFIG_ERROR", "Custom model name not configured. Set api.custom_model_name or CUSTOM_MODEL_NAME env.")

        base_url = (self.config.api.custom_model_base_url or self.config.api.openai_base_url)
        if not base_url:
            raise APIError("Custom", "CONFIG_ERROR", "Custom model base URL not configured. Set api.custom_model_base_url or OPENAI_BASE_URL env.")
        base_url = base_url.rstrip("/")

        api_key = self.config.api.custom_model_api_key or self.config.api.openai_api_key
        if not api_key:
            raise APIError("Custom", "AUTH_FAILED", "Custom model API key not configured. Set api.custom_model_api_key or OPENAI_API_KEY env.")

        client_timeout = self.config.api.custom_model_client_timeout or self.config.api.openai_timeout
        request_timeout = self.config.api.custom_model_timeout or self.config.api.openai_timeout
        signature = (base_url, api_key)

        if self.custom_openai_client is None or self._custom_client_signature != signature:
            if self._custom_http_client is not None:
                try:
                    await self._custom_http_client.aclose()
                except Exception as close_error:
                    self.logger.debug(f"⚠️ Failed to close previous custom HTTP client: {close_error}")
                finally:
                    self._custom_http_client = None

            client_kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url,
            }

            if client_timeout:
                client_kwargs["timeout"] = client_timeout

            if self.config.api.disable_proxy:
                self._custom_http_client = httpx.AsyncClient(trust_env=False)
                client_kwargs["http_client"] = self._custom_http_client

            self.custom_openai_client = openai.AsyncOpenAI(**client_kwargs)
            self._custom_client_signature = signature
            self.logger.info(f"🔌 Custom model client initialized for {target_model} @ {base_url}")
        else:
            self.logger.debug(f"♻️ Reusing cached custom client for {target_model}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _make_custom_call():
            async with await self.rate_limiter.acquire("custom"):
                response = await self.custom_openai_client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    max_tokens=self.config.api.custom_model_max_tokens,
                    temperature=self.config.api.custom_model_temperature,
                    timeout=request_timeout
                )

            content = response.choices[0].message.content if response.choices else None
            if content is None or content.strip() == "":
                raise APIError("Custom", "EMPTY_RESPONSE", f"Custom model {target_model} returned empty content")

            self.logger.info(f"📤 Custom model response length: {len(content)} chars")
            return content

        try:
            return await retry_with_backoff(_make_custom_call, provider=f"Custom {target_model}")
        except APIError:
            raise
        except asyncio.TimeoutError as e:
            raise APIError("Custom", "TIMEOUT", f"Request timeout for custom model {target_model}: {str(e)}") from e
        except Exception as e:
            raise APIError("Custom", "GENERATION_ERROR", f"Custom model {target_model} error: {str(e)}", original_error=e) from e


    async def generate_with_google(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content using Gemini 2.5 Pro with retry logic and rate limiting"""
        
        async def _make_google_call():
            if not self.config.api.google_api_key:
                raise APIError("Gemini 2.5 Pro", "AUTH_FAILED", "Google API key not configured")
            
            # Apply rate limiting
            async with await self.rate_limiter.acquire("google"):
                # Configure generation parameters for high-quality code generation
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500000,  # Maximized for Gemini 2.5 Pro (supports 1M+)
                    top_p=0.95,
                    top_k=40
                )
                
                # Ensure proper model name format for Gemini API
                model_name = self.config.api.default_model_google
                if not model_name.startswith('models/'):
                    model_name = f"models/{model_name}"
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # Run Gemini call in thread pool to avoid blocking the event loop
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, model.generate_content, full_prompt)
                content = response.text
                if content is None:
                    raise APIError("Gemini 2.5 Pro", "EMPTY_RESPONSE", "Gemini returned empty content")
                return content
        
        return await retry_with_backoff(_make_google_call, provider="Gemini 2.5 Pro")
    
    async def generate_with_claude(self, prompt: str, model_name: str = "claude-sonnet-4", system_prompt: str = None) -> str:
        """Generate content using Claude models via Bearer Token authentication"""
        
        async def _make_claude_call():
            """Direct Claude API call using Bearer token authentication"""
            if not self.config.api.claude_bearer_token:
                raise APIError("Claude", "AUTH_FAILED", "Claude Bearer Token not configured")
            
            # Import here to avoid dependency issues
            import json
            import aiohttp
            from ..core.config import get_claude_model_id, get_model_max_tokens
            
            # Convert human-friendly name to Claude model ID
            claude_model_id = get_claude_model_id(model_name)
            max_tokens = get_model_max_tokens(model_name)
            
            # Prepare the request body using Claude API format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0.5,
                "messages": []
            }
            
            # Add system prompt if provided
            if system_prompt:
                body["system"] = system_prompt
            
            # Add user message
            body["messages"].append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.config.api.claude_bearer_token}",
                "Content-Type": "application/json"
            }
            
            # Construct URL
            url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{claude_model_id}/invoke"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=600)) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            
                            if 'content' in response_data and response_data['content']:
                                content = response_data['content'][0]['text']
                                if content is None or content.strip() == "":
                                    raise APIError("Claude", "EMPTY_RESPONSE", f"Claude {model_name} returned empty content")
                                return content
                            else:
                                raise APIError("Claude", "INVALID_RESPONSE", f"Claude {model_name} returned invalid response format")
                        else:
                            error_text = await response.text()
                            # Try to parse error JSON
                            try:
                                error_data = await response.json()
                                error_message = error_data.get('message', error_text)
                            except:
                                error_message = error_text
                            
                            # Ensure we have a meaningful error message
                            if not error_message or error_message.strip() == "":
                                error_message = f"HTTP {response.status} error with empty response"
                            
                            if response.status == 401:
                                raise APIError("Claude", "AUTH_FAILED", f"Authentication failed for Claude {model_name}: {error_message}")
                            elif response.status == 429:
                                raise APIError("Claude", "RATE_LIMIT", f"Rate limit exceeded for Claude {model_name}: {error_message}")
                            elif response.status == 400:
                                raise APIError("Claude", "VALIDATION_ERROR", f"Validation error for Claude {model_name}: {error_message}")
                            else:
                                raise APIError("Claude", "BEARER_TOKEN_ERROR", f"Claude Bearer Token error for {model_name}: {error_message}")
                                
            except aiohttp.ClientError as e:
                raise APIError("Claude", "CONNECTION_ERROR", f"Connection error for Claude {model_name}: {str(e)}")
            except asyncio.TimeoutError as e:
                raise APIError("Claude", "TIMEOUT_ERROR", f"Request timeout for Claude {model_name}: {str(e)}")
            except Exception as e:
                error_message = str(e)
                if "APIError" in str(type(e)):
                    raise  # Re-raise APIError as-is
                else:
                    # Provide more detailed error information
                    error_details = f"Exception type: {type(e).__name__}, Message: {error_message}"
                    if not error_message or error_message.strip() == "":
                        error_details = f"Empty exception of type {type(e).__name__}"
                    raise APIError("Claude", "BEARER_TOKEN_ERROR", f"Unexpected error for Claude {model_name}: {error_details}")
        
        # Apply rate limiting and retry logic (reduced retries for Bearer token)
        async with await self.rate_limiter.acquire("claude"):
            return await retry_with_backoff(_make_claude_call, max_retries=2, base_delay=3.0, max_delay=60.0, provider=f"Claude {model_name}")
    
    async def generate_with_huggingface(self, model_name: str, prompt: str, system_prompt: str = None) -> str:
        """Generate content using Hugging Face models (local inference)"""
        if not HF_AVAILABLE:
            raise APIError("HuggingFace", "NOT_AVAILABLE", "transformers/torch not installed. Install with: pip install transformers torch")
        
        async def _make_hf_call():
            try:
                # Lazy load model if not already loaded
                if model_name not in self.hf_models:
                    self.logger.info(f"📦 Loading Hugging Face model: {model_name}")
                    
                    # Use CPU if CUDA not available
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # Load tokenizer
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    except Exception as e:
                        # Some models may need padding token
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                    
                    # Load model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        low_cpu_mem_usage=True
                    )
                    
                    if device == "cpu":
                        model = model.to(device)
                    
                    self.hf_models[model_name] = model
                    self.hf_tokenizers[model_name] = tokenizer
                    self.logger.info(f"✅ Loaded Hugging Face model: {model_name} on {device}")
                
                model = self.hf_models[model_name]
                tokenizer = self.hf_tokenizers[model_name]
                device = next(model.parameters()).device
                
                # Prepare input text
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt

                self.logger.info(f"🤖 Generating with Hugging Face model: {model_name}")
                self.logger.info(f"📝 Prompt length: {len(full_prompt)} chars")

                def _normalize_length(value: int) -> Optional[int]:
                    if value is None:
                        return None
                    if not isinstance(value, int):
                        try:
                            value = int(value)
                        except (TypeError, ValueError):
                            return None
                    if value <= 0 or value > 1_000_000:
                        return None
                    return value

                max_position_embeddings = _normalize_length(getattr(model.config, "max_position_embeddings", None))
                tokenizer_max_length = _normalize_length(getattr(tokenizer, "model_max_length", None))

                candidate_lengths = [length for length in [max_position_embeddings, tokenizer_max_length] if length]
                max_prompt_length = min(candidate_lengths) if candidate_lengths else 4096
                reserve_fraction = 0.3
                calculated_reserve = int(max_prompt_length * reserve_fraction)
                output_reserve = max(calculated_reserve, 128)
                output_reserve = min(output_reserve, max_prompt_length // 2)
                effective_prompt_limit = max_prompt_length - output_reserve
                if effective_prompt_limit < 256:
                    effective_prompt_limit = max(max_prompt_length - 128, max_prompt_length // 2, 64)
                    output_reserve = max_prompt_length - effective_prompt_limit
                effective_prompt_limit = min(effective_prompt_limit, max_prompt_length)
                output_reserve = max(max_prompt_length - effective_prompt_limit, 0)

                # Tokenize input
                inputs = tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=effective_prompt_limit,
                    padding=False
                ).to(device)

                input_token_count = inputs.input_ids.shape[-1]
                total_limit = max_prompt_length
                available_new_tokens = max(total_limit - input_token_count, 0)
                if available_new_tokens <= 0:
                    fallback_new_tokens = max(max_prompt_length // 2, 16)
                    max_new_tokens = min(fallback_new_tokens, 4096)
                else:
                    max_new_tokens = min(available_new_tokens, 4096)
                max_new_tokens = max(16, max_new_tokens)

                self.logger.info(
                    f"🔢 Tokens | prompt: {input_token_count}, reserve: {output_reserve}, max_new: {max_new_tokens}, limit: {total_limit}"
                )

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the newly generated part (after the prompt)
                if full_prompt in generated_text:
                    response = generated_text.split(full_prompt, 1)[1].strip()
                else:
                    # If prompt is not found, return the end part (likely the generated content)
                    response = generated_text[len(full_prompt):].strip()
                
                output_token_count = outputs[0].shape[-1]
                generated_token_count = max(output_token_count - input_token_count, 0)
                self.logger.info(f"📤 Hugging Face response length: {len(response)} chars")
                self.logger.info(
                    f"🧮 Tokens | generated: {generated_token_count}, total: {output_token_count}"
                )
                
                if not response or len(response.strip()) < 10:
                    raise APIError("HuggingFace", "EMPTY_RESPONSE", f"Model {model_name} returned empty or very short response")
                
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "cuda" in error_str:
                    raise APIError("HuggingFace", "OUT_OF_MEMORY", f"GPU/CPU memory error for {model_name}: {str(e)}")
                elif "no such file" in error_str or "not found" in error_str:
                    raise APIError("HuggingFace", "MODEL_NOT_FOUND", f"Model {model_name} not found on Hugging Face Hub: {str(e)}")
                else:
                    raise APIError("HuggingFace", "GENERATION_ERROR", f"Error generating with {model_name}: {str(e)}", original_error=e)
        
        # Apply rate limiting (lower rate for local models)
        async with await self.rate_limiter.acquire("huggingface"):
            return await retry_with_backoff(_make_hf_call, max_retries=2, base_delay=1.0, max_delay=30.0, provider=f"HuggingFace {model_name}")
    
    async def generate_with_model(self, model_type: str, prompt: str, system_prompt: str = None) -> str:
        """Generate content with specified model type (OpenAI o3, Gemini 2.5 Pro, Claude, or Hugging Face)"""
        try:
            if model_type == "openai":
                return await self.generate_with_openai(prompt, system_prompt)
            elif model_type == "google":
                return await self.generate_with_google(prompt, system_prompt)
            elif model_type == "custom":
                return await self.generate_with_custom_model(prompt, system_prompt)
            elif model_type.startswith("custom:"):
                override_model = model_type.split(":", 1)[1].strip()
                return await self.generate_with_custom_model(
                    prompt,
                    system_prompt,
                    model_override=override_model or None
                )
            elif model_type == "claude":
                # Default to Claude Sonnet 4 for balanced performance
                return await self.generate_with_claude(prompt, "claude-sonnet-4", system_prompt)
            elif model_type.startswith("claude-"):
                # Support direct Claude model specification
                return await self.generate_with_claude(prompt, model_type, system_prompt)
            elif model_type.startswith("huggingface:") or model_type.startswith("hf:"):
                # Hugging Face model: format is "huggingface:model-name" or "hf:model-name"
                model_name = model_type.split(":", 1)[1] if ":" in model_type else model_type
                return await self.generate_with_huggingface(model_name, prompt, system_prompt)
            elif "/" in model_type and HF_AVAILABLE:
                # Assume it's a Hugging Face model ID if it contains "/"
                return await self.generate_with_huggingface(model_type, prompt, system_prompt)
            else:
                supported = "'openai', 'google', 'custom', 'claude', 'claude-sonnet-4', 'claude-opus-4', 'claude-sonnet-3.7'"
                if HF_AVAILABLE:
                    supported += ", 'huggingface:model-name', 'hf:model-name', or any Hugging Face model ID"
                raise ValueError(f"Unknown model type: {model_type}. Supported: {supported}")
        except APIError as e:
            # Re-raise APIError with additional context about model assignment
            raise APIError(
                provider=e.provider,
                error_type=e.error_type,
                message=f"Model assignment '{model_type}' failed: {e.message}",
                original_error=e.original_error
            )
        except Exception as e:
            # Convert unexpected errors to APIError
            raise APIError(
                provider=f"{model_type.title()}",
                error_type="UNEXPECTED_ERROR",
                message=f"Unexpected error in {model_type}: {str(e)}",
                original_error=e
            )


class ProjectTemplateManager:
    """Manages project templates and domain-specific patterns"""
    
    def __init__(self):
        # Define base templates for major categories
        self.base_templates = {
            "web": {
                "features": [
                    "user_authentication", "database_integration", "api_endpoints",
                    "frontend_interface", "session_management", "data_validation",
                    "email_notifications", "file_upload", "search_functionality",
                    "admin_panel", "logging_system", "error_handling", "responsive_design",
                    "payment_processing", "social_login", "caching", "ssl_security"
                ],
                "patterns": [
                    "MVC", "REST_API", "Database_ORM", "Authentication_Middleware",
                    "Component_Architecture", "Service_Layer", "Repository_Pattern"
                ],
                "file_types": [
                    "controllers", "models", "views", "routes", "middleware",
                    "services", "utils", "config", "tests", "static", "templates"
                ]
            },
            "api": {
                "features": [
                    "rest_endpoints", "graphql_schema", "authentication", "rate_limiting",
                    "request_validation", "response_caching", "api_documentation",
                    "error_handling", "logging", "monitoring", "versioning", "pagination"
                ],
                "patterns": [
                    "REST_Architecture", "GraphQL_Schema", "Microservices", "API_Gateway",
                    "Repository_Pattern", "Service_Layer", "Command_Query_Separation"
                ],
                "file_types": [
                    "endpoints", "schemas", "models", "validators", "middleware",
                    "services", "utils", "config", "tests", "docs"
                ]
            },
            "data": {
                "features": [
                    "data_ingestion", "data_transformation", "data_validation",
                    "batch_processing", "stream_processing", "data_storage",
                    "monitoring", "error_recovery", "data_quality_checks",
                    "scheduling", "parallel_processing", "data_visualization"
                ],
                "patterns": [
                    "ETL_Pipeline", "Event_Streaming", "Data_Lake", "Microservices",
                    "Observer_Pattern", "Strategy_Pattern", "Pipeline_Pattern"
                ],
                "file_types": [
                    "extractors", "transformers", "loaders", "validators",
                    "schedulers", "monitors", "config", "tests", "pipelines"
                ]
            },
            "ml": {
                "features": [
                    "data_preprocessing", "feature_engineering", "model_training",
                    "model_evaluation", "hyperparameter_tuning", "model_serving",
                    "experiment_tracking", "data_visualization", "model_monitoring",
                    "automated_retraining", "feature_store", "model_versioning"
                ],
                "patterns": [
                    "Pipeline_Pattern", "Factory_Pattern", "Strategy_Pattern",
                    "Observer_Pattern", "MLOps_Architecture", "Model_Registry"
                ],
                "file_types": [
                    "data", "features", "models", "training", "evaluation",
                    "serving", "utils", "config", "tests", "experiments"
                ]
            },
            "desktop": {
                "features": [
                    "gui_interface", "file_management", "settings_configuration",
                    "plugin_system", "auto_updates", "crash_reporting", "user_preferences",
                    "keyboard_shortcuts", "drag_drop", "multi_window", "themes"
                ],
                "patterns": [
                    "MVC", "MVVM", "Observer_Pattern", "Command_Pattern", "Plugin_Architecture",
                    "Event_Driven", "State_Machine"
                ],
                "file_types": [
                    "views", "controllers", "models", "plugins", "resources",
                    "configs", "utils", "tests", "assets", "localization"
                ]
            },
            "mobile": {
                "features": [
                    "responsive_ui", "offline_sync", "push_notifications", "location_services",
                    "camera_integration", "local_storage", "biometric_auth", "social_sharing",
                    "in_app_purchases", "analytics", "crash_reporting"
                ],
                "patterns": [
                    "MVVM", "Repository_Pattern", "Observer_Pattern", "Singleton",
                    "Factory_Pattern", "Adapter_Pattern"
                ],
                "file_types": [
                    "views", "viewmodels", "models", "services", "repositories",
                    "utils", "config", "tests", "resources", "assets"
                ]
            },
            "system": {
                "features": [
                    "system_monitoring", "log_aggregation", "performance_metrics",
                    "alerting", "configuration_management", "deployment_automation",
                    "security_scanning", "backup_recovery", "load_balancing"
                ],
                "patterns": [
                    "Observer_Pattern", "Strategy_Pattern", "Command_Pattern",
                    "Chain_of_Responsibility", "Service_Mesh", "Event_Driven"
                ],
                "file_types": [
                    "monitors", "collectors", "processors", "alerts", "configs",
                    "scripts", "utils", "tests", "dashboards"
                ]
            },
            "fintech": {
                "features": [
                    "payment_processing", "transaction_management", "fraud_detection",
                    "compliance_reporting", "risk_assessment", "encryption", "audit_logging",
                    "multi_currency", "settlement", "kyc_verification", "regulatory_compliance"
                ],
                "patterns": [
                    "Saga_Pattern", "Event_Sourcing", "CQRS", "Microservices",
                    "Security_by_Design", "Audit_Trail"
                ],
                "file_types": [
                    "transactions", "payments", "security", "compliance", "audit",
                    "models", "services", "utils", "config", "tests"
                ]
            },
            "gaming": {
                "features": [
                    "game_engine", "physics_simulation", "graphics_rendering", "audio_system",
                    "input_handling", "ai_behavior", "networking", "save_system",
                    "resource_management", "scripting_system", "level_editor"
                ],
                "patterns": [
                    "Entity_Component_System", "Game_Loop", "State_Machine",
                    "Observer_Pattern", "Object_Pool", "Command_Pattern"
                ],
                "file_types": [
                    "engine", "physics", "graphics", "audio", "input", "ai",
                    "network", "scripts", "assets", "config", "tests"
                ]
            },
            "blockchain": {
                "features": [
                    "smart_contracts", "wallet_integration", "transaction_processing",
                    "consensus_mechanism", "cryptographic_functions", "token_management",
                    "defi_protocols", "nft_minting", "governance", "staking"
                ],
                "patterns": [
                    "Event_Driven", "Factory_Pattern", "Proxy_Pattern",
                    "State_Machine", "Observer_Pattern", "Strategy_Pattern"
                ],
                "file_types": [
                    "contracts", "tokens", "protocols", "wallets", "crypto",
                    "governance", "utils", "config", "tests", "migrations"
                ]
            }
        }
        
        # Map each domain subcategory to its base template
        self.domain_mapping = {
            # Web Applications
            ProjectDomain.WEB_ECOMMERCE: "web",
            ProjectDomain.WEB_SOCIAL: "web", 
            ProjectDomain.WEB_CMS: "web",
            ProjectDomain.WEB_DASHBOARD: "web",
            ProjectDomain.WEB_BLOG: "web",
            ProjectDomain.WEB_PORTFOLIO: "web",
            
            # API Services
            ProjectDomain.API_REST: "api",
            ProjectDomain.API_GRAPHQL: "api",
            ProjectDomain.API_MICROSERVICE: "api",
            ProjectDomain.API_GATEWAY: "api",
            
            # Data Systems
            ProjectDomain.DATA_ANALYTICS: "data",
            ProjectDomain.DATA_ETL: "data",
            ProjectDomain.DATA_WAREHOUSE: "data",
            ProjectDomain.DATA_STREAMING: "data",
            ProjectDomain.DATA_LAKE: "data",
            
            # ML/AI Systems
            ProjectDomain.ML_TRAINING: "ml",
            ProjectDomain.ML_INFERENCE: "ml",
            ProjectDomain.ML_NLP: "ml",
            ProjectDomain.ML_COMPUTER_VISION: "ml",
            
            # Desktop Applications
            ProjectDomain.DESKTOP_PRODUCTIVITY: "desktop",
            ProjectDomain.DESKTOP_MEDIA: "desktop",
            ProjectDomain.DESKTOP_DEVELOPMENT: "desktop",
            
            # Mobile Applications
            ProjectDomain.MOBILE_SOCIAL: "mobile",
            ProjectDomain.MOBILE_UTILITY: "mobile",
            ProjectDomain.MOBILE_GAME: "mobile",
            
            # System Infrastructure
            ProjectDomain.SYSTEM_MONITORING: "system",
            ProjectDomain.SYSTEM_AUTOMATION: "system",
            ProjectDomain.SYSTEM_NETWORKING: "system",
            ProjectDomain.SYSTEM_SECURITY: "system",
            
            # Financial Technology
            ProjectDomain.FINTECH_PAYMENT: "fintech",
            ProjectDomain.FINTECH_TRADING: "fintech",
            ProjectDomain.FINTECH_BANKING: "fintech",
            
            # Gaming & Simulation
            ProjectDomain.GAME_ENGINE: "gaming",
            ProjectDomain.GAME_SIMULATION: "gaming",
            
            # Blockchain Systems
            ProjectDomain.BLOCKCHAIN_DEFI: "blockchain",
            ProjectDomain.BLOCKCHAIN_NFT: "blockchain"
        }
    
    def get_template(self, domain: ProjectDomain, complexity: ProjectComplexity) -> Dict[str, Any]:
        """Get project template based on domain and complexity"""
        # Map the specific domain to its base template category
        template_category = self.domain_mapping.get(domain, "web")  # Default to web if not found
        base_template = self.base_templates[template_category]
        
        # Adjust template based on complexity
        complexity_multipliers = {
            ProjectComplexity.EASY: 0.5,
            ProjectComplexity.MEDIUM: 0.7,
            ProjectComplexity.HARD: 0.9,
            ProjectComplexity.EXPERT: 1.0
        }
        
        multiplier = complexity_multipliers[complexity]
        
        return {
            "features": random.sample(
                base_template["features"], 
                max(3, int(len(base_template["features"]) * multiplier))
            ),
            "patterns": random.sample(
                base_template["patterns"],
                max(2, int(len(base_template["patterns"]) * multiplier))
            ),
            "file_types": base_template["file_types"]
        }


class SyntheticProjectGenerator:
    """Main synthetic project generator"""
    
    def __init__(self, config: Config, log_file: str = None):
        self.config = config
        self.llm_generator = MultiLLMGenerator(config, log_file)
        self.template_manager = ProjectTemplateManager()
        
        # Create output directories
        self.generated_dir = Path(config.data.generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_project_specification(
        self, 
        domain: ProjectDomain, 
        complexity: ProjectComplexity,
        language: str
    ) -> ProjectSpecification:
        """Generate a detailed project specification"""
        
        template = self.template_manager.get_template(domain, complexity)
        
        # Calculate target metrics based on complexity and config constraints
        complexity_ranges = {
            ProjectComplexity.EASY: {
                "files": (self.config.phase2.min_files_per_project, min(self.config.phase2.max_files_per_project, 15)),
                "tokens": tuple(self.config.phase3.context_ranges["easy"])
            },
            ProjectComplexity.MEDIUM: {
                "files": (max(15, self.config.phase2.min_files_per_project), min(self.config.phase2.max_files_per_project, 40)),
                "tokens": tuple(self.config.phase3.context_ranges["medium"])
            },
            ProjectComplexity.HARD: {
                "files": (max(40, self.config.phase2.min_files_per_project), min(self.config.phase2.max_files_per_project, 80)),
                "tokens": tuple(self.config.phase3.context_ranges["hard"])
            },
            ProjectComplexity.EXPERT: {
                "files": (max(80, self.config.phase2.min_files_per_project), self.config.phase2.max_files_per_project),
                "tokens": tuple(self.config.phase3.context_ranges["expert"])
            }
        }
        
        ranges = complexity_ranges[complexity]
        # Ensure file count respects config constraints
        min_files = max(ranges["files"][0], self.config.phase2.min_files_per_project)
        max_files = min(ranges["files"][1], self.config.phase2.max_files_per_project)
        target_file_count = random.randint(min_files, max_files)
        target_token_count = random.randint(*ranges["tokens"])
        
        # Generate specification using LLM
        prompt = f"""
        Generate a detailed specification for a {complexity.value} {domain.value} project in {language}.
        
        Requirements:
        - Domain: {domain.value}
        - Complexity: {complexity.value}
        - Target files: {target_file_count}
        - Target tokens: {target_token_count}
        - Features to include: {', '.join(template['features'])}
        - Architecture patterns: {', '.join(template['patterns'])}
        
        Please provide:
        1. Project name (creative but realistic)
        2. Detailed description (2-3 paragraphs)
        3. List of main dependencies/libraries
        4. Brief explanation of why this complexity level is appropriate
        
        Return as JSON with keys: name, description, dependencies, complexity_justification
        """
        
        system_prompt = """You are a senior software architect specializing in designing realistic software projects for evaluation purposes. Generate specifications that would represent real-world projects of the specified complexity."""
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["requirements"],
            prompt,
            system_prompt
        )
        
        try:
            spec_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            spec_data = {
                "name": f"{domain.value.replace('_', ' ').title()} Project",
                "description": f"A {complexity.value} {domain.value} implementation in {language}",
                "dependencies": [],
                "complexity_justification": f"Designed for {complexity.value} complexity level"
            }
        
        return ProjectSpecification(
            unique_id=f"{domain.value}-{complexity.value}-{language}-{random.randint(1000, 9999)}", # Generate a unique ID
            name=spec_data["name"],
            description=spec_data["description"],
            domain=domain,
            complexity=complexity,
            language=language,
            architecture=ProjectArchitecture.MICROSERVICES, # Default to microservices for now
            theme=ProjectTheme.BUSINESS, # Default to business for now
            target_file_count=target_file_count,
            target_token_count=target_token_count,
            features=template["features"],
            architecture_patterns=template["patterns"],
            dependencies=spec_data.get("dependencies", []),
            seed=random.randint(1, 1000000) # Add a seed for deterministic generation
        )
    
    async def generate_project_specification_unique(
        self, 
        domain: ProjectDomain, 
        complexity: ProjectComplexity,
        language: str,
        architecture: ProjectArchitecture,
        theme: ProjectTheme,
        unique_id: str,
        seed: int
    ) -> ProjectSpecification:
        """Generate a unique project specification using all uniqueness factors"""
        
        # Set deterministic seed for consistent but unique generation
        import random
        random.seed(seed)
        
        template = self.template_manager.get_template(domain, complexity)
        
        # Calculate target metrics based on complexity and config constraints
        complexity_ranges = {
            ProjectComplexity.EASY: {
                "files": (self.config.phase2.min_files_per_project, min(self.config.phase2.max_files_per_project, 15)),
                "tokens": tuple(self.config.phase3.context_ranges["easy"])
            },
            ProjectComplexity.MEDIUM: {
                "files": (max(15, self.config.phase2.min_files_per_project), min(self.config.phase2.max_files_per_project, 40)),
                "tokens": tuple(self.config.phase3.context_ranges["medium"])
            },
            ProjectComplexity.HARD: {
                "files": (max(40, self.config.phase2.min_files_per_project), min(self.config.phase2.max_files_per_project, 80)),
                "tokens": tuple(self.config.phase3.context_ranges["hard"])
            },
            ProjectComplexity.EXPERT: {
                "files": (max(80, self.config.phase2.min_files_per_project), self.config.phase2.max_files_per_project),
                "tokens": tuple(self.config.phase3.context_ranges["expert"])
            }
        }
        
        ranges = complexity_ranges[complexity]
        # Add seed-based variation to target counts for uniqueness
        min_files = max(ranges["files"][0], self.config.phase2.min_files_per_project)
        max_files = min(ranges["files"][1], self.config.phase2.max_files_per_project)
        target_file_count = random.randint(min_files, max_files)
        target_token_count = random.randint(*ranges["tokens"])
        
        # Generate specification using LLM with unique factors
        prompt = f"""
        Generate a detailed specification for a {complexity.value} {domain.value} project in {language}.
        
        Requirements:
        - Project ID: {unique_id}
        - Domain: {domain.value}
        - Complexity: {complexity.value}
        - Architecture: {architecture.value}
        - Theme: {theme.value}
        - Target files: {target_file_count}
        - Target tokens: {target_token_count}
        - Features to include: {', '.join(template['features'])}
        - Architecture patterns: {', '.join(template['patterns'])}
        
        Create a project that specifically focuses on {theme.value} applications using {architecture.value} architecture.
        The project should be distinctly different from other {domain.value} projects by emphasizing the {theme.value} aspect.
        
        Please provide:
        1. Project name (creative, unique, reflecting the {theme.value} theme)
        2. Detailed description (2-3 paragraphs, emphasizing {theme.value} and {architecture.value})
        3. List of main dependencies/libraries (appropriate for {architecture.value})
        4. Brief explanation of why this {architecture.value} approach fits this {theme.value} project
        
        Return as JSON with keys: name, description, dependencies, architecture_justification
        """
        
        system_prompt = f"""You are a senior software architect specializing in {theme.value} applications using {architecture.value} architecture. 
        Generate specifications for unique, realistic projects that would represent real-world {theme.value} software of the specified complexity.
        Each project should be distinctly different, even within the same domain, by leveraging different aspects of the {theme.value} theme and {architecture.value} patterns."""
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["requirements"],
            prompt,
            system_prompt
        )
        
        try:
            # Ensure response is not None or empty
            if not response or response.strip() == "":
                raise json.JSONDecodeError("Empty response", "", 0)
            
            # Parse JSON directly - works perfectly without cleaning
            spec_data = json.loads(response)
        except (json.JSONDecodeError, TypeError) as e:
            # Fail immediately if JSON parsing fails - no fallbacks!
            raise APIError(
                provider=self.llm_generator.generators["requirements"].title(),
                error_type="INVALID_JSON",
                message=f"LLM failed to generate valid JSON for project specification {unique_id}: {str(e)}. Response was: {response[:200] if response else 'None'}..."
            )
        
        return ProjectSpecification(
            unique_id=unique_id,
            name=spec_data["name"],
            description=spec_data["description"],
            domain=domain,
            complexity=complexity,
            language=language,
            architecture=architecture,
            theme=theme,
            target_file_count=target_file_count,
            target_token_count=target_token_count,
            features=template["features"],
            architecture_patterns=template["patterns"],
            dependencies=spec_data.get("dependencies", []),
            seed=seed
        )
    
    async def generate_project_architecture(self, spec: ProjectSpecification) -> Tuple[Dict[str, Any], str]:
        """Generate project architecture and file structure"""
        
        prompt = f"""
        Design the file structure for this project:
        
        Project: {spec.name}
        Description: {spec.description}
        Language: {spec.language}
        Domain: {spec.domain.value}
        Complexity: {spec.complexity.value}
        Target files: {spec.target_file_count}
        Features: {', '.join(spec.features)}
        Patterns: {', '.join(spec.architecture_patterns)}
        
        IMPORTANT: Include both code AND documentation files:
        - Source code files (main application logic)
        - Configuration files (settings, configs)
        - Documentation files (README.md, API docs, etc.)
        - Test files (unit tests, integration tests)
        
        Return ONLY the file structure as nested JSON. Use this format:
        {{
          "file_structure": {{
            "project_root/": {{
              "file1.ext": "file_description",
              "subdirectory/": {{
                "file2.ext": "file_description"
              }}
            }}
          }},
          "overview": "Brief 1-2 sentence architectural overview"
        }}
        
        Do NOT include full file content - only file paths and brief descriptions.
        """
        
        system_prompt = """You are a senior software architect. Design realistic, well-structured file hierarchies that demonstrate good software engineering practices. Return ONLY file paths with brief descriptions - do NOT include full file content. The structure should include both code files AND essential documentation files (README.md, API docs, setup guides) as every professional project requires documentation. Be concise to avoid response truncation."""
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["architecture"],
            prompt,
            system_prompt
        )
        
        try:
            # Parse JSON directly - works perfectly without cleaning
            arch_data = json.loads(response)
            return arch_data.get("file_structure", {}), arch_data.get("overview", "")
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown if direct parsing fails
            import re
            from rich.console import Console
            console = Console()
            
            # More robust regex that captures everything between ```json and ```
            json_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response, re.DOTALL)
            if json_match:
                extracted_text = json_match.group(1).strip()
                console.print(f"   🔍 [blue]Extracted text length: {len(extracted_text)} chars[/blue]")
                
                # Smart JSON extraction - find the complete JSON object boundaries
                def extract_json_object(text):
                    """Extract the first complete JSON object from text"""
                    text = text.strip()
                    if not text.startswith('{'):
                        return None
                    
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    
                    for i, char in enumerate(text):
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                            continue
                            
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                            
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    return text[:i+1]
                    
                    return None
                
                extracted_json = extract_json_object(extracted_text)
                if extracted_json:
                    try:
                        # Parse extracted JSON directly
                        arch_data = json.loads(extracted_json)
                        return arch_data.get("file_structure", {}), arch_data.get("overview", "")
                    except json.JSONDecodeError as json_err:
                        # Add debugging info
                        console.print(f"   🐛 [red]Smart JSON extraction failed:[/red] {str(json_err)}")
                        console.print(f"   📝 [yellow]Extracted JSON (first 200 chars):[/yellow] {extracted_json[:200]}")
                        pass
                else:
                    console.print(f"   🐛 [red]Could not find complete JSON object in markdown[/red]")
                    console.print(f"   📝 [yellow]Raw text (first 200 chars):[/yellow] {extracted_text[:200]}")
                    
                    # Try a more lenient approach - just parse the extracted text directly
                    try:
                        # Sometimes the extraction logic fails but the text is still valid JSON
                        arch_data = json.loads(extracted_text)
                        console.print(f"   ✅ [green]Lenient parsing succeeded![/green]")
                        return arch_data.get("file_structure", {}), arch_data.get("overview", "")
                    except json.JSONDecodeError as lenient_err:
                        console.print(f"   🔍 [yellow]Lenient parsing also failed: {lenient_err}[/yellow]")
                        
                        # Final attempt: try to reconstruct truncated JSON
                        if extracted_text.count('{') > extracted_text.count('}'):
                            # JSON appears truncated, try to find a valid subset
                            console.print(f"   🛠️ [yellow]Attempting to reconstruct truncated JSON...[/yellow]")
                            
                            # Find the last complete object boundary
                            brace_count = 0
                            last_valid_pos = -1
                            
                            for i, char in enumerate(extracted_text):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        last_valid_pos = i + 1
                                        break
                            
                            if last_valid_pos > 0:
                                try:
                                    truncated_json = extracted_text[:last_valid_pos]
                                    arch_data = json.loads(truncated_json)
                                    console.print(f"   ✅ [green]Truncation fix succeeded![/green]")
                                    return arch_data.get("file_structure", {}), arch_data.get("overview", "")
                                except json.JSONDecodeError:
                                    console.print(f"   ❌ [red]Truncation fix failed[/red]")
            
            # Fail immediately if JSON parsing fails - no fallbacks!
            console.print(f"   📝 [yellow]Full response (first 500 chars):[/yellow] {response[:500] if response else 'None'}")
            raise APIError(
                provider=self.llm_generator.generators["architecture"].title(),
                error_type="INVALID_JSON",
                message=f"LLM failed to generate valid JSON for project architecture: {str(e)}. Response was: {response[:200] if response else 'None'}..."
            )
    
    async def generate_file_content(
        self, 
        file_path: str, 
        spec: ProjectSpecification,
        file_structure: Dict[str, Any],
        dependencies: List[str] = None
    ) -> GeneratedFile:
        """Generate content for a specific file"""
        
        dependencies = dependencies or []
        
        # Determine file type
        file_type = self._classify_file_type(file_path)
        
        prompt = f"""
        Generate realistic, production-quality code for this file:
        
        File path: {file_path}
        Project: {spec.name} ({spec.domain.value})
        Language: {spec.language}
        File type: {file_type}
        Dependencies: {', '.join(dependencies)}
        
        Project context:
        - Description: {spec.description}
        - Features: {', '.join(spec.features[:5])}  # Limit for prompt size
        - Architecture patterns: {', '.join(spec.architecture_patterns)}
        
        Requirements:
        1. Write complete, functional code
        2. Include appropriate imports/dependencies
        3. Add meaningful comments and docstrings
        4. Follow language best practices
        5. Make code realistic and non-trivial
        6. Include error handling where appropriate
        
        Return only the code content, no explanations.
        """
        
        system_prompt = f"""You are an expert {spec.language} developer. Write production-quality code that is realistic, well-structured, and follows best practices. The code should be complex enough to demonstrate real-world software development."""
        
        content = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["implementation"],
            prompt,
            system_prompt
        )
        
        # Fail immediately if LLM returns empty content - no fallbacks!
        if content is None or (isinstance(content, str) and content.strip() == ""):
            raise APIError(
                provider=self.llm_generator.generators["implementation"].title(),
                error_type="EMPTY_RESPONSE", 
                message=f"LLM failed to generate content for {file_path}. This indicates an API issue that must be resolved."
            )

        # Calculate complexity score (simple heuristic)
        complexity_score = self._calculate_complexity_score(content)
        
        return GeneratedFile(
            path=file_path,
            content=content,
            file_type=file_type,
            dependencies=dependencies,
            complexity_score=complexity_score
        )
    
    def _classify_file_type(self, file_path: str) -> str:
        """Classify file type based on path"""
        path_lower = file_path.lower()
        
        if any(test_dir in path_lower for test_dir in ['test', 'spec', '__test__']):
            return 'test'
        elif any(config_file in path_lower for config_file in ['config', 'settings', '.env', 'package.json']):
            return 'config'
        elif any(doc_file in path_lower for doc_file in ['readme', 'doc', 'documentation']):
            return 'documentation'
        else:
            return 'source'
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score for file content"""
        # Handle None or empty content
        if content is None or content.strip() == "":
            return 0.0
            
        # Simple heuristic based on content characteristics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Factors contributing to complexity
        line_count = len(non_empty_lines)
        comment_ratio = sum(1 for line in non_empty_lines if line.strip().startswith('#') or '//' in line) / max(line_count, 1)
        avg_line_length = sum(len(line) for line in non_empty_lines) / max(line_count, 1)
        
        # Simple scoring (0-1 scale)
        score = min(1.0, (line_count / 100) * 0.6 + comment_ratio * 0.2 + (avg_line_length / 80) * 0.2)
        
        return round(score, 2)
    
    def _validate_project_constraints(self, generated_files: List[dict], spec: ProjectSpecification) -> tuple:
        """Validate that generated project meets configuration constraints
        
        Returns: (is_valid, error_message)
        """
        file_count = len(generated_files)
        
        # Check file count constraints (ENFORCED)
        if file_count < self.config.phase2.min_files_per_project:
            return False, f"File count {file_count} below minimum {self.config.phase2.min_files_per_project}"
        elif file_count > self.config.phase2.max_files_per_project:
            return False, f"File count {file_count} above maximum {self.config.phase2.max_files_per_project}"
        
        # Calculate complexity scores for validation
        complexity_scores = []
        documentation_files = 0
        
        for file_info in generated_files:
            complexity_score = self._calculate_complexity_score(file_info['content'])
            complexity_scores.append(complexity_score)
            
            # Check if this is a documentation file
            if file_info['type'] == 'documentation' or any(doc_indicator in file_info['path'].lower() 
                for doc_indicator in ['readme', 'doc', 'documentation']):
                documentation_files += 1
        
        # Check average complexity constraints and filter if needed
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        # ENFORCED: Check complexity requirements
        if avg_complexity < self.config.phase2.min_complexity_score:
            return False, f"Average complexity {avg_complexity:.2f} below minimum {self.config.phase2.min_complexity_score}"
        elif avg_complexity > self.config.phase2.max_complexity_score:
            return False, f"Average complexity {avg_complexity:.2f} above maximum {self.config.phase2.max_complexity_score}"
        
        # ENFORCED: Check documentation ratio
        documentation_ratio = documentation_files / file_count if file_count > 0 else 0
        if documentation_ratio < self.config.phase2.min_documentation_ratio:
            return False, f"Documentation ratio {documentation_ratio:.2f} below minimum {self.config.phase2.min_documentation_ratio}"
        
        logger.debug(
            f"✅ Project '{spec.name}' validation passed: {file_count} files, "
            f"complexity {avg_complexity:.2f}, docs {documentation_ratio:.2f}"
        )
        
        return True, ""
    
    async def generate_complete_project(
        self, 
        domain: ProjectDomain,
        complexity: ProjectComplexity,
        language: str
    ) -> SyntheticProject:
        """Generate a complete synthetic project"""
        
        logger.info(f"Generating {complexity.value} {domain.value} project in {language}")
        
        # Step 1: Generate specification
        spec = await self.generate_project_specification(domain, complexity, language)
        logger.info(f"Generated specification for '{spec.name}'")
        
        # Step 2: Generate architecture
        file_structure, architecture_overview = await self.generate_project_architecture(spec)
        logger.info("Generated project architecture")
        
        # Step 3: Generate files
        file_paths = self._extract_file_paths(file_structure)
        files = []
        
        for file_path in file_paths:
            try:
                generated_file = await self.generate_file_content(file_path, spec, file_structure)
                files.append(generated_file)
                logger.debug(f"Generated file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate file {file_path}: {e}")
                continue
        
        logger.info(f"Generated {len(files)} files")
        
        # Step 4: Generate additional metadata
        setup_instructions = await self._generate_setup_instructions(spec, files)
        test_scenarios = await self._generate_test_scenarios(spec)
        
        return SyntheticProject(
            specification=spec,
            files=files,
            file_structure=file_structure,
            architecture_overview=architecture_overview,
            setup_instructions=setup_instructions,
            test_scenarios=test_scenarios
        )
    
    def _extract_file_paths(self, file_structure: Dict[str, Any], current_path: str = "") -> List[str]:
        """Extract all file paths from nested file structure"""
        paths = []
        
        for name, content in file_structure.items():
            full_path = f"{current_path}/{name}" if current_path else name
            
            if isinstance(content, dict):
                # Directory - recurse
                paths.extend(self._extract_file_paths(content, full_path))
            else:
                # File - add to paths
                paths.append(full_path)
        
        return paths
    
    async def _generate_setup_instructions(self, spec: ProjectSpecification, files: List[GeneratedFile]) -> str:
        """Generate setup and installation instructions"""
        prompt = f"""
        Generate clear setup instructions for this project:
        
        Project: {spec.name}
        Language: {spec.language}
        Dependencies: {', '.join(spec.dependencies)}
        File count: {len(files)}
        
        Include:
        1. Prerequisites and requirements
        2. Installation steps
        3. Configuration needed
        4. How to run the project
        5. Basic usage examples
        
        Keep it concise but complete.
        """
        
        return await self.llm_generator.generate_with_model(
            self.llm_generator.generators["scenarios"],
            prompt,
            "You are a technical writer creating clear, actionable setup instructions."
        )
    
    async def _generate_test_scenarios(self, spec: ProjectSpecification) -> List[str]:
        """Generate test scenarios for the project"""
        prompt = f"""
        Generate 5-10 realistic test scenarios for this project:
        
        Project: {spec.name}
        Domain: {spec.domain.value}
        Features: {', '.join(spec.features)}
        
        Each scenario should be:
        1. Specific and actionable
        2. Cover different aspects of the system
        3. Realistic for the domain
        4. Suitable for automated testing
        
        Return as JSON array of strings.
        """
        
        response = await self.llm_generator.generate_with_model(
            self.llm_generator.generators["scenarios"],
            prompt,
            "You are a QA engineer designing comprehensive test scenarios."
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return [
                "Test basic functionality",
                "Test error handling",
                "Test performance under load",
                "Test security measures",
                "Test integration points"
            ]
    
    async def save_project(self, project: SyntheticProject) -> Path:
        """Save a synthetic project to disk"""
        # Create project directory
        safe_name = project.specification.name.lower().replace(' ', '_').replace('-', '_')
        project_dir = self.generated_dir / safe_name
        project_dir.mkdir(exist_ok=True)
        
        # Save individual files
        for file in project.files:
            file_path = project_dir / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file.content)
        
        # Save project metadata
        metadata = {
            "specification": {
                "name": project.specification.name,
                "description": project.specification.description,
                "domain": project.specification.domain.value,
                "complexity": project.specification.complexity.value,
                "language": project.specification.language,
                "target_file_count": project.specification.target_file_count,
                "target_token_count": project.specification.target_token_count,
                "features": project.specification.features,
                "architecture_patterns": project.specification.architecture_patterns,
                "dependencies": project.specification.dependencies
            },
            "files": [{"path": f.path, "type": f.file_type} for f in project.files],
            "file_structure": project.file_structure,
            "architecture_overview": project.architecture_overview,
            "setup_instructions": project.setup_instructions,
            "test_scenarios": project.test_scenarios
        }
        
        with open(project_dir / "project_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved project to {project_dir}")
        return project_dir

    async def generate_project_files(self, spec_dict: dict, target_files: int, target_tokens: int) -> List[dict]:
        """Generate actual code files for a project specification from Phase 1"""
        
        # ⏰ START TIMING
        import time
        start_time = time.time()
        
        # Import Rich console for progress reporting
        from rich.console import Console
        console = Console()
        
        # Convert dict back to ProjectSpecification object
        spec = ProjectSpecification(
            unique_id=f"{ProjectDomain(spec_dict['domain']).value}-{ProjectComplexity(spec_dict['complexity']).value}-{spec_dict['language']}-{random.randint(1000, 9999)}", # Generate a unique ID
            name=spec_dict['name'],
            description=spec_dict['description'],
            domain=ProjectDomain(spec_dict['domain']),
            complexity=ProjectComplexity(spec_dict['complexity']),
            language=spec_dict['language'],
            architecture=ProjectArchitecture.MICROSERVICES, # Default to microservices for now
            theme=ProjectTheme.BUSINESS, # Default to business for now
            target_file_count=target_files,
            target_token_count=target_tokens,
            features=spec_dict.get('features', []),
            architecture_patterns=spec_dict.get('architecture_patterns', []),
            dependencies=spec_dict.get('dependencies', []),
            seed=random.randint(1, 1000000) # Add a seed for deterministic generation
        )
        
        # Generate a proper file structure for this project
        file_structure, _ = await self.generate_project_architecture(spec)
        
        # Extract file paths from the structure
        file_paths = self._extract_file_paths(file_structure)
        
        # Limit to target_files if we have too many
        if len(file_paths) > target_files:
            file_paths = file_paths[:target_files]
        
        # Generate additional files if we need more
        elif len(file_paths) < target_files:
            additional_files = await self._generate_additional_files(spec, target_files - len(file_paths))
            file_paths.extend(additional_files)
        
        # Generate content for each file using our 3 Elite Models in parallel
        generated_files = []
        
        console.print(f"      🏭 Generating {len(file_paths)} files in parallel...")
        
        async def generate_single_file(file_path, file_index):
            """Generate a single file with retry logic"""
            try:
                console.print(f"      📄 {file_index}/{len(file_paths)}: {file_path}")
                
                generated_file = await self.generate_file_content(file_path, spec, file_structure)
                
                # Ensure content is not None before processing
                file_content = generated_file.content or ""
                lines_count = len(file_content.splitlines())
                chars_count = len(file_content)
                
                console.print(f"      ✅ {file_path} ({lines_count} lines, {chars_count:,} chars)")
                
                return {
                    'path': file_path,
                    'content': generated_file.content,
                    'type': self._classify_file_type(file_path),
                    'retry_count': 0,  # First attempt succeeded
                    'success': True
                }
                
            except Exception as e:
                console.print(f"      ❌ {file_path}: {str(e)}")
                logger.error(f"Failed to generate file {file_path}: {e}")
                
                # Retry logic for individual file generation
                max_file_retries = 2
                
                for retry_attempt in range(max_file_retries):
                    try:
                        console.print(f"      🔄 Retry {retry_attempt + 1}/{max_file_retries} for {file_path}...")
                        
                        # Regenerate the file using the same method as the original attempt
                        retry_generated_file = await self.generate_file_content(file_path, spec, file_structure)
                        
                        if retry_generated_file and retry_generated_file.content:
                            file_content = retry_generated_file.content
                            lines_count = len(file_content.splitlines())
                            chars_count = len(file_content)
                            
                            console.print(f"      ✅ {file_path} regenerated on retry {retry_attempt + 1} ({lines_count} lines, {chars_count:,} chars)")
                            
                            return {
                                'path': file_path,
                                'content': file_content,
                                'type': self._classify_file_type(file_path),
                                'retry_count': retry_attempt + 1,
                                'success': True
                            }
                        else:
                            console.print(f"      ❌ Retry {retry_attempt + 1} failed: Empty content")
                            
                    except Exception as retry_error:
                        console.print(f"      ❌ Retry {retry_attempt + 1} failed: {str(retry_error)}")
                        logger.error(f"Retry {retry_attempt + 1} failed for {file_path}: {retry_error}")
                
                # If all retries failed, return failure
                console.print(f"      ⚠️  All retries exhausted. Skipping {file_path}...")
                return {
                    'path': file_path,
                    'content': '',
                    'type': self._classify_file_type(file_path),
                    'retry_count': max_file_retries,
                    'success': False
                }
        
        # Generate all files in parallel
        console.print(f"      🚀 Starting parallel file generation...")
        
        # Create tasks for all files
        file_tasks = []
        for i, file_path in enumerate(file_paths, 1):
            task = generate_single_file(file_path, i)
            file_tasks.append(task)
        
        # Execute all file generation tasks in parallel
        file_results = await asyncio.gather(*file_tasks, return_exceptions=True)
        
        # Process results and filter successful ones
        for result in file_results:
            if isinstance(result, Exception):
                console.print(f"      ❌ File generation exception: {str(result)}")
                logger.error(f"File generation failed with exception: {result}")
            elif result and result.get('success', False):
                generated_files.append(result)
        
        # Check if we have a reasonable number of files after skipping failures
        if len(generated_files) == 0:
            raise APIError("ProjectGeneration", "GENERATION_FAILED", 
                          f"Project '{spec.name}' failed: No files were successfully generated")
        elif len(generated_files) < target_files * 0.5:  # Less than 50% of target files
            logger.warning(f"Project '{spec.name}' generated only {len(generated_files)}/{target_files} files ({len(generated_files)/target_files*100:.1f}%)")
        
        total_lines = sum(len(f['content'].splitlines()) for f in generated_files)
        total_chars = sum(len(f['content']) for f in generated_files)
        
        # Calculate retry statistics
        first_attempt_success = sum(1 for f in generated_files if f.get('retry_count', 0) == 0)
        retry_success = sum(1 for f in generated_files if f.get('retry_count', 0) > 0)
        total_skipped = len(file_paths) - len(generated_files)
        
        # ⏰ END TIMING
        end_time = time.time()
        generation_time = end_time - start_time
        
        success_rate = len(generated_files) / len(file_paths) * 100
        console.print(f"      🎉 [bold green]Project completed![/bold green] {len(generated_files)}/{len(file_paths)} files ({success_rate:.1f}% success), {total_lines:,} lines, {total_chars:,} chars")
        if retry_success > 0 or total_skipped > 0:
            console.print(f"      📊 Retry stats: {first_attempt_success} first try, {retry_success} after retry, {total_skipped} skipped")
        console.print(f"      ⏱️  [cyan]Generated in {generation_time:.1f}s[/cyan]")
        
        # Validate project constraints with enforcement
        is_valid, error_message = self._validate_project_constraints(generated_files, spec)
        if not is_valid:
            raise APIError("ProjectGeneration", "QUALITY_VALIDATION_FAILED", 
                          f"Project '{spec.name}' failed quality validation: {error_message}")
        
        # Return with timing and count data for CLI analysis
        return {
            'files': generated_files,
            'files_created': len(generated_files),
            'lines_created': total_lines,
            'chars_created': total_chars,
            'generation_time': generation_time,
            'project_name': spec.name
        }
    
    async def _generate_additional_files(self, spec: ProjectSpecification, count: int) -> List[str]:
        """Generate additional file paths if needed to reach target file count"""
        
        lang_extensions = {
            'python': '.py',
            'javascript': '.js', 
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp',
            'go': '.go'
        }
        
        ext = lang_extensions.get(spec.language, '.txt')
        additional_files = []
        
        # Generate common additional files based on project type
        base_files = [
            f"src/utils{ext}",
            f"src/config{ext}",
            f"src/constants{ext}",
            f"tests/test_main{ext}",
            f"tests/test_utils{ext}",
            "README.md",
            "requirements.txt" if spec.language == 'python' else "package.json",
            ".gitignore",
            "Dockerfile",
            f"docs/api{'.md'}",
        ]
        
        # Add files until we reach the target count
        for i, file_path in enumerate(base_files):
            if i >= count:
                break
            additional_files.append(file_path)
        
        # If we still need more files, generate numbered modules
        remaining = count - len(additional_files)
        for i in range(remaining):
            additional_files.append(f"src/module_{i+1}{ext}")
        
        return additional_files




# Example usage and testing
async def main():
    """Example usage of the synthetic generator"""
    config = Config()
    generator = SyntheticProjectGenerator(config)
    
    # Generate a sample project
    project = await generator.generate_complete_project(
        domain=ProjectDomain.WEB_APPLICATION,
        complexity=ProjectComplexity.MEDIUM,
        language="python"
    )
    
    print(f"Generated project: {project.specification.name}")
    print(f"Files: {len(project.files)}")
    print(f"Architecture: {project.architecture_overview[:200]}...")
    
    # Save project
    await generator.save_project(project)


if __name__ == "__main__":
    asyncio.run(main()) 