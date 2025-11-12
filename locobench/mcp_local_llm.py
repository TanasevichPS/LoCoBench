"""
Local LLM Integration for MCP Tools

Supports:
- Hugging Face Transformers (local inference)
- Ollama (local LLM server)
- LocalAI / LM Studio (OpenAI-compatible local API)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import asyncio

logger = logging.getLogger(__name__)

# Hugging Face support
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# OpenAI-compatible local API (for LocalAI, LM Studio, etc.)
try:
    import openai
    OPENAI_COMPATIBLE_AVAILABLE = True
except ImportError:
    OPENAI_COMPATIBLE_AVAILABLE = False

from .mcp_retrieval import LoCoBenchMCPServer
from .core.config import Config, APIConfig

logger = logging.getLogger(__name__)


class LocalLLMIntegrator:
    """
    Integrates MCP tools with local LLM models.
    
    Supports:
    - Hugging Face Transformers (direct inference)
    - Ollama (via API)
    - LocalAI / LM Studio (OpenAI-compatible API)
    """
    
    def __init__(
        self,
        mcp_server: LoCoBenchMCPServer,
        config: Optional[Config] = None,
        api_config: Optional[APIConfig] = None,
    ):
        self.mcp_server = mcp_server
        self.config = config
        self.api_config = api_config or (config.api if config else APIConfig())
        
        # Local model cache
        self.hf_model = None
        self.hf_tokenizer = None
        
        # Ollama client
        self.ollama_client = None
        
        # LocalAI/LM Studio client (OpenAI-compatible)
        self.local_openai_client = None
    
    def _initialize_huggingface(self, model_name: str):
        """Initialize Hugging Face model"""
        if not HF_AVAILABLE:
            raise ValueError("Hugging Face not available. Install: pip install transformers torch")
        
        if self.hf_model is None or self.hf_tokenizer is None:
            logger.info(f"Loading Hugging Face model: {model_name}")
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                logger.info(f"✅ Hugging Face model {model_name} loaded")
            except Exception as e:
                logger.error(f"Failed to load Hugging Face model {model_name}: {e}")
                raise
    
    def _initialize_ollama(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client"""
        if not OPENAI_COMPATIBLE_AVAILABLE:
            raise ValueError("OpenAI library not available for Ollama client")
        
        try:
            self.ollama_client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key="ollama",  # Ollama doesn't require real API key
            )
            logger.info(f"✅ Ollama client initialized at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    def _initialize_local_openai(self, base_url: str, api_key: Optional[str] = None):
        """Initialize LocalAI/LM Studio client (OpenAI-compatible)"""
        if not OPENAI_COMPATIBLE_AVAILABLE:
            raise ValueError("OpenAI library not available")
        
        try:
            self.local_openai_client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed",
            )
            logger.info(f"✅ Local OpenAI-compatible client initialized at {base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize local OpenAI client: {e}")
            raise
    
    def _convert_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format"""
        tools = []
        for tool in self.mcp_server.tools:
            tool_dict = tool.to_dict()
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict["description"],
                    "parameters": tool_dict["parameters"],
                }
            })
        return tools
    
    async def retrieve_with_huggingface(
        self,
        model_name: str,
        max_iterations: int = 3,  # Fewer iterations for local models
    ) -> str:
        """
        Retrieve files using Hugging Face model with tool calling simulation.
        
        Note: Most HF models don't support native tool calling, so we simulate it
        by prompting the model to output JSON with tool calls.
        """
        if not HF_AVAILABLE:
            raise ValueError("Hugging Face not available")
        
        self._initialize_huggingface(model_name)
        
        # Convert tools to description format
        tools = self._convert_tools_to_openai_format()
        tools_description = self._format_tools_for_prompt(tools)
        
        # Create prompt
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt() + f"\n\nAvailable tools:\n{tools_description}\n\nPlease output JSON with tool calls in format: {{\"tool_calls\": [{{\"name\": \"tool_name\", \"arguments\": {{...}}}}]}}"
        
        selected_files: Set[str] = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Hugging Face tool calling iteration {iteration}/{max_iterations}")
            
            # Generate response
            try:
                inputs = self.hf_tokenizer(
                    f"{system_prompt}\n\n{user_prompt}",
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=True,
                    )
                
                response_text = self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Try to parse tool calls from response
                tool_calls = self._parse_tool_calls_from_text(response_text)
                
                if not tool_calls:
                    logger.info("✅ Hugging Face finished tool calling")
                    break
                
                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})
                    
                    if tool_name:
                        tool_results = self.mcp_server.execute_tool_call(tool_name, tool_args)
                        
                        # Track selected files
                        for result in tool_results:
                            if "path" in result:
                                selected_files.add(result["path"])
                        
                        # Update prompt with results
                        user_prompt += f"\n\nTool {tool_name} returned {len(tool_results)} files."
                
            except Exception as e:
                logger.error(f"Error in Hugging Face iteration {iteration}: {e}")
                break
        
        return self.mcp_server.format_selected_context()
    
    async def retrieve_with_ollama(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        max_iterations: int = 5,
    ) -> str:
        """
        Retrieve files using Ollama with tool calling.
        
        Requires Ollama server running locally.
        Install: https://ollama.ai
        """
        if not OPENAI_COMPATIBLE_AVAILABLE:
            raise ValueError("OpenAI library not available")
        
        self._initialize_ollama(base_url)
        
        # Convert tools to OpenAI format
        tools = self._convert_tools_to_openai_format()
        
        if not tools:
            logger.warning("No tools available")
            return ""
        
        # Create prompts
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        selected_files: Set[str] = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Ollama tool calling iteration {iteration}/{max_iterations}")
            
            try:
                # Ollama supports OpenAI-compatible API
                response = await self.ollama_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools if iteration == 1 else None,  # Only send tools on first call
                    tool_choice="auto",
                    temperature=0.1,
                )
                
                message = response.choices[0].message
                
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in (message.tool_calls or [])
                    ],
                })
                
                # Check if model wants to call tools
                if not message.tool_calls:
                    logger.info("✅ Ollama finished tool calling")
                    break
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {e}")
                        tool_result = f"Error: Invalid JSON in tool arguments"
                    else:
                        # Execute tool
                        tool_results = self.mcp_server.execute_tool_call(tool_name, tool_args)
                        
                        # Track selected files
                        for result in tool_results:
                            if "path" in result:
                                selected_files.add(result["path"])
                        
                        # Format tool result
                        tool_result = json.dumps({
                            "files_found": len(tool_results),
                            "files": [r.get("path", "") for r in tool_results[:10]],
                            "summary": f"Found {len(tool_results)} relevant files",
                        })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
                
            except Exception as e:
                logger.error(f"Error in Ollama iteration {iteration}: {e}")
                break
        
        return self.mcp_server.format_selected_context()
    
    async def retrieve_with_local_openai(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_iterations: int = 5,
    ) -> str:
        """
        Retrieve files using LocalAI or LM Studio (OpenAI-compatible API).
        
        Requires LocalAI or LM Studio running locally with OpenAI-compatible endpoint.
        """
        if not OPENAI_COMPATIBLE_AVAILABLE:
            raise ValueError("OpenAI library not available")
        
        self._initialize_local_openai(base_url, api_key)
        
        # Convert tools to OpenAI format
        tools = self._convert_tools_to_openai_format()
        
        if not tools:
            logger.warning("No tools available")
            return ""
        
        # Create prompts
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        selected_files: Set[str] = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Local OpenAI tool calling iteration {iteration}/{max_iterations}")
            
            try:
                response = await self.local_openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1,
                )
                
                message = response.choices[0].message
                
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in (message.tool_calls or [])
                    ],
                })
                
                # Check if model wants to call tools
                if not message.tool_calls:
                    logger.info("✅ Local OpenAI finished tool calling")
                    break
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {e}")
                        tool_result = f"Error: Invalid JSON in tool arguments"
                    else:
                        # Execute tool
                        tool_results = self.mcp_server.execute_tool_call(tool_name, tool_args)
                        
                        # Track selected files
                        for result in tool_results:
                            if "path" in result:
                                selected_files.add(result["path"])
                        
                        # Format tool result
                        tool_result = json.dumps({
                            "files_found": len(tool_results),
                            "files": [r.get("path", "") for r in tool_results[:10]],
                            "summary": f"Found {len(tool_results)} relevant files",
                        })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
                
            except Exception as e:
                logger.error(f"Error in Local OpenAI iteration {iteration}: {e}")
                break
        
        return self.mcp_server.format_selected_context()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM"""
        return f"""You are an intelligent file retrieval assistant for code analysis tasks.

Your task is to help select the most relevant files from a codebase for a specific task.

Task Category: {self.mcp_server.task_category}
Task Description: {self.mcp_server.task_prompt}

You have access to specialized tools for this task category. Use these tools strategically to find the most relevant files.

Guidelines:
1. Start by understanding what the task requires
2. Use the appropriate tools to find relevant files
3. You can call multiple tools in sequence or parallel
4. Focus on finding files that are directly relevant to the task
5. Stop when you have found sufficient relevant files

Available tools:
{self._format_tools_description()}
"""
    
    def _create_user_prompt(self) -> str:
        """Create user prompt for LLM"""
        return f"""Please help me find the most relevant files for this task:

Task: {self.mcp_server.task_prompt}
Category: {self.mcp_server.task_category}

Use the available tools to identify and select the files that are most relevant to completing this task.
Start by analyzing what the task requires, then use the appropriate tools to find the relevant files.
"""
    
    def _format_tools_description(self) -> str:
        """Format tools description for prompt"""
        descriptions = []
        for tool in self.mcp_server.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for text prompt (for models without native tool calling)"""
        formatted = []
        for tool in tools:
            func = tool["function"]
            formatted.append(f"- {func['name']}: {func['description']}")
            formatted.append(f"  Parameters: {', '.join(func['parameters']['properties'].keys())}")
        return "\n".join(formatted)
    
    def _parse_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model text output (for models without native tool calling)"""
        # Try to find JSON in the response
        import re
        
        # Look for JSON object
        json_match = re.search(r'\{[^{}]*"tool_calls"[^{}]*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed.get("tool_calls", [])
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract tool names and arguments from text
        tool_calls = []
        for tool in self.mcp_server.tools:
            if tool.name.lower() in text.lower():
                # Simple heuristic: if tool name mentioned, try to call it
                tool_calls.append({
                    "name": tool.name,
                    "arguments": {"keywords": self.mcp_server.task_prompt[:100]},
                })
        
        return tool_calls


async def retrieve_with_local_llm(
    context_files: Dict[str, str],
    task_prompt: str,
    task_category: str,
    project_dir: Path,
    provider: str = "ollama",  # "ollama", "huggingface", or "local_openai"
    model: str = "llama3.2",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """
    Main entry point for MCP-based retrieval with local LLM.
    
    Args:
        context_files: Available context files
        task_prompt: Task description
        task_category: Category of the task
        project_dir: Project directory
        provider: "ollama", "huggingface", or "local_openai"
        model: Model name
        base_url: Base URL for Ollama or LocalAI (default: http://localhost:11434 for Ollama)
        api_key: API key for LocalAI (optional)
        config: Configuration object (optional)
    
    Returns:
        Formatted context string with selected files
    """
    from .mcp_retrieval import LoCoBenchMCPServer
    
    # Create MCP server
    mcp_server = LoCoBenchMCPServer(
        project_dir=project_dir,
        task_category=task_category,
        context_files=context_files,
        task_prompt=task_prompt,
    )
    
    # Create integrator
    integrator = LocalLLMIntegrator(
        mcp_server=mcp_server,
        config=config,
    )
    
    # Retrieve files using local LLM
    if provider == "ollama":
        base_url = base_url or "http://localhost:11434"
        return await integrator.retrieve_with_ollama(
            model=model,
            base_url=base_url,
        )
    elif provider == "huggingface":
        return await integrator.retrieve_with_huggingface(
            model_name=model,
        )
    elif provider == "local_openai":
        base_url = base_url or "http://localhost:1234"  # Default LM Studio port
        return await integrator.retrieve_with_local_openai(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Use 'ollama', 'huggingface' (or 'hf'), or 'local_openai' (or 'local')."
        )
