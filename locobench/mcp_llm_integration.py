"""
MCP-LLM Integration for LoCoBench

This module provides integration between MCP tools and LLM clients
(OpenAI and Anthropic) for intelligent file retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import asyncio

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .mcp_retrieval import LoCoBenchMCPServer, MCPTool
from .core.config import Config, APIConfig

logger = logging.getLogger(__name__)


class MCPLLMIntegrator:
    """
    Integrates MCP tools with LLM clients for intelligent file retrieval.
    
    Supports both OpenAI and Anthropic (Claude) APIs with tool calling.
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
        
        # Initialize LLM clients
        self.openai_client: Optional[Any] = None
        self.anthropic_client: Optional[Any] = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize OpenAI and Anthropic clients"""
        if OPENAI_AVAILABLE and self.api_config.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(
                    api_key=self.api_config.openai_api_key,
                    base_url=self.api_config.openai_base_url.rstrip("/") if self.api_config.openai_base_url else None,
                    timeout=self.api_config.openai_timeout,
                )
                logger.info("✅ OpenAI client initialized for MCP tool calling")
            except ImportError:
                logger.warning("OpenAI library available but AsyncOpenAI not found")
        
        if ANTHROPIC_AVAILABLE and self.api_config.claude_bearer_token:
            try:
                from anthropic import AsyncAnthropic
                self.anthropic_client = AsyncAnthropic(
                    api_key=self.api_config.claude_bearer_token,
                )
                logger.info("✅ Anthropic client initialized for MCP tool calling")
            except ImportError:
                logger.warning("Anthropic library available but AsyncAnthropic not found")
    
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
    
    def _convert_tools_to_anthropic_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic tool use format"""
        tools = []
        for tool in self.mcp_server.tools:
            tool_dict = tool.to_dict()
            # Anthropic uses a slightly different format
            anthropic_tool = {
                "name": tool_dict["name"],
                "description": tool_dict["description"],
                "input_schema": tool_dict["parameters"],
            }
            tools.append(anthropic_tool)
        return tools
    
    async def retrieve_with_openai(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 5,
    ) -> str:
        """
        Retrieve files using OpenAI with tool calling.
        
        Args:
            model: OpenAI model to use (must support tool calling)
            max_iterations: Maximum number of tool calling iterations
        
        Returns:
            Formatted context string with selected files
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
        
        # Convert tools to OpenAI format
        tools = self._convert_tools_to_openai_format()
        
        if not tools:
            logger.warning("No tools available for OpenAI tool calling")
            return ""
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create user prompt
        user_prompt = self._create_user_prompt()
        
        # Conversation history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        selected_files: Set[str] = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 OpenAI tool calling iteration {iteration}/{max_iterations}")
            
            # Call OpenAI with tools
            try:
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",  # Let model decide when to use tools
                    temperature=0.1,  # Low temperature for deterministic tool selection
                )
                
                message = response.choices[0].message
                
                # Add assistant message to conversation
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
                    logger.info("✅ OpenAI finished tool calling")
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
                            "files": [r.get("path", "") for r in tool_results[:10]],  # Limit to 10 for response
                            "summary": f"Found {len(tool_results)} relevant files",
                        })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
                
            except Exception as e:
                logger.error(f"Error in OpenAI tool calling iteration {iteration}: {e}")
                break
        
        # Format final context from selected files
        return self.mcp_server.format_selected_context()
    
    async def retrieve_with_anthropic(
        self,
        model: str = "claude-sonnet-4",
        max_iterations: int = 5,
    ) -> str:
        """
        Retrieve files using Anthropic Claude with tool use.
        
        Args:
            model: Anthropic model to use (must support tool use)
            max_iterations: Maximum number of tool calling iterations
        
        Returns:
            Formatted context string with selected files
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Set ANTHROPIC_API_KEY or CLAUDE_BEARER_TOKEN.")
        
        # Convert tools to Anthropic format
        tools = self._convert_tools_to_anthropic_format()
        
        if not tools:
            logger.warning("No tools available for Anthropic tool use")
            return ""
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create user prompt
        user_prompt = self._create_user_prompt()
        
        # Conversation history
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]
        
        selected_files: Set[str] = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 Anthropic tool use iteration {iteration}/{max_iterations}")
            
            # Call Anthropic with tools
            try:
                response = await self.anthropic_client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=4096,
                )
                
                # Process response
                assistant_message = {
                    "role": "assistant",
                    "content": [],
                }
                
                # Handle text content
                text_parts = []
                tool_use_parts = []
                
                for content_block in response.content:
                    if content_block.type == "text":
                        text_parts.append(content_block.text)
                    elif content_block.type == "tool_use":
                        tool_use_parts.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input,
                        })
                
                if text_parts:
                    assistant_message["content"].extend(text_parts)
                
                if tool_use_parts:
                    assistant_message["content"].extend(tool_use_parts)
                
                messages.append(assistant_message)
                
                # Check if model wants to use tools
                if not tool_use_parts:
                    logger.info("✅ Anthropic finished tool use")
                    break
                
                # Execute tool calls
                tool_results = []
                for tool_use in tool_use_parts:
                    tool_name = tool_use["name"]
                    tool_args = tool_use["input"]
                    
                    # Execute tool
                    tool_execution_results = self.mcp_server.execute_tool_call(tool_name, tool_args)
                    
                    # Track selected files
                    for result in tool_execution_results:
                        if "path" in result:
                            selected_files.add(result["path"])
                    
                    # Format tool result
                    tool_result = {
                        "tool_use_id": tool_use["id"],
                        "content": json.dumps({
                            "files_found": len(tool_execution_results),
                            "files": [r.get("path", "") for r in tool_execution_results[:10]],
                            "summary": f"Found {len(tool_execution_results)} relevant files",
                        }),
                    }
                    tool_results.append(tool_result)
                
                # Add tool results to conversation
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                
            except Exception as e:
                logger.error(f"Error in Anthropic tool use iteration {iteration}: {e}")
                break
        
        # Format final context from selected files
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
    
    async def retrieve(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        max_iterations: int = 5,
    ) -> str:
        """
        Main retrieval method that uses the specified provider.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (optional, uses default if not specified)
            max_iterations: Maximum tool calling iterations
        
        Returns:
            Formatted context string with selected files
        """
        if provider.lower() == "openai":
            if not model:
                model = self.api_config.default_model_openai or "gpt-4o"
            return await self.retrieve_with_openai(model=model, max_iterations=max_iterations)
        
        elif provider.lower() in ("anthropic", "claude"):
            if not model:
                model = self.api_config.default_model_claude or "claude-sonnet-4"
            return await self.retrieve_with_anthropic(model=model, max_iterations=max_iterations)
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


async def retrieve_with_mcp_llm(
    context_files: Dict[str, str],
    task_prompt: str,
    task_category: str,
    project_dir: Path,
    config: Optional[Config] = None,
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Main entry point for MCP-based retrieval with LLM integration.
    
    Args:
        context_files: Available context files
        task_prompt: Task description
        task_category: Category of the task
        project_dir: Project directory
        config: Configuration object (optional)
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (optional, uses default from config)
    
    Returns:
        Formatted context string with selected files
    """
    # Create MCP server
    mcp_server = LoCoBenchMCPServer(
        project_dir=project_dir,
        task_category=task_category,
        context_files=context_files,
        task_prompt=task_prompt,
    )
    
    # Create integrator
    integrator = MCPLLMIntegrator(
        mcp_server=mcp_server,
        config=config,
    )
    
    # Retrieve files using LLM
    return await integrator.retrieve(
        provider=provider,
        model=model,
    )
