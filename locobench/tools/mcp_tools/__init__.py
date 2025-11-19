"""Reusable MCP-powered scenario extraction utilities.

This module consolidates the scenario-loading, context-file resolution and
relevance scoring logic that used to be spread across several ad-hoc scripts.
The resulting API can be reused by other benchmarks (not just LoCoBench) by
instantiating :class:`ScenarioExtractionAgent` with the desired paths.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from locobench.tools.chunk_analysis import analyze_chunk_relevance
from locobench.tools.file_chunking import chunk_file_smart

logger = logging.getLogger(__name__)

DEFAULT_TASK_CATEGORIES: Tuple[str, ...] = (
    "architectural_understanding",
    "cross_file_refactoring",
    "feature_implementation",
    "bug_investigation",
    "multi_session_development",
    "code_comprehension",
    "integration_testing",
    "security_analysis",
)

DEFAULT_DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard", "expert")


@dataclass
class ScenarioExtractionConfig:
    """Configuration for resolving scenarios and scoring files."""

    scenarios_dir: str = "data/output/scenarios"
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated"
    max_files_to_scan: int = 10
    max_chunks_per_file: int = 3
    chunk_size: int = 2000
    keyword_fallback_weight: float = 1.0
    task_categories: Sequence[str] = field(default_factory=lambda: DEFAULT_TASK_CATEGORIES)
    difficulties: Sequence[str] = field(default_factory=lambda: DEFAULT_DIFFICULTIES)

    def resolve_paths(self) -> Tuple[Path, Path]:
        """Return absolute paths for scenarios directory and base project directory."""
        scenarios_path = Path(self.scenarios_dir)
        if not scenarios_path.is_absolute():
            scenarios_path = (Path.cwd() / scenarios_path).resolve()

        base_path = Path(self.base_path)
        if not base_path.is_absolute():
            base_path = (Path.cwd() / base_path).resolve()

        return scenarios_path, base_path


class ScenarioExtractionAgent:
    """High-level helper for loading scenarios and selecting relevant files."""

    def __init__(self, config: Optional[ScenarioExtractionConfig] = None) -> None:
        self.config = config or ScenarioExtractionConfig()

    # --------------------------------------------------------------------- #
    # Scenario loading helpers
    # --------------------------------------------------------------------- #
    def _load_scenario_json(self, scenario_id: str) -> Optional[Dict]:
        scenarios_path, _ = self.config.resolve_paths()
        scenario_path = scenarios_path / f"{scenario_id}.json"
        if not scenario_path.exists():
            logger.error("Scenario file not found: %s", scenario_path)
            return None

        try:
            return json.loads(scenario_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Could not parse scenario %s: %s", scenario_id, exc)
            logger.debug("Failed scenario path: %s", scenario_path, exc_info=True)
            return None

    def _normalize_context_files(
        self, scenario_data: Dict,
    ) -> Tuple[List[str], Optional[Dict[str, str]]]:
        context_obj = scenario_data.get("context_files", [])
        inline_content: Optional[Dict[str, str]] = None

        if isinstance(context_obj, dict):
            if context_obj and all(isinstance(v, str) for v in context_obj.values()):
                inline_content = context_obj
            context_list = list(context_obj.keys())
        elif isinstance(context_obj, list):
            context_list = context_obj
        else:
            context_list = []

        return context_list, inline_content

    def _extract_project_name(self, scenario_id: str) -> Optional[str]:
        parts = scenario_id.split("_")

        for category in self.config.task_categories:
            marker = f"_{category}"
            if marker in scenario_id:
                return scenario_id.split(marker)[0]

        for difficulty in self.config.difficulties:
            marker = f"_{difficulty}"
            if marker in scenario_id:
                prefix = scenario_id.rsplit(marker, 1)[0]
                return prefix.rstrip("_0123456789")

        if len(parts) >= 4:
            return "_".join(parts[:-3])

        logger.error("Could not extract project name from scenario ID: %s", scenario_id)
        return None

    def _normalize_rel_path(self, rel_path: str) -> str:
        return rel_path.replace("//", "/").replace("\\", "/").lstrip("/")

    def _build_candidate_entries(
        self,
        scenario_id: str,
        scenario_data: Dict,
        context_list: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, Path]]:
        _, base_path = self.config.resolve_paths()

        project_name = self._extract_project_name(scenario_id)
        if not project_name:
            return []

        context_files = list(context_list) if context_list is not None else self._normalize_context_files(scenario_data)[0]
        if not context_files:
            logger.warning("No context files defined for scenario %s", scenario_id)
            return []

        base_dir = base_path / project_name
        entries: List[Tuple[str, Path]] = []

        for rel_path in context_files:
            normalized = self._normalize_rel_path(rel_path)
            entries.append((normalized, base_dir / normalized))

        return entries

    def get_candidate_file_paths(
        self,
        scenario_id: str,
        scenario_data: Optional[Dict] = None,
    ) -> List[str]:
        scenario_data = scenario_data or self._load_scenario_json(scenario_id)
        if not scenario_data:
            return []

        context_list, _ = self._normalize_context_files(scenario_data)
        entries = self._build_candidate_entries(scenario_id, scenario_data, context_list)
        return [str(path) for _, path in entries]

    # --------------------------------------------------------------------- #
    # Context loading
    # --------------------------------------------------------------------- #
    def get_context_files(self, scenario_id: str) -> Dict[str, str]:
        scenario_data = self._load_scenario_json(scenario_id)
        if not scenario_data:
            return {}

        context_list, inline_context = self._normalize_context_files(scenario_data)
        if inline_context:
            # The scenario already ships content; no need to hit disk.
            return inline_context

        entries = self._build_candidate_entries(scenario_id, scenario_data, context_list)
        context_files_content: Dict[str, str] = {}

        for rel_path, full_path in entries:
            try:
                if full_path.exists() and full_path.is_file():
                    context_files_content[rel_path] = full_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    logger.debug("Context path does not exist: %s", full_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not read file %s: %s", full_path, exc)

        return context_files_content

    # --------------------------------------------------------------------- #
    # Scoring helpers
    # --------------------------------------------------------------------- #
    def _score_file_by_chunks(self, file_path: Path, task_prompt: str) -> float:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.debug("Failed to read %s: %s", file_path, exc)
            return 0.0

        try:
            chunks = chunk_file_smart(content, max_chunk_size=self.config.chunk_size)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Chunking failed for %s: %s", file_path, exc)
            chunks = [{
                "content": content,
                "chunk_index": 0,
                "start_line": 1,
                "end_line": len(content.splitlines()),
            }]

        if not chunks:
            return 0.0

        limit = max(1, self.config.max_chunks_per_file)
        scores: List[float] = []
        for chunk in chunks[: limit * 2]:
            chunk_content = chunk.get("content", "")
            try:
                scores.append(analyze_chunk_relevance(chunk_content, task_prompt))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Chunk analysis failed: %s", exc)
                scores.append(0.0)

        best_chunk_score = max(scores) if scores else 0.0
        if best_chunk_score > 0:
            return best_chunk_score

        return self._score_by_keywords(content, task_prompt)

    def _score_by_keywords(self, content: str, task_prompt: str) -> float:
        task_words = set(re.findall(r"\b\w+\b", task_prompt.lower()))
        if not task_words:
            return 0.0

        content_words = set(re.findall(r"\b\w+\b", content.lower()))
        matches = len(task_words.intersection(content_words))
        file_size_factor = min(1.0, 1000 / max(len(content), 1))
        return matches * file_size_factor * self.config.keyword_fallback_weight

    def _select_file_by_keywords(
        self,
        task_prompt: str,
        file_paths: Iterable[str],
    ) -> Optional[str]:
        best_file: Optional[str] = None
        best_score = 0.0
        for file_path in list(file_paths)[: self.config.max_files_to_scan]:
            path = Path(file_path)
            if not path.exists():
                continue
            score = self._score_file_by_chunks(path, task_prompt)
            if score > best_score:
                best_score = score
                best_file = file_path
        return best_file

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def select_most_relevant_file(
        self,
        scenario_id: str,
        task_prompt: Optional[str] = None,
    ) -> Optional[str]:
        scenario_data = self._load_scenario_json(scenario_id)
        if not scenario_data:
            return None

        prompt = task_prompt or scenario_data.get("task_prompt", "")
        entries = self._build_candidate_entries(scenario_id, scenario_data)
        if not entries:
            return None

        best_file: Optional[str] = None
        best_score = 0.0

        for _, full_path in entries[: self.config.max_files_to_scan]:
            if not full_path.exists() or not full_path.is_file():
                continue
            score = self._score_file_by_chunks(full_path, prompt)
            if score > best_score:
                best_score = score
                best_file = str(full_path)

        if best_file:
            logger.info("Found most relevant file for %s: %s (score=%.3f)", scenario_id, best_file, best_score)
        else:
            logger.warning("No relevant file found for scenario %s", scenario_id)

        return best_file


# ------------------------------------------------------------------------- #
# Module-level helpers for backwards compatibility
# ------------------------------------------------------------------------- #
def get_context_files_from_scenario(
    scenario_id: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated",
) -> Dict[str, str]:
    agent = ScenarioExtractionAgent(
        ScenarioExtractionConfig(
            scenarios_dir=scenarios_dir,
            base_path=base_path,
        )
    )
    return agent.get_context_files(scenario_id)


def get_most_relevant_file_from_scenario(
    scenario_id: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated",
) -> Optional[str]:
    agent = ScenarioExtractionAgent(
        ScenarioExtractionConfig(
            scenarios_dir=scenarios_dir,
            base_path=base_path,
        )
    )
    return agent.select_most_relevant_file(scenario_id)


@dataclass
class MCPAgentConfig:
    """Parameters for invoking the LangChain-based MCP agent."""

    base_url: str = "http://10.199.178.176:8080/v1"
    api_key: str = "111"
    model: str = "gpt-oss"
    temperature: float = 0.0
    timeout: float = 30.0
    max_prompt_chars: int = 300


def get_most_relevant_file_with_mcp_agent(
    scenario_id: str,
    task_prompt: str,
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated",
    mcp_base_url: Optional[str] = None,
    mcp_api_key: Optional[str] = None,
    mcp_model: Optional[str] = None,
    agent_config: Optional[MCPAgentConfig] = None,
) -> Optional[str]:
    """Invoke a LangChain MCP agent to pick the most relevant file."""
    config = agent_config or MCPAgentConfig()
    agent = ScenarioExtractionAgent(
        ScenarioExtractionConfig(
            scenarios_dir=scenarios_dir,
            base_path=base_path,
        )
    )

    scenario_data = agent._load_scenario_json(scenario_id)
    if not scenario_data:
        return None

    prompt = task_prompt or scenario_data.get("task_prompt", "")
    candidate_paths = agent.get_candidate_file_paths(scenario_id, scenario_data)
    if not candidate_paths:
        logger.warning("No candidate files available for scenario %s", scenario_id)
        return None

    try:
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        logger.warning("LangChain dependencies missing for MCP agent: %s", exc)
        return agent._select_file_by_keywords(prompt, candidate_paths)

    model = ChatOpenAI(
        model=mcp_model or config.model,
        temperature=config.temperature,
        base_url=mcp_base_url or config.base_url,
        api_key=mcp_api_key or config.api_key,
        streaming=False,
        timeout=config.timeout,
    )

    max_prompt = prompt[: config.max_prompt_chars]
    files = candidate_paths

    @tool
    def read_file_chunk(file_index: int, chunk_index: int = 0) -> str:
        if 0 <= file_index < len(files):
            file_path = Path(files[file_index])
            if not file_path.exists():
                return f"Error: File '{file_path}' does not exist"
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_file_smart(content, max_chunk_size=2000)
                if 0 <= chunk_index < len(chunks):
                    chunk = chunks[chunk_index]
                    return (
                        f"File: {file_path.name}\n"
                        f"Chunk {chunk_index + 1}/{len(chunks)}\n"
                        f"Lines {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}\n\n"
                        f"{chunk['content']}"
                    )
                return f"Error: Chunk index {chunk_index} out of range. File has {len(chunks)} chunks."
            except Exception as exc:  # pragma: no cover - defensive
                return f"Error reading file '{file_path}': {exc}"
        return f"Error: Invalid file index {file_index}. Available files: {len(files)}"

    @tool
    def get_file_chunk_count(file_index: int) -> str:
        if 0 <= file_index < len(files):
            file_path = Path(files[file_index])
            if not file_path.exists():
                return f"Error: File '{file_path}' does not exist"
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_file_smart(content, max_chunk_size=2000)
            return f"File {file_index} ({file_path.name}) has {len(chunks)} chunks"
        return f"Error: Invalid file index {file_index}"

    @tool
    def list_available_files() -> str:
        return "\n".join(
            f"{idx}: {Path(path).name} ({path})"
            for idx, path in enumerate(files)
        )

    lc_agent = create_agent(
        model,
        tools=[read_file_chunk, get_file_chunk_count, list_available_files],
        system_prompt=(
            "You are a helpful assistant that finds the most relevant file for a task.\n\n"
            f"Task prompt: {max_prompt}\n\n"
            "Your goal is to identify which file (by index) is most relevant to the task prompt.\n"
            "Files are split into chunks to avoid context overflow.\n"
            "Process:\n"
            "1. Use list_available_files() to see all files.\n"
            "2. For 2-3 promising files use get_file_chunk_count() and read_file_chunk().\n"
            "3. Return ONLY the file index number of the most relevant file."
        ),
    )

    try:
        result = lc_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Find the most relevant file for this task. "
                            "Use list_available_files(), then inspect up to three files "
                            "with get_file_chunk_count() and read_file_chunk(). "
                            "Return only the index number."
                        ),
                    }
                ]
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error invoking MCP agent: %s", exc)
        return agent._select_file_by_keywords(prompt, candidate_paths)

    response_content = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            response_content = str(messages[-1].content)
        else:
            response_content = str(result)
    else:
        response_content = str(result)

    numbers = re.findall(r"\b(\d+)\b", response_content)
    for num_str in numbers:
        idx = int(num_str)
        if 0 <= idx < len(files):
            logger.info("MCP agent selected file index %s: %s", idx, files[idx])
            return files[idx]

    logger.warning("Could not parse file index from MCP agent response: %s", response_content)
    return files[0] if files else None


__all__ = [
    "ScenarioExtractionAgent",
    "ScenarioExtractionConfig",
    "MCPAgentConfig",
    "get_context_files_from_scenario",
    "get_most_relevant_file_from_scenario",
    "get_most_relevant_file_with_mcp_agent",
]
