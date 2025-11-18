"""
Utilities for post-processing raw LLM responses before downstream parsing.

The ResponseFilterConfig dataclass captures toggleable rules (strip <think>
sections, enforce code-only outputs, etc.), while ModelResponseFilter applies
those rules to arbitrary LLM responses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ResponseFilterConfig:
    """Configurable rules for cleaning custom-model responses."""

    strip_reasoning_tags: bool = True
    reasoning_tags: List[str] = field(
        default_factory=lambda: ["think", "analysis", "reasoning", "scratchpad"]
    )
    unwrap_output_tags: List[str] = field(
        default_factory=lambda: ["final", "output", "answer", "response", "solution"]
    )
    strip_prefixes: List[str] = field(default_factory=list)
    strip_suffixes: List[str] = field(default_factory=list)
    enforce_code_only: bool = False
    code_only_triggers: List[str] = field(
        default_factory=lambda: [
            "output only the code without any explanations",
            "return only the code",
            "output only valid json",
            "respond with raw code only",
        ]
    )
    trim_whitespace: bool = True
    custom_regex_filters: List[Dict[str, Any]] = field(default_factory=list)


class ModelResponseFilter:
    """Applies configured response-cleanup rules to raw model outputs."""

    _FLAG_MAP = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }

    def __init__(self, config: Optional[ResponseFilterConfig] = None):
        self.config = config or ResponseFilterConfig()
        self._compiled_regex_filters: List[Tuple[re.Pattern, str]] = (
            self._compile_custom_filters(self.config.custom_regex_filters)
        )

    def apply(self, response: Optional[str], prompt: Optional[str] = None) -> Optional[str]:
        """Apply configured filters to the raw response."""
        if response is None:
            return response

        filtered = response

        if self.config.strip_reasoning_tags and self.config.reasoning_tags:
            filtered = self._remove_reasoning_blocks(filtered, self.config.reasoning_tags)

        if self.config.unwrap_output_tags:
            filtered = self._unwrap_tags(filtered, self.config.unwrap_output_tags)

        if self.config.strip_prefixes:
            filtered = self._strip_prefixes(filtered, self.config.strip_prefixes)

        if self.config.strip_suffixes:
            filtered = self._strip_suffixes(filtered, self.config.strip_suffixes)

        if self._compiled_regex_filters:
            filtered = self._apply_custom_regex_filters(filtered)

        if self._should_enforce_code_only(prompt, filtered):
            enforced = self._extract_code_payload(filtered)
            if enforced:
                filtered = enforced

        if self.config.trim_whitespace:
            filtered = filtered.strip()

        return filtered

    # ------------------------------------------------------------------ #
    # Individual filter helpers
    # ------------------------------------------------------------------ #

    def _remove_reasoning_blocks(self, text: str, tags: Sequence[str]) -> str:
        result = text
        for tag in tags:
            if not tag:
                continue
            pattern = re.compile(
                rf"<\s*{re.escape(tag)}\b[^>]*>.*?</\s*{re.escape(tag)}\s*>",
                re.IGNORECASE | re.DOTALL,
            )
            result = pattern.sub("", result)
        return result

    def _unwrap_tags(self, text: str, tags: Sequence[str]) -> str:
        result = text
        for tag in tags:
            if not tag:
                continue
            pattern = re.compile(rf"</?{re.escape(tag)}\b[^>]*>", re.IGNORECASE)
            result = pattern.sub("", result)
        return result

    def _strip_prefixes(self, text: str, prefixes: Sequence[str]) -> str:
        result = text
        for prefix in prefixes:
            if not prefix:
                continue
            pattern = re.compile(rf"^\s*{re.escape(prefix)}\s*", re.IGNORECASE)
            result, count = pattern.subn("", result, count=1)
            if count > 0:
                break  # Only strip the first matching prefix
        return result

    def _strip_suffixes(self, text: str, suffixes: Sequence[str]) -> str:
        result = text
        for suffix in suffixes:
            if not suffix:
                continue
            pattern = re.compile(rf"\s*{re.escape(suffix)}\s*$", re.IGNORECASE)
            result, count = pattern.subn("", result, count=1)
            if count > 0:
                break
        return result

    def _compile_custom_filters(
        self, filters: Sequence[Dict[str, Any]]
    ) -> List[Tuple[re.Pattern, str]]:
        compiled: List[Tuple[re.Pattern, str]] = []
        for entry in filters:
            pattern = entry.get("pattern")
            if not pattern:
                continue
            replacement = entry.get("replacement", "")
            flags_value = 0
            flag_entries = entry.get("flags", [])
            if isinstance(flag_entries, str):
                flag_entries = [flag_entries]
            for flag_name in flag_entries:
                if not isinstance(flag_name, str):
                    continue
                flags_value |= self._FLAG_MAP.get(flag_name.upper(), 0)
            try:
                compiled.append((re.compile(pattern, flags_value), replacement))
            except re.error as exc:
                logger.warning("Invalid regex pattern skipped (%s): %s", pattern, exc)
        return compiled

    def _apply_custom_regex_filters(self, text: str) -> str:
        result = text
        for pattern, replacement in self._compiled_regex_filters:
            result = pattern.sub(replacement, result)
        return result

    def _should_enforce_code_only(self, prompt: Optional[str], response: str) -> bool:
        if not self.config.enforce_code_only:
            return False
        triggers = self.config.code_only_triggers or []
        if not triggers:
            return True
        haystack = " ".join(part for part in [prompt, response] if part).lower()
        return any(trigger.lower() in haystack for trigger in triggers if trigger)

    def _extract_code_payload(self, text: str) -> Optional[str]:
        code_blocks = self._extract_markdown_blocks(text)
        if code_blocks:
            payload = "\n\n".join(block.strip() for block in code_blocks if block.strip())
            if payload:
                return payload

        json_candidate = self._extract_first_json_object(text)
        if json_candidate:
            return json_candidate.strip()

        code_lines = self._collect_code_like_lines(text)
        if code_lines:
            return "\n".join(code_lines).strip()

        return text.strip() if text.strip() else None

    def _extract_markdown_blocks(self, text: str) -> List[str]:
        return re.findall(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        brace_count = 0
        in_string = False
        escape_next = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start : idx + 1]
        return None

    def _collect_code_like_lines(self, text: str) -> List[str]:
        indicators = [
            "def ",
            "class ",
            "import ",
            "package ",
            "public ",
            "func ",
            "=",
            "{",
            "}",
            "(",
            ")",
            ";",
        ]
        code_lines = []
        for line in text.splitlines():
            if any(token in line for token in indicators):
                code_lines.append(line.rstrip())
        return code_lines

