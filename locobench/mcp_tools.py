"""
MCP Tools - Compatibility alias for mcp_scenario_filter

This module provides backward compatibility by re-exporting
the MCP scenario filter functionality.
"""

from .mcp_scenario_filter import (
    LoCoBenchScenarioFilter,
    create_scenario_filter
)

__all__ = ['LoCoBenchScenarioFilter', 'create_scenario_filter']
