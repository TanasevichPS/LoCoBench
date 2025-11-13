"""
MCP Tools - Compatibility alias for mcp_scenario_filter

This module provides backward compatibility by re-exporting
the MCP scenario filter functionality.
"""

from .mcp_scenario_filter import (
    LoCoBenchMCPTool,
    create_scenario_filter
)

# Backward compatibility alias
LoCoBenchScenarioFilter = LoCoBenchMCPTool

__all__ = ['LoCoBenchMCPTool', 'LoCoBenchScenarioFilter', 'create_scenario_filter']
