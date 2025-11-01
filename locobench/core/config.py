"""
Configuration management for LoCoBench
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    """Configuration for LLM APIs"""
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_timeout: float = 60.0
    google_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    # Claude Bearer Token Authentication (replaces AWS credentials)
    claude_bearer_token: Optional[str] = None
    # Custom model support (OpenAI-compatible APIs)
    custom_model_name: Optional[str] = None
    custom_model_base_url: Optional[str] = None
    custom_model_api_key: Optional[str] = None
    custom_model_temperature: float = 0.1
    custom_model_max_tokens: int = 32000
    custom_model_timeout: float = 30.0
    custom_model_client_timeout: float = 45.0
    disable_proxy: bool = False
    
    # Rate limiting settings 
    max_requests_per_minute: int = 1000
    max_concurrent_requests: int = 1000
    
    # Model configurations
    # 🏆 Elite Models for LoCoBench:
    default_model_openai: str = "o3"                                  # ✅ Elite: OpenAI o3 (reasoning model)
    default_model_google: str = "gemini-2.5-pro"                      # ✅ Elite: Gemini 2.5 Pro (1M+ tokens)
    default_model_claude: str = "claude-sonnet-4"                     # ✅ Elite: Claude Sonnet 4 (balanced performance)


@dataclass
class DataConfig:
    """Configuration for data storage"""
    # Local storage
    output_dir: str = "./data/output"
    generated_dir: str = "./data/generated"


@dataclass
class Phase1Config:
    """Configuration for Phase 1: Project Specification Generation"""
    
    # Supported programming languages (optimized for long-context software development)
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "cpp", "java", "c", "csharp", "javascript", "typescript", "go", "rust", "php"
    ])
    
    # Project generation distribution
    projects_per_language: int = 100  # OPTIMAL: 10 languages × 100 = 1,000 total projects
    
    # Complexity levels distribution for synthetic projects
    complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "easy": 0.25,      # 25% easy projects
        "medium": 0.25,    # 25% medium projects  
        "hard": 0.25,      # 25% hard projects
        "expert": 0.25     # 25% expert projects
    })


@dataclass
class Phase2Config:
    """Configuration for Phase 2: Synthetic Codebase Generation"""
    
    # File count constraints per project
    min_files_per_project: int = 10
    max_files_per_project: int = 100
    
    # Generation quality controls (ENFORCED with retry logic)
    min_complexity_score: float = 0.3
    max_complexity_score: float = 1.0
    min_documentation_ratio: float = 0.03


@dataclass 
class Phase3Config:
    """Configuration for Phase 3: Long-Context Evaluation Scenario Creation"""
    
    # Scale parameters
    total_instances: int = 8000  # OPTIMIZED: 10 languages × 100 projects × 8 categories
    
    # Task category distribution
    task_distribution: Dict[str, int] = field(default_factory=lambda: {
        "architectural_understanding": 1000,
        "cross_file_refactoring": 1000, 
        "feature_implementation": 1000,
        "bug_investigation": 1000,
        "multi_session_development": 1000,
        "code_comprehension": 1000,
        "integration_testing": 1000,
        "security_analysis": 1000
    })
    
    # Difficulty distribution
    difficulty_distribution: Dict[str, int] = field(default_factory=lambda: {
        "easy": 2000,      # 10K-100K tokens
        "medium": 2000,    # 100K-200K tokens 
        "hard": 2000,      # 200K-500K tokens
        "expert": 2000     # 500K-1000K tokens
    })
    
    # Context length ranges (min_tokens, max_tokens)
    context_ranges: Dict[str, List[int]] = field(default_factory=lambda: {
        "easy": [10000, 100000],       # Small to medium codebases
        "medium": [100000, 200000],    # Medium to large codebases
        "hard": [200000, 500000],      # Large enterprise codebases
        "expert": [500000, 1000000]    # Massive enterprise systems
    })
    
    # Information coverage requirements
    min_information_coverage: float = 0.20
    coverage_ranges: Dict[str, List[float]] = field(default_factory=lambda: {
        "easy": [0.20, 0.40],
        "medium": [0.40, 0.60],
        "hard": [0.60, 0.80],
        "expert": [0.80, 1.00]
    })


@dataclass
class RetrievalConfig:
    """Configuration for Retrieval-Augmented Generation (RAG)"""
    
    # Enable/disable retrieval mechanism
    enabled: bool = False
    
    # Difficulty levels where retrieval should be applied
    difficulties: List[str] = field(default_factory=lambda: ["hard", "expert"])
    
    # Number of top-K fragments to retrieve
    top_k: int = 5
    
    # Retrieval method: 'embedding' or 'keyword'
    method: str = "embedding"
    
    # Embedding model name (for embedding method)
    model_name: str = "all-MiniLM-L6-v2"
    
    # Chunk size for code splitting (characters)
    chunk_size: int = 512


@dataclass
class Phase4Config:
    """Configuration for Phase 4: Automated Validation & Evaluation"""
    
    # Comprehensive evaluation enabled by default (all 11 metrics)
    comprehensive_evaluation: bool = True
    
    # Metric weights for LCBS (LoCoBench Score) - 4 Evaluation Dimensions
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "software_engineering": 0.40,
        "functional_correctness": 0.30,
        "code_quality": 0.20,
        "longcontext_utilization": 0.10
    })
    
    # Software Engineering Excellence metric weights (within the 40% software_engineering category)
    software_engineering_weights: Dict[str, float] = field(default_factory=lambda: {
        "architectural_coherence": 0.125,
        "dependency_traversal": 0.125,
        "cross_file_reasoning": 0.125,
        "system_thinking": 0.125,
        "robustness": 0.125,
        "comprehensiveness": 0.125,
        "innovation": 0.125,
        "solution_elegance": 0.125
    })
    
    # Long-Context Utilization metric weights (within the 10% longcontext_utilization category)
    longcontext_utilization_weights: Dict[str, float] = field(default_factory=lambda: {
        "information_coverage": 0.50,
        "multi_session_memory": 0.50
    })
    
    # Scoring thresholds
    score_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "excellent": {"min": 4.0, "max": 5.0},
        "good": {"min": 3.0, "max": 4.0},
        "fair": {"min": 2.0, "max": 3.0}, 
        "poor": {"min": 0.0, "max": 2.0}
    })
    
    # Evaluation timeouts (seconds)
    task_timeout: int = 1800
    session_timeout: int = 3600


@dataclass
class Config:
    """Main configuration class"""
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod 
    def from_yaml(cls, config_path: str = None) -> 'Config':
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Extract environment variables for API keys
        api_config = yaml_data.get('api', {}).copy()

        def _apply_env(var_name: str, key: str, cast=None):
            value = os.getenv(var_name)
            if value is None:
                return
            if cast is not None:
                try:
                    value = cast(value)
                except ValueError:
                    # Keep original string if cast fails; validation can surface issue later
                    pass
            api_config[key] = value

        _apply_env('OPENAI_API_KEY', 'openai_api_key')
        _apply_env('OPENAI_BASE_URL', 'openai_base_url')
        _apply_env('OPENAI_TIMEOUT', 'openai_timeout', float)
        _apply_env('GEMINI_API_KEY', 'google_api_key')  # Using GEMINI_API_KEY as set in api.sh
        _apply_env('HUGGINGFACE_TOKEN', 'huggingface_token')
        _apply_env('CLAUDE_BEARER_TOKEN', 'claude_bearer_token')

        # Custom model overrides via environment
        _apply_env('CUSTOM_MODEL_NAME', 'custom_model_name')
        _apply_env('CUSTOM_MODEL_BASE_URL', 'custom_model_base_url')
        _apply_env('CUSTOM_MODEL_API_KEY', 'custom_model_api_key')
        _apply_env('CUSTOM_MODEL_TEMPERATURE', 'custom_model_temperature', float)
        _apply_env('CUSTOM_MODEL_MAX_TOKENS', 'custom_model_max_tokens', int)
        _apply_env('CUSTOM_MODEL_TIMEOUT', 'custom_model_timeout', float)
        _apply_env('CUSTOM_MODEL_CLIENT_TIMEOUT', 'custom_model_client_timeout', float)

        # Allow OPENAI_* env vars to override custom model defaults if dedicated vars not provided
        if 'custom_model_base_url' not in api_config and api_config.get('openai_base_url'):
            api_config['custom_model_base_url'] = api_config['openai_base_url']
        if 'custom_model_api_key' not in api_config and api_config.get('openai_api_key'):
            api_config['custom_model_api_key'] = api_config['openai_api_key']

        disable_proxy_env = os.getenv('OPENAI_DISABLE_PROXY')
        if disable_proxy_env is not None:
            api_config['disable_proxy'] = disable_proxy_env.lower() in {'1', 'true', 'yes', 'on'}
        
        return cls(
            api=APIConfig(**api_config),
            data=DataConfig(**yaml_data.get('data', {})),
            phase1=Phase1Config(**yaml_data.get('phase1', {})),
            phase2=Phase2Config(**yaml_data.get('phase2', {})),
            phase3=Phase3Config(**yaml_data.get('phase3', {})),
            phase4=Phase4Config(**yaml_data.get('phase4', {})),
            retrieval=RetrievalConfig(**yaml_data.get('retrieval', {}))
        )
    
    def create_directories(self):
        """Create necessary directories"""
        Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.generated_dir).mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        output_path = Path(self.data.output_dir)
        (output_path / "scenarios").mkdir(exist_ok=True)
        (output_path / "validation").mkdir(exist_ok=True)
        (output_path / "references").mkdir(exist_ok=True)
    
    def summary(self):
        """Return a human-readable summary of the configuration"""
        return {
            'api_models': f"OpenAI: {self.api.default_model_openai}, Google: {self.api.default_model_google}",
            'rate_limits': f"{self.api.max_requests_per_minute} req/min, {self.api.max_concurrent_requests} concurrent",
            'languages': f"{len(self.phase1.supported_languages)} languages: {', '.join(self.phase1.supported_languages)}",
            'scale': f"{self.phase1.projects_per_language} projects/language = {len(self.phase1.supported_languages) * self.phase1.projects_per_language} total projects",
            'scenarios': f"{self.phase3.total_instances} evaluation scenarios across {len(self.phase3.task_distribution)} task categories",
            'file_constraints': f"{self.phase2.min_files_per_project}-{self.phase2.max_files_per_project} files per project",
            'quality_controls': f"Complexity: {self.phase2.min_complexity_score}-{self.phase2.max_complexity_score}, Docs: {self.phase2.min_documentation_ratio:.2f}+",
            'timeouts': f"Task: {self.phase4.task_timeout}s, Session: {self.phase4.session_timeout}s",
            'storage': f"Output: {self.data.output_dir}, Generated: {self.data.generated_dir}"
        }

    def validate(self):
        """Validate configuration and return list of errors"""
        errors = []
        
        # 1. API Configuration
        if not any([self.api.openai_api_key, self.api.google_api_key]):
            errors.append("At least one API key must be provided (OpenAI or Google)")
        
        # 2. Phase 1 Validation (Project Specification)
        if self.phase1.projects_per_language <= 0:
            errors.append("projects_per_language must be positive")
        
        complexity_sum = sum(self.phase1.complexity_distribution.values())
        if abs(complexity_sum - 1.0) > 0.01:
            errors.append(f"Phase1 complexity_distribution must sum to 1.0, got {complexity_sum}")
        
        if not self.phase1.supported_languages:
            errors.append("At least one supported language must be specified")
        
        # 3. Phase 2 Validation (Codebase Generation)
        if self.phase2.min_files_per_project <= 0:
            errors.append("min_files_per_project must be positive")
        
        if self.phase2.max_files_per_project < self.phase2.min_files_per_project:
            errors.append("max_files_per_project must be >= min_files_per_project")
        
        if not (0 <= self.phase2.min_complexity_score <= self.phase2.max_complexity_score <= 1.0):
            errors.append("Complexity scores must be between 0 and 1, with min <= max")
        
        if not (0 <= self.phase2.min_documentation_ratio <= 1.0):
            errors.append("min_documentation_ratio must be between 0 and 1")
        
        # 4. Phase 3 Validation (Scenario Creation)
        task_total = sum(self.phase3.task_distribution.values())
        if task_total != self.phase3.total_instances:
            errors.append(f"Task distribution sum ({task_total}) must equal total_instances ({self.phase3.total_instances})")
        
        difficulty_total = sum(self.phase3.difficulty_distribution.values())
        if difficulty_total != self.phase3.total_instances:
            errors.append(f"Difficulty distribution sum ({difficulty_total}) must equal total_instances ({self.phase3.total_instances})")
        
        # Validate context ranges
        for difficulty, (min_tokens, max_tokens) in self.phase3.context_ranges.items():
            if min_tokens <= 0 or max_tokens <= 0:
                errors.append(f"Context range for {difficulty} must have positive token counts")
            if min_tokens >= max_tokens:
                errors.append(f"Context range for {difficulty}: min_tokens must be < max_tokens")
        
        if not (0 <= self.phase3.min_information_coverage <= 1.0):
            errors.append("min_information_coverage must be between 0 and 1")
        
        for difficulty, coverage in self.phase3.coverage_ranges.items():
            if not (0 <= coverage[0] <= coverage[1] <= 1.0):
                errors.append(f"coverage_ranges[{difficulty}] must be between 0 and 1, with min <= max")
        
        # 5. Phase 4 Validation (Evaluation)
        metric_weight_sum = sum(self.phase4.metric_weights.values())
        if abs(metric_weight_sum - 1.0) > 0.01:
            errors.append(f"Phase4 metric_weights must sum to 1.0, got {metric_weight_sum}")
        
        se_weight_sum = sum(self.phase4.software_engineering_weights.values())
        if abs(se_weight_sum - 1.0) > 0.01:
            errors.append(f"Phase4 software_engineering_weights must sum to 1.0, got {se_weight_sum}")
        
        lcu_weight_sum = sum(self.phase4.longcontext_utilization_weights.values())
        if abs(lcu_weight_sum - 1.0) > 0.01:
            errors.append(f"Phase4 longcontext_utilization_weights must sum to 1.0, got {lcu_weight_sum}")
        
        # Validate score thresholds
        for grade, thresholds in self.phase4.score_thresholds.items():
            if thresholds["min"] >= thresholds["max"]:
                errors.append(f"Score threshold for {grade}: min must be < max")
            if not (0 <= thresholds["min"] <= 5.0) or not (0 <= thresholds["max"] <= 5.0):
                errors.append(f"Score thresholds for {grade} must be between 0 and 5")
        
        if self.phase4.task_timeout <= 0 or self.phase4.session_timeout <= 0:
            errors.append("Timeout values must be positive")
        
        # 6. Directory Validation
        try:
            Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.data.generated_dir).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            errors.append(f"Permission error creating directories: {e}")
        
        # 7. Cross-Phase Validation
        expected_projects = len(self.phase1.supported_languages) * self.phase1.projects_per_language
        expected_scenarios = expected_projects * len(self.phase3.task_distribution)
        if expected_scenarios != self.phase3.total_instances:
            errors.append(f"Expected scenarios ({expected_scenarios}) based on languages×projects×categories doesn't match total_instances ({self.phase3.total_instances})")
        
        return errors


# Claude model definitions and configuration
CLAUDE_MODELS = {
    # Human-friendly names to Claude model IDs (Bearer Token Authentication)
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",        # Claude Sonnet 4 - Balanced performance
    "claude-sonnet-3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",    # Claude Sonnet 3.7 - Hybrid reasoning
}

def get_claude_model_id(human_name: str) -> str:
    """Convert human-friendly Claude model name to Claude model ID for Bearer token authentication"""
    return CLAUDE_MODELS.get(human_name, CLAUDE_MODELS["claude-sonnet-4"])

def get_model_max_tokens(model_name: str) -> int:
    """Get the maximum output tokens for a specific model (Updated January 2025)"""
    model_name = model_name.lower()
    
    # OpenAI models (Updated with latest 2025 limits)
    if model_name.startswith(("o1", "o3", "o4")):
        return 100000  # o-series supports 100K+ tokens
    elif model_name.startswith("gpt-5"):
        return 128000  # GPT-5 series: 128K tokens (Updated from 50K)
    elif model_name.startswith("gpt-4.1"):
        return 32768   # GPT-4.1 series: 32,768 tokens
    elif model_name.startswith(("gpt-4o", "gpt-4-turbo")):
        return 16384   # GPT-4o/turbo max limit
    elif model_name.startswith("gpt-4"):
        return 8192    # Standard GPT-4 limit
    elif "openai" in model_name or "gpt" in model_name:
        return 8192    # Default OpenAI limit
    
    # Google Gemini models (Updated with latest 2025 limits)
    elif model_name.startswith("gemini-2.5-flash"):
        return 65000   # Gemini 2.5 Flash: 65K tokens (Updated from 8K)
    elif model_name.startswith("gemini-2.5-pro"):
        return 64000   # Gemini 2.5 Pro: 64K tokens (Updated from 8K)
    elif model_name.startswith("gemini-2.5"):
        return 32000   # Other Gemini 2.5 models: 32K tokens
    elif model_name.startswith("gemini-2.0-pro"):
        return 8192    # Gemini 2.0 Pro: 8K tokens
    elif model_name.startswith("gemini-2.0-flash"):
        return 8192    # Gemini 2.0 Flash: 8K tokens
    elif model_name.startswith("gemini-2.0"):
        return 8192    # Other Gemini 2.0 models: 8K tokens
    elif model_name.startswith("gemini-1.5"):
        return 8192    # Gemini 1.5 series: 8K tokens
    elif model_name.startswith("gemini"):
        return 8192    # Standard Gemini limit
    elif "google" in model_name or "gemini" in model_name:
        return 8192    # Default Google limit
    
    # Claude models (via Bearer Token) - Confirmed accurate
    elif model_name.startswith("claude-opus-4") or model_name.startswith("claude-sonnet-4"):
        return 8192    # Claude 4 series optimal limit for code generation
    elif model_name.startswith("claude-3.7"):
        return 8192    # Claude 3.7 Sonnet optimal limit
    elif "claude" in model_name:
        return 8192    # Default Claude limit
    
    # Default fallback
    else:
        return 8192    # Conservative default

