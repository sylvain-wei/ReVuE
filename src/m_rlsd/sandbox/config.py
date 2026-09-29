"""Sandbox configuration and safety settings."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    
    # Timeout settings
    timeout_seconds: int = 5
    max_retries: int = 1
    
    # Memory limits
    max_memory_mb: int = 512
    max_processes: int = 1
    
    # Security settings
    blacklist_modules: List[str] = field(default_factory=lambda: [
        "os.system",
        "subprocess",
        "pickle",
        "marshal",
        "import",
        "__import__",
    ])
    
    whitelist_modules: List[str] = field(default_factory=lambda: [
        "numpy",
        "pandas",
        "matplotlib",
        "json",
        "math",
        "statistics",
        "itertools",
        "collections",
    ])
    
    # Output settings
    max_stdout_chars: int = 10000
    max_stderr_chars: int = 5000
    capture_images: bool = True
    image_output_dir: str = "/tmp/sandbox_outputs"
    
    # Code validation
    allow_imports: bool = True
    allow_eval: bool = False
    allow_exec: bool = False
    
    # Execution strategy
    use_subprocess: bool = False  # True for more safety, False for speed
    deterministic: bool = True
    seed: int = 42


def get_default_config() -> SandboxConfig:
    """Get default sandbox configuration."""
    return SandboxConfig()


def get_safe_config() -> SandboxConfig:
    """Get conservative sandbox configuration."""
    config = SandboxConfig()
    config.timeout_seconds = 3
    config.max_memory_mb = 256
    config.use_subprocess = True
    config.allow_eval = False
    config.allow_exec = False
    return config


def get_fast_config() -> SandboxConfig:
    """Get fast sandbox configuration (less safe)."""
    config = SandboxConfig()
    config.timeout_seconds = 10
    config.max_memory_mb = 1024
    config.use_subprocess = False
    return config
