"""Configuration management for QPO pipeline."""

import os
from dataclasses import dataclass, field


@dataclass
class OllamaConfig:
    """Configuration for Ollama local model servers.

    Attributes:
        local_7b_endpoint: Endpoint for 7B pre-scorer (Predator)
        remote_32b_endpoint: Endpoint for 32B deep evaluator (M1 Pro)
        timeout_s: HTTP timeout for model calls
        retry_count: Number of retries on failure
    """
    local_7b_endpoint: str = "http://localhost:11434"
    remote_32b_endpoint: str = "http://192.168.1.100:11434"  # M1 Pro on LAN
    timeout_s: int = 180
    retry_count: int = 3
    local_7b_model: str = "mistral:7b"
    remote_32b_model: str = "qwen2.5:32b"


@dataclass
class QuantumConfig:
    """Configuration for quantum layer.

    Attributes:
        backend: Which quantum backend to use (lightning|stub|braket-sim|braket-qpu)
        circuit_depth: Maximum circuit depth for QAOA (p parameter)
        num_iterations: Max COBYLA optimization iterations
        seed: Random seed for reproducibility
    """
    backend: str = "lightning"  # Lightning.qubit (CPU sim) for Phase 1
    circuit_depth: int = 1
    num_iterations: int = 30
    seed: int = 42


@dataclass
class PipelineConfig:
    """Configuration for the 5-stage pipeline.

    Attributes:
        max_candidates: Cap on candidate space size (2^N)
        pre_score_batch_size: How many candidates to score at once
        min_feature_axes: Minimum decomposed axes (AC-1.1)
        max_feature_axes: Maximum decomposed axes (AC-1.1)
    """
    max_candidates: int = 512  # Cap at 512 per AC-1.2
    pre_score_batch_size: int = 10
    min_feature_axes: int = 6
    max_feature_axes: int = 14
    pre_score_timeout_s: int = 900  # 15 minutes per AC-1.3
    qaoa_prefilter_size: int = 20  # Pre-filter to top-N before QAOA (limits qubit count)


@dataclass
class Config:
    """Root configuration container.

    Attributes:
        ollama: Ollama server configuration
        quantum: Quantum layer configuration
        pipeline: Pipeline execution configuration
        log_level: Python logging level (DEBUG|INFO|WARNING|ERROR)
        debug: Enable debug mode with extra logging
    """
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # Single endpoint override — routes all LLM traffic to one host (e.g. remote Mac)
        if unified_ep := os.getenv("QPO_OLLAMA_ENDPOINT"):
            config.ollama.local_7b_endpoint = unified_ep
            config.ollama.remote_32b_endpoint = unified_ep
        # Per-model overrides (take precedence over unified)
        if local_ep := os.getenv("QPO_OLLAMA_7B"):
            config.ollama.local_7b_endpoint = local_ep
        if remote_ep := os.getenv("QPO_OLLAMA_32B"):
            config.ollama.remote_32b_endpoint = remote_ep
        if model_7b := os.getenv("QPO_MODEL_7B"):
            config.ollama.local_7b_model = model_7b
        if model_32b := os.getenv("QPO_MODEL_32B"):
            config.ollama.remote_32b_model = model_32b
        if timeout := os.getenv("QPO_OLLAMA_TIMEOUT"):
            config.ollama.timeout_s = int(timeout)

        # Quantum backend from env
        if backend := os.getenv("QPO_QUANTUM_BACKEND"):
            config.quantum.backend = backend
        if depth := os.getenv("QPO_CIRCUIT_DEPTH"):
            config.quantum.circuit_depth = int(depth)
        if iterations := os.getenv("QPO_QAOA_ITERATIONS"):
            config.quantum.num_iterations = int(iterations)
        if prefilter := os.getenv("QPO_QAOA_PREFILTER"):
            config.pipeline.qaoa_prefilter_size = int(prefilter)

        # Log level
        if level := os.getenv("QPO_LOG_LEVEL"):
            config.log_level = level

        config.debug = os.getenv("QPO_DEBUG", "false").lower() == "true"

        return config


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Override the global config (useful for testing)."""
    global _config
    _config = config
