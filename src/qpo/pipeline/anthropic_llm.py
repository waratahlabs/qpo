"""AWS Bedrock backend for QPO pipeline LLM calls.

Uses boto3 bedrock-runtime (InvokeModel) with the Anthropic Claude model family
on Bedrock. Auth is IAM — no ANTHROPIC_API_KEY needed; credentials come from
the instance profile (EC2/Batch) or ~/.aws/credentials locally.

Backend selection: set QPO_LLM_BACKEND=bedrock to activate.
Region: AWS_DEFAULT_REGION or QPO_AWS_REGION (default: us-east-1).
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Bedrock model ARN prefixes for Claude on Bedrock differ from direct API IDs.
# Map from the short IDs used in config to Bedrock's cross-region inference profile IDs.
_BEDROCK_MODEL_MAP: dict[str, str] = {
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6-20251101-v1:0",
    # Fallbacks for older IDs that might appear in config
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6-20251101": "us.anthropic.claude-sonnet-4-6-20251101-v1:0",
}


def _bedrock_model_id(model: str) -> str:
    """Resolve a short model name to its Bedrock inference profile ID."""
    return _BEDROCK_MODEL_MAP.get(model, model)


def _get_client() -> Any:
    try:
        import boto3  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 not installed. Run: uv add boto3"
        ) from exc
    region = os.getenv("QPO_AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)


def call(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    timeout: int = 180,
) -> str:
    """Invoke a Claude model on AWS Bedrock (Messages API via InvokeModel).

    Uses the Anthropic Messages API shape that Bedrock exposes. Prompt caching
    via cache_control is supported on Bedrock for Claude 3+ models.

    Args:
        model: Short model ID (e.g. "claude-haiku-4-5") — mapped to Bedrock profile ID
        system_prompt: System instruction (cached via ephemeral cache_control)
        user_prompt: Per-call variable content
        max_tokens: Response token ceiling
        timeout: Unused (boto3 uses its own timeout config); kept for interface parity

    Returns:
        Response text string

    Raises:
        RuntimeError: On Bedrock API errors or missing boto3
    """
    client = _get_client()
    bedrock_model = _bedrock_model_id(model)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_prompt}],
    }

    try:
        response = client.invoke_model(
            modelId=bedrock_model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        text = next(
            (block["text"] for block in result.get("content", []) if block.get("type") == "text"),
            "",
        )
        usage = result.get("usage", {})
        logger.debug(
            "Bedrock call model=%s input_tokens=%s output_tokens=%s cached=%s",
            bedrock_model,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("cache_read_input_tokens", 0),
        )
        return text
    except Exception as exc:
        raise RuntimeError(f"Bedrock call failed (model={bedrock_model}): {exc}") from exc
