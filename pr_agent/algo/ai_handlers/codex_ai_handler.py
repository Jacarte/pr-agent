"""Optional Codex/ChatGPT AI backend for PR-Agent.

Install ``openai-codex`` and provision Codex's own auth.json in the runtime.
This handler uses the supported SDK, *not* raw ChatGPT OAuth tokens.

Each completion creates an ephemeral, read-only thread in an empty directory.
Codex may still have read-only command tools available; this is not a security
boundary for credentials. Use only trusted runner code AND trusted PR content.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.run_details import record_ai_call
from pr_agent.config_loader import get_settings


class CodexAIHandler(BaseAiHandler):
    """Implement PR-Agent's async (text, finish_reason) contract using Codex."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def deployment_id(self) -> None:
        return None

    @staticmethod
    def _timeout_seconds() -> float:
        # Prefer the deployment override to PR-Agent's configured timeout.
        raw = os.environ.get("PR_AGENT_CODEX_TIMEOUT_SECONDS")
        if raw is None:
            raw = get_settings().config.get("ai_timeout", 120)
        try:
            seconds = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Codex timeout must be a positive number") from exc
        if not 0 < seconds <= 3600:
            raise ValueError("Codex timeout must be between 0 and 3600 seconds")
        return seconds

    async def chat_completion(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        img_path: str | None = None,
    ) -> tuple[str, str]:
        """Return a response compatible with LiteLLM handlers: (text, 'stop').

        Temperature is intentionally ignored: Codex reasoning models generally
        don't expose a temperature control. Image handling is not implemented.
        """
        if img_path is not None:
            raise NotImplementedError("CodexAIHandler supports text-only prompts")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A Codex model ID is required")

        timeout = self._timeout_seconds()
        try:
            from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
            from openai_codex.types import TurnStatus
        except ImportError as exc:
            raise RuntimeError(
                "Codex backend requires the codex extra: uv sync --extra codex"
            ) from exc

        async def invoke():
            # Start both the runtime and thread outside the checkout. This does
            # not prevent tools from reading other files or inherited credentials.
            with tempfile.TemporaryDirectory(prefix="pr-agent-codex-") as workdir:
                # thread_start initializes lazily; close even if initialization fails or is cancelled.
                codex = AsyncCodex(CodexConfig(cwd=workdir))
                try:
                    thread = await codex.thread_start(
                        model=model,
                        cwd=workdir,
                        ephemeral=True,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                        base_instructions=(
                            "You are a text-only language-model backend for PR-Agent. "
                            "Answer the supplied user prompt precisely. "
                            "Do not run commands, use tools, inspect local files, "
                            "or edit the workspace. Return only the final response."
                        ),
                        developer_instructions=system,
                    )
                    return await thread.run(
                        user,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                    )
                finally:
                    await codex.close()

        try:
            result = await asyncio.wait_for(invoke(), timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError("Codex completion timed out") from exc

        if result.status != TurnStatus.completed:
            # Do not pretend incomplete/failed results are successful responses.
            raise RuntimeError(f"Codex turn did not complete: {result.status}")
        if not result.final_response or not result.final_response.strip():
            raise RuntimeError("Codex completed without a final text response")

        # PR-Agent expects OpenAI/LiteLLM-shaped usage for its own metrics.
        # Use the fresh thread's total, including any intermediate model/tool calls.
        if result.usage is None:
            record_ai_call(model=model)
        else:
            usage = result.usage.total
            record_ai_call(
                {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "cache_read_input_tokens": usage.cached_input_tokens,
                    "cache_creation_input_tokens": usage.cache_write_input_tokens or 0,
                },
                model=model,
            )
        return result.final_response, "stop"
