"""Tests that do not require the optional SDK, a Codex login, or network access."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest

from pr_agent.algo.ai_handlers import codex_ai_handler as adapter


class FakeThread:
    def __init__(self, final_response="valid YAML: yes", status="completed", usage=None):
        self.result = SimpleNamespace(status=status, final_response=final_response, usage=usage)
        self.run = AsyncMock(return_value=self.result)


class FakeCodex:
    def __init__(self, thread):
        self.thread = thread
        self.config = None
        self.started = None
        self.closed = False

    async def close(self):
        assert Path(self.config.cwd).is_dir()
        self.closed = True

    async def thread_start(self, **kwargs):
        self.started = kwargs
        assert Path(kwargs["cwd"]).is_dir()
        assert not list(Path(kwargs["cwd"]).iterdir())
        return self.thread


def install_fake_sdk(monkeypatch, thread):
    instance = FakeCodex(thread)

    def factory(config):
        instance.config = config
        return instance

    sdk = SimpleNamespace(
        AsyncCodex=factory,
        CodexConfig=SimpleNamespace,
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
        Sandbox=SimpleNamespace(read_only="read_only"),
    )
    monkeypatch.setitem(sys.modules, "openai_codex", sdk)
    monkeypatch.setitem(sys.modules, "openai_codex.types", SimpleNamespace(
        TurnStatus=SimpleNamespace(completed="completed"),
    ))
    monkeypatch.setenv("PR_AGENT_CODEX_TIMEOUT_SECONDS", "3")
    monkeypatch.setattr(adapter, "record_ai_call", Mock())
    return instance


@pytest.mark.parametrize("last", [None, SimpleNamespace(input_tokens=5)])
async def test_success_uses_readonly_ephemeral_thread_and_records_total_usage(monkeypatch, last):
    # A turn can include intermediate model/tool calls; last alone would undercount.
    usage = SimpleNamespace(total=SimpleNamespace(
        input_tokens=100, output_tokens=23, total_tokens=123,
        cached_input_tokens=10, cache_write_input_tokens=None,
    ), last=last)
    thread = FakeThread(usage=usage)
    sdk = install_fake_sdk(monkeypatch, thread)

    result = await adapter.CodexAIHandler().chat_completion("gpt-5.4", "SYSTEM", "USER", temperature=0.9)

    assert result == ("valid YAML: yes", "stop")
    assert sdk.started["model"] == "gpt-5.4"
    assert sdk.started["developer_instructions"] == "SYSTEM"
    assert sdk.started["ephemeral"] is True
    assert sdk.started["sandbox"] == "read_only"
    assert sdk.started["approval_mode"] == "deny_all"
    assert sdk.config.cwd == sdk.started["cwd"]
    assert sdk.closed
    assert not Path(sdk.config.cwd).exists()
    thread.run.assert_awaited_once_with("USER", sandbox="read_only", approval_mode="deny_all")
    adapter.record_ai_call.assert_called_once_with(
        {"prompt_tokens": 100, "completion_tokens": 23, "total_tokens": 123,
         "cache_read_input_tokens": 10, "cache_creation_input_tokens": 0},
        model="gpt-5.4",
    )


async def test_success_without_usage_still_records_call(monkeypatch):
    install_fake_sdk(monkeypatch, FakeThread())
    await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user")
    adapter.record_ai_call.assert_called_once_with(model="gpt-5.4")


@pytest.mark.parametrize("status", ["interrupted", "failed", "inProgress"])
async def test_noncompleted_turn_is_not_treated_as_completion(monkeypatch, status):
    sdk = install_fake_sdk(monkeypatch, FakeThread(status=status))
    with pytest.raises(RuntimeError, match="did not complete"):
        await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user")
    adapter.record_ai_call.assert_not_called()
    assert sdk.closed


@pytest.mark.parametrize("text", [None, "", " \n"])
async def test_missing_text_is_error(monkeypatch, text):
    install_fake_sdk(monkeypatch, FakeThread(final_response=text))
    with pytest.raises(RuntimeError, match="without a final text response"):
        await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user")
    adapter.record_ai_call.assert_not_called()


async def test_image_prompts_fail_explicitly():
    with pytest.raises(NotImplementedError, match="text-only"):
        await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user", img_path="image.png")


@pytest.mark.parametrize("model", [None, "", "  ", 42])
async def test_missing_model_is_error(model):
    with pytest.raises(ValueError, match="model ID"):
        await adapter.CodexAIHandler().chat_completion(model, "system", "user")


async def test_missing_optional_sdk_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    with pytest.raises(RuntimeError, match="uv sync --extra codex"):
        await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user")


@pytest.mark.parametrize("value", ["", "not-a-number", "0", "-1", "3601", "nan", "inf"])
async def test_invalid_timeout_does_not_start_sdk(monkeypatch, value):
    sdk = install_fake_sdk(monkeypatch, FakeThread())
    monkeypatch.setenv("PR_AGENT_CODEX_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="Codex timeout"):
        await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user")
    assert sdk.config is None


def test_timeout_uses_config_unless_environment_overrides(monkeypatch):
    monkeypatch.setattr(adapter, "get_settings", lambda: SimpleNamespace(config={"ai_timeout": 42}))
    monkeypatch.delenv("PR_AGENT_CODEX_TIMEOUT_SECONDS", raising=False)
    assert adapter.CodexAIHandler._timeout_seconds() == 42
    monkeypatch.setenv("PR_AGENT_CODEX_TIMEOUT_SECONDS", "3600")
    assert adapter.CodexAIHandler._timeout_seconds() == 3600


@pytest.mark.parametrize("phase", ["thread_start", "run"])
@pytest.mark.parametrize("failure", ["timeout", "cancel", "exception"])
async def test_cleanup_on_timeout_cancellation_and_sdk_errors(monkeypatch, phase, failure):
    thread = FakeThread()
    sdk = install_fake_sdk(monkeypatch, thread)
    entered = asyncio.Event()

    async def block_or_fail(*args, **kwargs):
        entered.set()
        if failure == "exception":
            raise RuntimeError("SDK failure")
        await asyncio.Event().wait()

    monkeypatch.setattr(sdk if phase == "thread_start" else thread, phase, block_or_fail)
    if failure == "timeout":
        monkeypatch.setenv("PR_AGENT_CODEX_TIMEOUT_SECONDS", "0.02")
    task = asyncio.create_task(adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user"))
    await entered.wait()
    if failure == "cancel":
        task.cancel()
        error, message = asyncio.CancelledError, None
    elif failure == "timeout":
        error, message = TimeoutError, "Codex completion timed out"
    else:
        error, message = RuntimeError, "SDK failure"
    with pytest.raises(error, match=message):
        await task
    assert sdk.closed
    assert not Path(sdk.config.cwd).exists()
    adapter.record_ai_call.assert_not_called()


async def test_installed_sdk_contract_without_starting_runtime(monkeypatch):
    sdk = pytest.importorskip("openai_codex")
    from openai_codex.types import ThreadTokenUsage, TurnStatus

    # Autospec validates actual SDK signatures; only the protocol models execute.
    client = create_autospec(sdk.AsyncCodex, instance=True)
    thread = create_autospec(sdk.AsyncThread, instance=True)
    client.thread_start.return_value = thread
    counts = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
              "cached_input_tokens": 1, "reasoning_output_tokens": 0}
    thread.run.return_value = sdk.TurnResult(
        id="turn", status=TurnStatus.completed, error=None, started_at=None,
        completed_at=None, duration_ms=None, final_response="text", items=[],
        usage=ThreadTokenUsage.model_validate({"last": counts, "total": counts}),
    )
    monkeypatch.setattr(sdk, "AsyncCodex", Mock(return_value=client))
    monkeypatch.setattr(adapter, "record_ai_call", Mock())
    result = await adapter.CodexAIHandler().chat_completion("gpt-5.4", "system", "user")
    assert result == ("text", "stop")
    client.close.assert_awaited_once_with()
    assert client.thread_start.call_args.kwargs["sandbox"] is sdk.Sandbox.read_only
    assert client.thread_start.call_args.kwargs["approval_mode"] is sdk.ApprovalMode.deny_all
    adapter.record_ai_call.assert_called_once_with(
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
         "cache_read_input_tokens": 1, "cache_creation_input_tokens": 0},
        model="gpt-5.4",
    )
