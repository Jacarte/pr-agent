"""The deployment selector must not change defaults or override injected factories."""

import sys

import pytest

from pr_agent.agent.pr_agent import PRAgent
from pr_agent.algo.ai_handlers.codex_ai_handler import CodexAIHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


@pytest.mark.parametrize("backend", [None, "litellm", " LiteLLM "])
def test_default_backend_is_litellm_without_codex_dependency(monkeypatch, backend):
    monkeypatch.delenv("PR_AGENT_AI_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("PR_AGENT_AI_BACKEND", backend)
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    assert PRAgent().ai_handler is LiteLLMAIHandler


@pytest.mark.parametrize("backend", ["codex", " CODEX "])
def test_codex_opt_in_lazy_loads_sdk(monkeypatch, backend):
    monkeypatch.setenv("PR_AGENT_AI_BACKEND", backend)
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    assert PRAgent().ai_handler is CodexAIHandler


@pytest.mark.parametrize("backend", ["", "other"])
def test_unknown_backend_fails_explicitly(monkeypatch, backend):
    monkeypatch.setenv("PR_AGENT_AI_BACKEND", backend)
    with pytest.raises(ValueError, match="Unsupported PR_AGENT_AI_BACKEND"):
        PRAgent()


@pytest.mark.parametrize("backend", ["litellm", "codex", "invalid"])
def test_explicit_factory_wins_over_selector(monkeypatch, backend):
    monkeypatch.setenv("PR_AGENT_AI_BACKEND", backend)

    class Factory:
        def __bool__(self):
            return False

        def __call__(self):
            return "injected handler"

    factory = Factory()
    assert PRAgent(ai_handler=factory).ai_handler is factory
