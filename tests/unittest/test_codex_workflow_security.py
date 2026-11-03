"""Exercise the manual workflow's guards without credentials, inference, or GitHub calls."""

import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/codex-review-manual.yml"


def workflow_job():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["review"]


def step(step_id):
    return next(item for item in workflow_job()["steps"] if item.get("id") == step_id)


def run_step(step_id, env):
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step(step_id)["run"]],
        env={"PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}", **env},
        capture_output=True, text=True, check=False,
    )


def test_manual_workflow_has_minimal_permissions_and_trusted_checkout():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML treats YAML 1.1's unquoted `on` as True.
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert trigger["workflow_dispatch"]["inputs"]["trusted_content"]["default"] is False
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"review"}
    job = workflow_job()
    assert job["if"] == (
        "github.event_name == 'workflow_dispatch' && "
        "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
    )
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {"contents": "read", "pull-requests": "write"}
    assert job["concurrency"] == {"group": "codex-chatgpt-login", "cancel-in-progress": False}
    assert not job.get("env")
    for step_id in ("auth", "review", "cleanup"):
        assert step(step_id)["env"]["CODEX_HOME"] == "${{ runner.temp }}/pr-agent-codex"
    steps = job["steps"]
    assert steps[0]["id"] == "validate"
    actions = [item for item in steps if "uses" in item]
    assert {item["uses"].split("@")[0] for item in actions} == {"actions/checkout", "astral-sh/setup-uv"}
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", item["uses"]) for item in actions)
    checkout, uv = actions
    assert checkout["with"] == {
        "ref": "refs/heads/${{ github.event.repository.default_branch }}", "persist-credentials": False,
    }
    assert uv["with"] == {"version": "0.13.5", "enable-cache": False}
    assert step("install")["run"] == "uv sync --frozen --no-dev --extra codex --extra github --python 3.12"
    ids = [item.get("id") for item in steps]
    assert ids.index("install") < ids.index("auth") < ids.index("review") < ids.index("cleanup")
    for item in steps:
        env = item.get("env", {})
        if "GITHUB_TOKEN" in env:
            assert item["id"] == "review"
            assert env["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
        if "CODEX_AUTH_JSON" in env:
            assert item["id"] == "auth"
        assert "${{" not in item.get("run", "")  # Never interpolate inputs into source code.
    assert step("cleanup")["if"] == "always()"


@pytest.mark.parametrize("default_branch", ["main", "trunk"])
def test_valid_same_repository_url_on_actual_default_branch(default_branch):
    result = run_step("validate", {
        "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": f"refs/heads/{default_branch}",
        "DEFAULT_BRANCH": default_branch, "GITHUB_REPOSITORY": "org/repo",
        "TRUSTED_CONTENT": "true", "PR_URL": "https://github.com/org/repo/pull/123",
    })
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("overrides", [
    {"GITHUB_EVENT_NAME": "pull_request"},
    {"GITHUB_REF": "refs/heads/feature"},
    {"GITHUB_REF": "refs/tags/main"},
    {"TRUSTED_CONTENT": "false"},
    {"PR_URL": "https://github.com/other/repo/pull/1"},
    {"PR_URL": "https://github.com/org/other/pull/1"},
    {"PR_URL": "https://github.com.evil/org/repo/pull/1"},
    {"PR_URL": "https://github.com@evil/org/repo/pull/1"},
    {"PR_URL": "https://user@github.com/org/repo/pull/1"},
    {"PR_URL": "http://github.com/org/repo/pull/1"},
    {"PR_URL": "https://github.com:443/org/repo/pull/1"},
    {"PR_URL": "https://github.com/org/repo/pull/1?token=secret"},
    {"PR_URL": "https://github.com/org/repo/pull/1#issuecomment-1"},
    {"PR_URL": "https://github.com/org/repo/pull/1/"},
    {"PR_URL": "https://github.com/org/repo/pull/1\n"},
    {"PR_URL": "https://github.com/org/repo/pull/0"},
    {"PR_URL": "https://github.com/org/repo/pull/01"},
    {"PR_URL": "https://github.com/org/repo/pull/\u0661"},
    {"PR_URL": "https://github.com/org/repo/pull/../pull/1"},
    {"PR_URL": "https://github.com/org/repo/pull/%31"},
    {"PR_URL": "https://github.com/org/repo/pull/" + "1" * 2048},
    {"PR_URL": "$(echo injected)"},
])
def test_invalid_dispatch_or_url_is_rejected_without_echoing_input(overrides):
    env = {
        "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/heads/main", "DEFAULT_BRANCH": "main",
        "GITHUB_REPOSITORY": "org/repo", "TRUSTED_CONTENT": "true", "PR_URL": "https://github.com/org/repo/pull/1",
        **overrides,
    }
    result = run_step("validate", env)
    assert result.returncode != 0
    assert env["PR_URL"] not in result.stdout + result.stderr


def test_auth_permissions_and_cleanup_of_all_runtime_state(tmp_path):
    home = tmp_path / "codex"
    env = {"CODEX_HOME": str(home), "CODEX_AUTH_JSON": '{"tokens": {"access_token": "fake-secret"}}'}
    restored = run_step("auth", env)
    assert restored.returncode == 0, restored.stderr
    assert "fake-secret" not in restored.stdout + restored.stderr
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert stat.S_IMODE((home / "auth.json").stat().st_mode) == 0o600
    (home / "runtime-state").write_text("rotated fake token")
    cleaned = run_step("cleanup", {"CODEX_HOME": str(home)})
    assert cleaned.returncode == 0
    assert not home.exists()


@pytest.mark.parametrize(
    "payload", ["", "fake-secret-not-json", '{"fake-secret":', "null", "[]", '"fake-secret"', "{}"],
)
def test_bad_auth_fails_without_writing_or_printing_credentials(tmp_path, payload):
    home = tmp_path / "codex"
    restored = run_step("auth", {"CODEX_HOME": str(home), "CODEX_AUTH_JSON": payload})
    assert restored.returncode != 0
    assert not (home / "auth.json").exists()
    assert "fake-secret" not in restored.stdout + restored.stderr
    assert run_step("cleanup", {"CODEX_HOME": str(home)}).returncode == 0
    assert not home.exists()


def test_review_uses_fixed_settings_without_repository_overlays(monkeypatch):
    # Execute only the embedded Python with a stub CLI; never import the real runner or call GitHub/Codex.
    settings = SimpleNamespace(set=Mock())
    cli = ModuleType("pr_agent.cli")
    cli.run_command = Mock(return_value=0)
    loader = ModuleType("pr_agent.config_loader")
    loader.get_settings = lambda: settings
    package = ModuleType("pr_agent")
    package.cli = cli
    monkeypatch.setitem(sys.modules, "pr_agent", package)
    monkeypatch.setitem(sys.modules, "pr_agent.cli", cli)
    monkeypatch.setitem(sys.modules, "pr_agent.config_loader", loader)
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setenv("PR_URL", "https://github.com/org/repo/pull/1")
    script = step("review")["run"].split("<<'PYCODE'\n", 1)[1].rsplit("\nPYCODE", 1)[0]
    with pytest.raises(SystemExit) as result:
        exec(compile(script, str(WORKFLOW), "exec"), {})
    assert result.value.code == 0
    assert {call.args[0]: call.args[1] for call in settings.set.call_args_list} == {
        "github.user_token": "fake-token", "github.deployment_type": "user",
        "config.use_repo_settings_file": False, "config.model": "gpt-5.4", "config.fallback_models": [],
        "config.custom_model_max_tokens": 32000, "config.verbosity_level": 0, "config.propagate_tool_errors": True,
    }
    cli.run_command.assert_called_once_with("https://github.com/org/repo/pull/1", "/review")
    assert step("review")["env"]["PR_AGENT_AI_BACKEND"] == "codex"
    assert step("review")["env"]["LOG_LEVEL"] == "WARNING"
