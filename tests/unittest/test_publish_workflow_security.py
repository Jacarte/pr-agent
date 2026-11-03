import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"
INVALID_VERSION_ERROR = (
    "Version must be a valid SemVer 2.0.0 value (for example, 0.35.0 or 0.35.0-rc.1)."
)


def test_release_publishes_only_cli_and_github_action_images() -> None:
    workflow_text = PUBLISH_WORKFLOW.read_text()
    workflow = yaml.safe_load(workflow_text)
    jobs = workflow["jobs"]

    assert set(jobs) == {"prepare", "publish-docker", "finalize"}
    assert "pypi" not in workflow_text.lower()
    docker_job = jobs["publish-docker"]
    assert docker_job["strategy"]["matrix"] == {
        "include": [
            {"dockerfile": "docker/Dockerfile", "target": "cli", "suffix": "", "rolling": "latest"},
            {
                "dockerfile": "docker/Dockerfile",
                "target": "github_action",
                "suffix": "github_action",
                "rolling": "github_action",
            },
        ]
    }
    push_step = next(step for step in docker_job["steps"] if step.get("id") == "push")
    assert push_step["with"]["push"] is True
    assert push_step["with"]["file"] == "${{ matrix.dockerfile }}"
    assert push_step["with"]["target"] == "${{ matrix.target }}"
    assert push_step["with"]["tags"] == "${{ steps.tags.outputs.tags }}"
    provenance_step = next(
        step for step in docker_job["steps"] if step.get("uses", "").startswith("actions/attest-build-provenance@")
    )
    assert provenance_step["with"]["subject-name"] == "index.docker.io/jacarte/pr-agent"
    assert provenance_step["with"]["subject-digest"] == "${{ steps.push.outputs.digest }}"
    assert provenance_step["with"]["push-to-registry"] is True
    finalize = jobs["finalize"]
    assert finalize["needs"] == ["prepare", "publish-docker"]
    assert "needs.prepare.result == 'success'" in finalize["if"]
    assert "needs.publish-docker.result == 'success'" in finalize["if"]


@pytest.mark.parametrize(
    ("target", "expected_tags"),
    [
        ("cli", "jacarte/pr-agent:1.2.3,jacarte/pr-agent:latest"),
        ("github_action", "jacarte/pr-agent:1.2.3-github_action,jacarte/pr-agent:github_action"),
    ],
)
def test_release_image_tags(tmp_path: Path, target: str, expected_tags: str) -> None:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    docker_job = workflow["jobs"]["publish-docker"]
    image = next(image for image in docker_job["strategy"]["matrix"]["include"] if image["target"] == target)
    tag_step = next(step for step in docker_job["steps"] if step.get("id") == "tags")
    script = tag_step["run"].replace("${{ needs.prepare.outputs.version }}", "1.2.3")
    for key in ("suffix", "rolling"):
        script = script.replace("${{ matrix." + key + " }}", image[key])
    output = tmp_path / "github-output"

    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"tags={expected_tags}\n"


def test_normal_ci_still_builds_and_tests_without_publishing() -> None:
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github" / "workflows" / "build-and-test.yaml").read_text()
    )
    jobs = workflow["jobs"]
    steps = jobs["build-and-test"]["steps"]
    builds = [step for step in steps if step.get("uses", "").startswith("docker/build-push-action@")]

    assert {step["with"]["target"] for step in builds} == {"test", "github_app"}
    assert all(step["with"]["push"] is False and step["with"]["load"] is True for step in builds)
    assert any("tests/packaging/test_help_docs_package.py" in step.get("run", "") for step in steps)
    assert any("pytest -v tests/unittest" in step.get("run", "") for step in steps)
    assert any(step.get("id") == "smoke" for step in steps)
    native = jobs["unit-tests-native"]
    assert native["strategy"]["matrix"]["python-version"] == ["3.12", "3.13", "3.14"]
    assert any("pytest -v tests/unittest" in step.get("run", "") for step in native["steps"])


def test_docs_ci_builds_and_checks_without_publishing() -> None:
    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github" / "workflows" / "docs-ci.yaml").read_text()
    )

    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"docs"}
    docs_job = workflow["jobs"]["docs"]
    assert docs_job["permissions"] == {"contents": "read"}
    steps = docs_job["steps"]
    assert [step["uses"].split("@", 1)[0] for step in steps if "uses" in step] == [
        "actions/checkout", "actions/setup-node"
    ]
    setup_node = next(step for step in steps if step.get("uses", "").startswith("actions/setup-node@"))
    assert setup_node["with"] == {
        "node-version": 22,
        "cache": "npm",
        "cache-dependency-path": "docs/package-lock.json",
    }
    assert [(step["run"], step.get("working-directory")) for step in steps if "run" in step] == [
        ("npm ci", "docs"),
        ("npm run build", "docs"),
        ("python3 scripts/check_docs_urls.py", None),
    ]


def test_release_version_input_is_not_interpolated_into_shell() -> None:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    ref_guard_step = next(
        step
        for step in workflow["jobs"]["prepare"]["steps"]
        if step.get("name") == "Reject workflow_dispatch from non-main ref"
    )
    resolve_step = next(
        step for step in workflow["jobs"]["prepare"]["steps"] if step.get("name") == "Resolve version"
    )

    assert (
        'echo "::error::workflow_dispatch must be triggered from the main branch (got $GITHUB_REF)."'
        in ref_guard_step["run"]
    )
    assert resolve_step["env"]["RELEASE_INPUT_VERSION"] == "${{ inputs.version }}"
    assert 'VERSION="$RELEASE_INPUT_VERSION"' in resolve_step["run"]
    assert '[[ "$VERSION" =~ $SEMVER ]]' in resolve_step["run"]
    assert "grep -P" not in resolve_step["run"]
    assert f"::error::{INVALID_VERSION_ERROR}" in resolve_step["run"]
    assert "::error::SemVer build metadata is not supported for releases." in resolve_step["run"]

    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            assert "inputs.version" not in step.get("run", "")


@pytest.mark.parametrize(
    ("event_name", "ref_name", "input_version", "expected_version", "expected_tag"),
    [
        ("workflow_dispatch", "main", "1.2.3", "1.2.3", "v1.2.3"),
        ("workflow_dispatch", "main", "1.0.0-alpha.1", "1.0.0-alpha.1", "v1.0.0-alpha.1"),
        ("workflow_dispatch", "main", "1.0.0-x.7.z.92", "1.0.0-x.7.z.92", "v1.0.0-x.7.z.92"),
        ("workflow_dispatch", "main", f"1.2.3-{'a' * 97}", f"1.2.3-{'a' * 97}", f"v1.2.3-{'a' * 97}"),
        ("release", "v2.0.0-rc.1", "", "2.0.0-rc.1", "v2.0.0-rc.1"),
    ],
)
def test_release_version_writes_expected_outputs(
    tmp_path: Path,
    event_name: str,
    ref_name: str,
    input_version: str,
    expected_version: str,
    expected_tag: str,
) -> None:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    resolve_step = next(
        step for step in workflow["jobs"]["prepare"]["steps"] if step.get("name") == "Resolve version"
    )
    output = tmp_path / "github-output"
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_REF_NAME": ref_name,
        "GITHUB_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(output),
        "RELEASE_INPUT_VERSION": input_version,
    }

    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", resolve_step["run"]],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output.read_text().splitlines() == [
        f"version={expected_version}",
        f"tag={expected_tag}",
        f"sha={'a' * 40}",
    ]


@pytest.mark.parametrize(
    ("event_name", "ref_name", "input_version", "expected_error"),
    [
        ("workflow_dispatch", "main", "1.2", INVALID_VERSION_ERROR),
        ("workflow_dispatch", "main", "01.2.3", INVALID_VERSION_ERROR),
        ("workflow_dispatch", "main", "1.0.0-01", INVALID_VERSION_ERROR),
        ("workflow_dispatch", "main", "1.0.0-alpha..1", INVALID_VERSION_ERROR),
        (
            "workflow_dispatch",
            "main",
            "1.2.3+build.7",
            "SemVer build metadata is not supported for releases.",
        ),
        (
            "workflow_dispatch",
            "main",
            "1.2.3-rc.1+build.7",
            "SemVer build metadata is not supported for releases.",
        ),
        ("workflow_dispatch", "main", "1.2.3+", INVALID_VERSION_ERROR),
        (
            "release",
            "v1.2.3+build.7",
            "",
            "SemVer build metadata is not supported for releases.",
        ),
        (
            "workflow_dispatch",
            "main",
            f"1.2.3-{'a' * 98}",
            "Version is too long for release image tags (maximum 103 characters).",
        ),
    ],
)
def test_release_version_rejects_unsupported_or_invalid_versions(
    tmp_path: Path,
    event_name: str,
    ref_name: str,
    input_version: str,
    expected_error: str,
) -> None:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    resolve_step = next(
        step for step in workflow["jobs"]["prepare"]["steps"] if step.get("name") == "Resolve version"
    )
    output = tmp_path / "github-output"
    output.write_text("existing-output\n")
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_REF_NAME": ref_name,
        "GITHUB_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(output),
        "RELEASE_INPUT_VERSION": input_version,
    }

    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", resolve_step["run"]],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected_error in result.stdout + result.stderr
    if expected_error == INVALID_VERSION_ERROR:
        assert f"Rejected version: {input_version}" in result.stdout
    assert output.read_text() == "existing-output\n"


@pytest.mark.parametrize("line_break", ["\n", "\r"])
def test_release_version_cannot_inject_workflow_outputs(tmp_path: Path, line_break: str) -> None:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    resolve_step = next(
        step for step in workflow["jobs"]["prepare"]["steps"] if step.get("name") == "Resolve version"
    )
    output = tmp_path / "github-output"
    output.write_text("existing-output\n")
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF_NAME": "main",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(output),
        "RELEASE_INPUT_VERSION": f"1.2.3{line_break}sha=attacker",
    }

    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", resolve_step["run"]],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert output.read_text() == "existing-output\n"


def test_release_finalize_keeps_uv_lock_in_sync() -> None:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text())
    finalize_steps = workflow["jobs"]["finalize"]["steps"]
    bump_step = next(
        step for step in finalize_steps if step.get("name") == "Bump pyproject.toml and sync uv.lock"
    )
    commit_step = next(
        step for step in finalize_steps if step.get("name") == "Commit & push bump (fast-forward only)"
    )

    run = bump_step["run"]
    assert run.index("scripts/set_pyproject_version.py") < run.index("uv lock")
    assert "git diff --quiet pyproject.toml uv.lock" in commit_step["run"]
    assert "git add pyproject.toml uv.lock" in commit_step["run"]
