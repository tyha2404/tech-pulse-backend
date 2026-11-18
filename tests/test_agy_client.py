import json
import os
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.services import agy_client
from app.services.agy_client import AgyError, RoutedClient, agy_complete, build_prompt, parse_output
from app.services.ai_analyzer import message_text, record_ai_failure
from app.services.circuit_breaker import ai_circuit_breaker


def _fake_agy(tmp_path, body: str):
    script = tmp_path / "fake_agy"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_parse_output_takes_the_json_line_after_warnings():
    out = 'warning: something\n{"status":"SUCCESS","response":"  ok\\n"}\n'
    assert parse_output(out) == "ok"


@pytest.mark.parametrize("out", ["", "no json here", '{"status":"DEADLINE_EXCEEDED"}', '{"status":"SUCCESS","response":""}'])
def test_parse_output_rejects_failures_and_empty(out):
    with pytest.raises(AgyError):
        parse_output(out)


def test_prompt_forbids_tools_and_keeps_roles():
    p = build_prompt([{"role": "system", "content": "SYS"}, {"role": "user", "content": "ART"}])
    assert "không dùng công cụ" in p.lower() and "không đáng tin" in p.lower()
    assert p.index("SYS") < p.index("ART") and p.endswith("[TRẢ LỜI]")


def test_env_passed_to_agy_has_no_backend_secrets(monkeypatch):
    monkeypatch.setenv("DB_PASS", "secret")
    monkeypatch.setenv("NINEROUTERS_API_KEY", "secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    env = agy_client._clean_env()
    assert not {"DB_PASS", "NINEROUTERS_API_KEY", "TELEGRAM_BOT_TOKEN"} & set(env)
    assert "PATH" in env


async def test_agy_complete_runs_cli_in_empty_dir_with_sandbox_flags(tmp_path, monkeypatch):
    log = tmp_path / "args.json"
    cli = _fake_agy(tmp_path, f'''python3 - "$@" <<'PY'
import json, os, sys
json.dump({{"args": sys.argv[1:], "cwd_entries": os.listdir("."), "has_db_pass": "DB_PASS" in os.environ}}, open("{log}", "w"))
print(json.dumps({{"status": "SUCCESS", "response": "xin chào"}}))
PY
''')
    monkeypatch.setattr(settings, "AGY_BIN", cli)
    monkeypatch.setenv("DB_PASS", "secret")
    out = await agy_complete("gemini-4-hydrogen-medium", [{"role": "user", "content": "hi"}], 30)
    assert out == "xin chào"
    seen = json.loads(log.read_text())
    assert "--sandbox" in seen["args"] and "--disable-slash-commands" in seen["args"]
    assert "--dangerously-skip-permissions" not in seen["args"]
    assert seen["args"][seen["args"].index("--model") + 1] == "gemini-4-hydrogen-medium"
    assert seen["cwd_entries"] == [] and seen["has_db_pass"] is False


async def test_agy_complete_nonzero_exit_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "AGY_BIN", _fake_agy(tmp_path, "echo 'not logged in' >&2\nexit 1\n"))
    with pytest.raises(AgyError) as e:
        await agy_complete("m", [{"role": "user", "content": "x"}], 5)
    assert e.value.status_code == 401  # login problems trip the breaker for 30 min

    monkeypatch.setattr(settings, "AGY_BIN", _fake_agy(tmp_path, "sleep 30\n"))
    monkeypatch.setattr(agy_client.asyncio, "wait_for", _short_wait(agy_client.asyncio.wait_for))
    with pytest.raises(AgyError, match="timed out"):
        await agy_complete("m", [{"role": "user", "content": "x"}], 0.2)


def _short_wait(real):
    async def wait(aw, timeout):
        return await real(aw, timeout=0.5)  # skip the +10s grace in tests
    return wait


async def test_missing_binary_trips_breaker(monkeypatch):
    monkeypatch.setattr(settings, "AGY_BIN", "/nonexistent/agy")
    with pytest.raises(AgyError) as e:
        await agy_complete("m", [{"role": "user", "content": "x"}], 5)
    record_ai_failure("agy/m", e.value)
    assert not ai_circuit_breaker.can_execute("agy/m")


async def test_routed_client_sends_agy_models_to_cli_and_others_to_openai(monkeypatch):
    openai = MagicMock()
    openai.chat.completions.create = AsyncMock(return_value="openai-response")
    seen = {}

    async def fake_agy(model, messages, timeout):
        seen["model"] = model
        return "from agy"

    monkeypatch.setattr(agy_client, "agy_complete", fake_agy)
    client = RoutedClient(openai, 30.0)

    r = await client.chat.completions.create(model="agy/gemini-4-hydrogen-medium", messages=[{"role": "user", "content": "x"}], temperature=0.1)
    assert message_text(r) == "from agy" and seen["model"] == "gemini-4-hydrogen-medium"
    openai.chat.completions.create.assert_not_called()

    assert await client.chat.completions.create(model="nexo-chat", messages=[], temperature=0.1) == "openai-response"
    assert client.embeddings is openai.embeddings  # embeddings always stay on 9router
