from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import health_monitor


@pytest.fixture(autouse=True)
def clear_health_state():
    health_monitor._active.clear()
    yield
    health_monitor._active.clear()


def _results(**problems):
    base = {name: None for name in health_monitor.CHECK_LABELS}
    base.update(problems)
    return base


async def test_alerts_once_then_reports_recovery():
    send = AsyncMock(return_value=True)
    collect = AsyncMock(return_value=_results(ai_gateway="Không model nào phản hồi."))
    with patch.object(health_monitor, "collect_problems", collect), patch.object(
        health_monitor, "send_telegram_message", send
    ):
        await health_monitor.run_health_check()
        await health_monitor.run_health_check()
        assert send.await_count == 1
        assert "Phát hiện lỗi" in send.call_args.args[0]

        collect.return_value = _results()
        await health_monitor.run_health_check()
        assert send.await_count == 2
        assert "Đã ổn định lại" in send.call_args.args[0]

        await health_monitor.run_health_check()
        assert send.await_count == 2


async def test_reminds_after_interval():
    send = AsyncMock(return_value=True)
    collect = AsyncMock(return_value=_results(crawl_stale="Không có bài mới."))
    with patch.object(health_monitor, "collect_problems", collect), patch.object(
        health_monitor, "send_telegram_message", send
    ), patch.object(health_monitor.settings, "HEALTH_REMIND_HOURS", 0):
        await health_monitor.run_health_check()
        await health_monitor.run_health_check()
        assert send.await_count == 2
        assert "Vẫn còn lỗi" in send.call_args.args[0]


async def test_failed_send_is_retried_next_run():
    send = AsyncMock(return_value=False)
    collect = AsyncMock(return_value=_results(ai_fallback="8/10 bài heuristic."))
    with patch.object(health_monitor, "collect_problems", collect), patch.object(
        health_monitor, "send_telegram_message", send
    ):
        await health_monitor.run_health_check()
        await health_monitor.run_health_check()
        assert send.await_count == 2


async def test_ai_gateway_check_passes_if_any_model_answers():
    client = MagicMock()
    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="ok"))]
    client.chat.completions.create = AsyncMock(side_effect=[RuntimeError("503"), ok])
    with patch.object(health_monitor, "get_ai_client", return_value=client), patch.object(
        health_monitor.settings, "AI_FALLBACK_MODELS", "backup"
    ):
        assert await health_monitor.check_ai_gateway() is None


async def test_ai_gateway_check_reports_all_failures():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("tunnel down"))
    with patch.object(health_monitor, "get_ai_client", return_value=client):
        problem = await health_monitor.check_ai_gateway()
    assert problem and "tunnel down" in problem
