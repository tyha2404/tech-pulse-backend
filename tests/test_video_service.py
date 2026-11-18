from types import SimpleNamespace

from app.services.video_service import (
    _fallback_script,
    _validate_script,
    build_segments,
)


def _art(i, score=8.0):
    return SimpleNamespace(
        id=i, url=f"https://example.com/{i}", title=f"T{i}", vietnamese_title=f"Tiêu đề {i}",
        vietnamese_summary="Câu một. Câu hai.", key_takeaways=[],
        relevance_score=score, source=SimpleNamespace(name="HN"),
    )


def test_validate_script_requires_every_article():
    arts = [_art(1), _art(2)]
    ok = {"hook": "Mở đầu", "items": [{"id": 1, "line": "a"}, {"id": 2, "line": "b"}], "cta": ""}
    assert _validate_script(ok, arts)["cta"]  # default CTA filled
    missing = {"hook": "Mở đầu", "items": [{"id": 1, "line": "a"}], "cta": "x"}
    assert _validate_script(missing, arts) is None
    assert _validate_script({"items": ok["items"]}, arts) is None  # no hook


def test_validate_script_ignores_unknown_ids_and_garbage():
    arts = [_art(1)]
    data = {"hook": "h", "items": [{"id": 99, "line": "x"}, {"id": "zz"}, None, {"id": 1, "line": "a"}]}
    assert [i["id"] for i in _validate_script(data, arts)["items"]] == [1]


def test_fallback_script_and_segments():
    arts = [_art(1), _art(2)]
    segs = build_segments(_fallback_script(arts), arts)
    assert [s["kind"] for s in segs] == ["intro", "item", "item", "outro"]
    assert segs[1]["line" if False else "text"] == "Câu một."
    assert segs[1]["score"] == 8.0


def test_unscored_article_has_no_score_badge():
    arts = [_art(1, score=None)]
    segs = build_segments(_fallback_script(arts), arts)
    assert segs[1]["score"] is None


def test_spoken_text_cleans_fallback_artifacts():
    from app.services.video_service import spoken_text

    assert spoken_text("Trích xuất tự động: Xin chào bạn....") == "Xin chào bạn."
