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
    ok = {"items": [{"id": 1, "line": "a"}, {"id": 2, "line": "b"}]}
    script = _validate_script(ok, arts)
    assert script["caption"] and "hook" not in script and "cta" not in script
    missing = {"items": [{"id": 1, "line": "a"}]}
    assert _validate_script(missing, arts) is None


def test_validate_script_ignores_unknown_ids_and_garbage():
    arts = [_art(1)]
    data = {"items": [{"id": 99, "line": "x"}, {"id": "zz"}, None, {"id": 1, "line": "a"}]}
    assert [i["id"] for i in _validate_script(data, arts)["items"]] == [1]


def test_fallback_script_and_segments():
    arts = [_art(1), _art(2)]
    segs = build_segments(_fallback_script(arts), arts)
    assert [s["kind"] for s in segs] == ["item", "item"]  # no intro or outro
    assert [s["rank"] for s in segs] == [2, 1]
    assert segs[0]["text"] == "Top 2. Câu một."
    assert "score" not in segs[0]  # the AI score is not shown in the video


def test_segments_carry_source_and_topics():
    art = _art(1)
    art.tags = ["AI", "PostgreSQL", "Backend", "Extra"]
    seg = build_segments(_fallback_script([art]), [art])[0]
    assert seg["meta"] == "HN" and seg["tags"] == ["AI", "PostgreSQL", "Backend"]
    assert "age" not in seg and "reading" not in seg


def test_segment_without_tags_still_builds():
    seg = build_segments(_fallback_script([_art(1)]), [_art(1)])[0]
    assert seg["tags"] == []


def test_spoken_text_cleans_fallback_artifacts():
    from app.services.video_service import spoken_text

    assert spoken_text("Trích xuất tự động: Xin chào bạn....") == "Xin chào bạn."
