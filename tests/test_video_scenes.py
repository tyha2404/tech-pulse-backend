from PIL import Image

from app.services.video_scenes import H, W, Scene, ease, word_starts


def _seg(image=None, kind="item"):
    return {"kind": kind, "title": "Tiêu đề thử nghiệm khá dài để xuống dòng", "text": "Một câu tóm tắt ngắn gọn.",
            "score": 8.4, "meta": "HN", "image": image, "rank": 2}


def test_word_starts_are_monotonic_and_start_at_zero():
    s = word_starts(["a", "bb", "ccc"])
    assert s[0] == 0.0 and s == sorted(s) and s[-1] < 1.0


def test_ease_clamps():
    assert ease(-1) == 0.0 and ease(2) == 1.0


def test_scene_frames_with_and_without_image():
    img = Image.new("RGB", (1600, 900), "#336699")
    for seg in (_seg(img), _seg(None), _seg(None, "intro")):
        sc = Scene(seg, 0, 3, speech=3.0, duration=3.5)
        for t in (0.0, 0.3, 1.5, 3.4):
            f = sc.frame(t)
            assert f.size == (W, H) and f.mode == "RGB"


def test_frames_change_over_time():
    sc = Scene(_seg(Image.new("RGB", (1600, 900), "#336699")), 0, 3, speech=3.0, duration=3.5)
    assert sc.frame(0.5).tobytes() != sc.frame(2.5).tobytes()
