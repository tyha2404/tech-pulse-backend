import pytest
from app.services.clustering_service import (
    normalize_vietnamese_text,
    compute_title_similarity,
    extract_shingles,
)

def test_normalize_vietnamese_text():
    text = "Mở Đặt Cọc iPhone 18 Pro Max Tại Việt Nam, Giá Cực Tốt!!!"
    normalized = normalize_vietnamese_text(text)
    assert "mo dat coc iphone 18 pro max tai viet nam gia cuc tot" == normalized

def test_compute_title_similarity_matching():
    title1 = "Mở đặt cọc iPhone 18 Pro tại Viettel Store và FPT Shop"
    title2 = "FPT Shop và Viettel Store mở đặt cọc iPhone 18 Pro"
    score = compute_title_similarity(title1, title2)
    assert score >= 0.70

def test_compute_title_similarity_different():
    title1 = "OpenAI ra mắt mô hình GPT-5 đa phương thức"
    title2 = "Thị trường vàng hôm nay tiếp tục tăng mạnh"
    score = compute_title_similarity(title1, title2)
    assert score < 0.20
