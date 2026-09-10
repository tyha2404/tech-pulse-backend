import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.schemas import AIAnalysisResult

SYSTEM_PROMPT = """Bạn là một Chuyên gia Công nghệ Cấp cao kiêm Kiến trúc sư Hệ thống (Principal Backend & AI Systems Engineer).
Nhiệm vụ của bạn là đọc nội dung bài viết kỹ thuật và phân tích cho cộng đồng kỹ sư Backend & AI.

Bạn cần:
1. Đánh giá `relevance_score` (thang điểm 1.0 - 10.0) dựa trên mức độ chuyên sâu kỹ thuật, tính thực tiễn, tính thời sự (về AI/LLMs, Distributed Systems, Database, Cloud Native, Tech Stack mới). Trừ điểm nặng các bài quảng cáo sản phẩm hời hợt, tin giật gân, hoặc bài PR nông cạn.
2. Xác định `is_worth_reading` (true nếu relevance_score >= 7.0).
3. Đặt `vietnamese_title`: Tiêu đề tiếng Việt ngắn gọn, chuyên nghiệp, hấp dẫn cho kỹ sư.
4. Viết `vietnamese_summary`: Tóm tắt 3-5 câu cô đọng giá trị cốt lõi nhất.
5. Rút ra `key_takeaways`: Danh sách 3-5 bài học/điểm lưu ý kỹ thuật mà kỹ sư Backend/AI cần biết.
6. Trích xuất `new_tech_stack`: Các công nghệ, framework, library, DB, kiến trúc mới xuất hiện trong bài kèm mô tả ngắn.
7. Gắn `tags`: Ví dụ ["AI/LLM", "PostgreSQL", "System Design", "Microservices", "Go", "Python", "Kubernetes", "DevOps"].
8. `target_audience`: Ví dụ ["Backend Engineer", "AI Engineer", "DevOps", "Architect"].

Trả về kết quả DUY NHẤT dưới dạng JSON hợp lệ (không kèm markdown code block thừa, hoặc đặt trong ```json):
{
  "relevance_score": 8.5,
  "is_worth_reading": true,
  "target_audience": ["Backend Engineer", "AI Engineer"],
  "vietnamese_title": "...",
  "vietnamese_summary": "...",
  "key_takeaways": ["...", "..."],
  "new_tech_stack": [{"name": "...", "category": "...", "desc": "..."}],
  "tags": ["..."]
}
"""


async def analyze_article_with_9router(
    title: str, content: str, url: str
) -> AIAnalysisResult:
    # Connect to local 9router gateway
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL, api_key=settings.NINEROUTERS_API_KEY
    )

    prompt = f"""TIÊU ĐỀ BÀI VIẾT: {title}
URL: {url}
NỘI DUNG:
{content[:5000] if content else title}
"""

    try:
        response = await client.chat.completions.create(
            model=settings.AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw_answer = response.choices[0].message.content.strip()

        # Clean JSON if model returns ```json ... ```
        if "```" in raw_answer:
            cleaned = raw_answer.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            raw_answer = cleaned.split("```")[0].strip()

        data = json.loads(raw_answer)
        return AIAnalysisResult(**data)
    except Exception as e:
        # Graceful fallback if 9router is temporarily offline or model fails
        return AIAnalysisResult(
            relevance_score=6.0,
            is_worth_reading=True,
            target_audience=["Developer"],
            vietnamese_title=title,
            vietnamese_summary=f"Bài viết từ {url}. (Lưu ý: 9router AI phân tích trả về lỗi fallback: {str(e)[:100]})",
            key_takeaways=["Xem chi tiết bài viết tại liên kết gốc."],
            new_tech_stack=[],
            tags=["Tech"],
        )
