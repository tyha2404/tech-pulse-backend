import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.schemas import AIAnalysisResult

SYSTEM_PROMPT = """Bạn là một Chuyên gia Công nghệ Cấp cao kiêm Kiến trúc sư Hệ thống (Principal Backend & AI Systems Engineer), chuyên sâu về NestJS, TypeScript, Microservices, Distributed Systems và Hạ tầng AI (LLMs, Vector DBs, RAG, Agentic Workflows).
Nhiệm vụ của bạn là đọc nội dung bài viết kỹ thuật và phân tích chuyên sâu cho cộng đồng kỹ sư Backend NestJS & AI.

Bạn cần:
1. Đánh giá `relevance_score` (thang điểm 1.0 - 10.0):
   - ƯU TIÊN CAO (8.0 - 10.0): Các cập nhật công nghệ mới thực tiễn, tính năng mới ra mắt của các AI Labs/công ty lớn, xu hướng công nghệ sắp tới (AI Agents, Reasoning Models, Multimodal, On-Device AI), công cụ và ứng dụng thực tế dễ tiếp cận.
   - TRỪ ĐIỂM NẶNG (dưới 5.0): Các bài báo học thuật thuần lý thuyết hàn lâm, ngập tràn công thức toán học/chứng minh định lý phức tạp (như paper ArXiv lý thuyết), khó áp dụng ngay cho thực tế hoặc thiếu tính đại chúng. Đồng thời trừ điểm các bài PR rác, quảng cáo nông cạn.
2. Xác định `is_worth_reading` (true nếu relevance_score >= 7.0).
3. Đặt `vietnamese_title`: Tiêu đề tiếng Việt ngắn gọn, chuyên nghiệp, hấp dẫn, dễ hiểu. TUYỆT ĐỐI KHÔNG sử dụng ký tự tiếng Trung, tiếng Nhật.
4. Viết `vietnamese_summary`: Tóm tắt 3-5 câu cô đọng giá trị cốt lõi nhất bằng 100% tiếng Việt chuẩn, sáng rõ, dễ hiểu, tránh thuật ngữ hàn lâm trừu tượng không cần thiết. NGHIÊM CẤM xuất hiện chữ Hán/ký tự tiếng Trung Quốc trong bản tóm tắt. Trả lời rõ: Công nghệ này giải quyết vấn đề gì và xu hướng sắp tới ra sao?
5. Rút ra `key_takeaways`: Danh sách 3-5 bài học/điểm lưu ý kỹ thuật mà kỹ sư Backend/AI cần biết (100% tiếng Việt).
6. Trích xuất `new_tech_stack`: Các công nghệ, framework, library, DB, kiến trúc mới xuất hiện trong bài kèm mô tả ngắn.
7. Gắn `tags`: Ví dụ ["NestJS", "AI/LLM", "PostgreSQL", "pgvector", "System Design", "Microservices", "TypeScript"].
8. `target_audience`: Ví dụ ["Backend NestJS Engineer", "AI Systems Engineer", "Tech Lead"].
9. Phân tích `architectural_tradeoffs` (Đánh đổi kiến trúc):
   - `pros`: Ưu điểm kỹ thuật thực tế.
   - `cons`: Nhược điểm, chi phí vận hành, tài nguyên.
   - `when_not_to_use`: Các trường hợp cụ thể KHÔNG NÊN áp dụng để tránh over-engineering hoặc lãng phí tài nguyên.
   - `scalability_bottlenecks`: Điểm nghẽn hiệu năng khi tải cao / dữ liệu phình to.
10. Thiết kế `nestjs_blueprint` (Gợi ý hiện thực hóa trong hệ sinh thái NestJS / Node.js):
   - `architectural_pattern`: Pattern khuyên dùng (ví dụ: "Hexagonal / Ports & Adapters", "CQRS with Event Sourcing", "Repository & Service Pattern").
   - `suggested_module_structure`: Đường dẫn file/thư mục NestJS gợi ý.
   - `code_snippet`: Đoạn code TypeScript / NestJS mẫu (Module/Service/Guard/Interceptor/Prisma) cụ thể, sạch sẽ, chuẩn production.
   - `database_integration`: Gợi ý tích hợp DB (ví dụ: Prisma ORM với pgvector, TypeORM, Redis cache, BullMQ queue).
11. Xây dựng `learning_path`:
   - `prerequisites`: Các kiến thức nền tảng cần biết trước khi đọc bài này.
   - `recommended_next_topics`: Các chủ đề chuyên sâu nên đào sâu tiếp theo.
12. Gán `cluster_topic_key`: Một slug ngắn gọn dạng kebab-case nhận diện sự kiện cốt lõi của bài viết để gom chùm tin (ví dụ: "apple-iphone-18-launch", "deepseek-v3-release", "anthropic-claude-3-7", "postgresql-17-performance").

Trả về kết quả DUY NHẤT dưới dạng JSON hợp lệ (không kèm markdown thừa hoặc đặt trong ```json):
{
  "relevance_score": 8.5,
  "is_worth_reading": true,
  "target_audience": ["Backend NestJS Engineer", "AI Engineer"],
  "vietnamese_title": "...",
  "vietnamese_summary": "...",
  "cluster_topic_key": "slug-kebab-case",
  "key_takeaways": ["...", "..."],
  "new_tech_stack": [{"name": "...", "category": "...", "desc": "..."}],
  "tags": ["..."],
  "architectural_tradeoffs": {
    "pros": ["..."],
    "cons": ["..."],
    "when_not_to_use": ["..."],
    "scalability_bottlenecks": ["..."]
  },
  "nestjs_blueprint": {
    "architectural_pattern": "...",
    "suggested_module_structure": "...",
    "code_snippet": "...",
    "database_integration": "..."
  },
  "learning_path": {
    "prerequisites": ["..."],
    "recommended_next_topics": ["..."]
  }
}
"""


def _clean_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except Exception:
                continue
    # Try direct parse
    return json.loads(cleaned)


async def analyze_article_with_9router(
    title: str, content: str, url: str
) -> AIAnalysisResult:
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL,
        api_key=settings.NINEROUTERS_API_KEY,
        timeout=120.0,
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
        data = _clean_json_response(raw_answer)
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
            architectural_tradeoffs=None,
            nestjs_blueprint=None,
            learning_path=None,
        )


async def chat_with_article(
    article_title: str,
    article_content: str,
    user_message: str,
    history: list = None,
) -> dict:
    """Multi-turn technical Q&A with Senior NestJS & AI Architect persona."""
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL, api_key=settings.NINEROUTERS_API_KEY
    )

    system_instruction = f"""Bạn là một Principal Backend Engineer & AI Systems Architect, chuyên gia về NestJS, TypeScript, Microservices, RAG và AI Engineering.
Bạn đang hỗ trợ một lập trình viên thảo luận và đào sâu về bài viết kỹ thuật sau:

TIÊU ĐỀ: {article_title}
NỘI DUNG TÓM TẮT & BÀI VIẾT:
{article_content[:6000]}

Mục tiêu của bạn:
1. Giải đáp các thắc mắc kỹ thuật thực tế dựa trên nội dung bài viết.
2. Nếu người dùng hỏi cách áp dụng, hãy cung cấp kiến trúc và code TypeScript/NestJS mẫu chuẩn mực (sử dụng dependency injection, DTOs, Decorators, Prisma, Redis, BullMQ hoặc Vercel AI SDK phù hợp).
3. Luôn chỉ rõ trade-offs (ưu nhược điểm), corner cases và điểm nghẽn hiệu năng khi scale.
4. Trả lời bằng tiếng Việt chuyên nghiệp, súc tích, thực chiến.

Cuối câu trả lời, hãy kèm 2-3 câu hỏi gợi ý đào sâu tiếp theo ở định dạng:
FOLLOW_UPS:
- Câu hỏi 1...
- Câu hỏi 2...
- Câu hỏi 3...
"""

    messages = [{"role": "system", "content": system_instruction}]

    if history:
        for msg in history[-6:]:  # Keep recent turns for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        response = await client.chat.completions.create(
            model=settings.AI_MODEL,
            messages=messages,
            temperature=0.4,
        )
        raw_text = response.choices[0].message.content.strip()

        suggested_followups = []
        reply_body = raw_text
        if "FOLLOW_UPS:" in raw_text:
            parts = raw_text.split("FOLLOW_UPS:")
            reply_body = parts[0].strip()
            follow_lines = parts[1].strip().split("\n")
            for line in follow_lines:
                clean_line = line.strip().lstrip("-*•0123456789. ")
                if clean_line:
                    suggested_followups.append(clean_line)

        return {
            "reply": reply_body,
            "suggested_followups": suggested_followups[:3],
        }
    except Exception as e:
        return {
            "reply": f"Hiện tại không thể kết nối tới mô hình AI để trả lời (Chi tiết: {str(e)[:150]}). Vui lòng thử lại sau.",
            "suggested_followups": [
                "Làm sao áp dụng bài viết này vào NestJS?",
                "Những rủi ro về hiệu năng khi áp dụng là gì?",
            ],
        }


async def generate_weekly_radar_digest(top_articles: list) -> dict:
    """Synthesize cross-article insights into a weekly tech radar intelligence report."""
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL,
        api_key=settings.NINEROUTERS_API_KEY,
        timeout=35.0,
    )

    articles_summary = []
    for a in top_articles[:15]:
        articles_summary.append(
            f"- [{a.get('source_name', 'Tech')}] {a.get('vietnamese_title') or a.get('title')}: {a.get('vietnamese_summary', '')} (Tags: {', '.join(a.get('tags') or [])})"
        )
    articles_context = "\n".join(articles_summary)

    prompt = f"""Dưới đây là danh sách các bài viết công nghệ nổi bật nhất trong tuần qua dành cho Backend NestJS & AI Engineers:

{articles_context}

Nhiệm vụ: Tổng hợp thành bản báo cáo "Tech Intelligence & Radar Heatmap" chuyên sâu cho Backend NestJS & AI Engineers.
Yêu cầu trả về JSON hợp lệ (không kèm markdown ngoài JSON):
{{
  "week_label": "Báo cáo Radar Công nghệ Tuần này",
  "dominant_trends": [
    {{
      "topic": "Tên chủ đề / Công nghệ",
      "status": "Adopt | Trial | Assess | Hold",
      "summary": "Tóm tắt ngắn 1-2 câu về đột phá",
      "relevance": "Ý nghĩa đối với Backend NestJS / Distributed Systems"
    }}
  ],
  "architectural_shifts": [
    "Sự dịch chuyển kiến trúc 1...",
    "Sự dịch chuyển kiến trúc 2..."
  ],
  "actionable_recommendations": [
    "Khuyến nghị hành động thực chiến 1 cho backend team...",
    "Khuyến nghị hành động thực chiến 2..."
  ]
}}
"""

    try:
        response = await client.chat.completions.create(
            model=settings.AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là Giám đốc Công nghệ (CTO) & Chief AI Architect. Hãy tổng hợp báo cáo công nghệ chiến lược, sắc bén, hoàn toàn bằng 100% tiếng Việt chuẩn. TUYỆT ĐỐI KHÔNG xuất hiện bất kỳ ký tự chữ Hán/tiếng Trung Quốc nào trong phản hồi.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw_text = response.choices[0].message.content.strip()
        data = _clean_json_response(raw_text)
        return data
    except Exception as e:
        return {
            "week_label": "Báo cáo Radar Công nghệ Tuần này",
            "dominant_trends": [
                {
                    "topic": "Hạ tầng AI & Backend NestJS",
                    "status": "Trial",
                    "summary": f"Tổng hợp cập nhật từ các nguồn tin công nghệ (Fallback: {str(e)[:80]})",
                    "relevance": "Theo dõi các nâng cấp về RAG và async worker.",
                }
            ],
            "architectural_shifts": [
                "Gia tăng tích hợp Vector Database trực tiếp vào hạ tầng backend hiện có",
                "Chuyển dịch sang mô hình Microservices phân tán với hàng đợi BullMQ/Kafka",
            ],
            "actionable_recommendations": [
                "Khảo sát và benchmark pgvector trên PostgreSQL nội bộ",
                "Xây dựng API gateway phân luồng traffic giữa API truyền thống và Agentic LLM flows",
            ],
        }


