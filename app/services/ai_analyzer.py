import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.schemas import AIAnalysisResult

SYSTEM_PROMPT = """You are a Principal Backend & AI Systems Engineer, specializing in NestJS, TypeScript, Microservices, Distributed Systems, and AI Infrastructure (LLMs, Vector DBs, RAG, Agentic Workflows).
Your task is to analyze the provided technical article and output an in-depth, production-oriented evaluation for software engineers.

Evaluation & Scoring Criteria:
1. `relevance_score` (float between 1.0 and 10.0):
   - HIGH RELEVANCE (8.0 - 10.0): Practical tech breakthroughs, new releases from leading AI labs / tech firms, emerging industry shifts (AI Agents, Reasoning Models, Multimodal, On-Device AI, Distributed Systems), actionable developer tools, and high-impact system patterns.
   - LOW RELEVANCE (below 5.0): Purely theoretical academic papers full of mathematical proofs without real-world software applicability (e.g. dense theoretical ArXiv proofs), shallow marketing/PR noise, sponsored puff pieces, or duplicate fluff.
2. `is_worth_reading` (boolean): true if `relevance_score` >= 7.0, otherwise false.
3. `vietnamese_title`: A concise, professional, engaging Vietnamese title (100% natural Vietnamese). STRICTLY NO Chinese characters (no Hanzi/Kanji).
4. `vietnamese_summary`: 3-5 concise, high-value sentences in 100% natural, fluent Vietnamese summarizing what problem this technology solves and the key implications. STRICTLY FORBIDDEN to include Chinese/Japanese characters.
5. `key_takeaways`: 3-5 practical, bulleted technical lessons for Backend & AI engineers (in 100% Vietnamese).
6. `new_tech_stack`: List of new technologies, frameworks, libraries, databases, or architectural patterns introduced, each with a brief description.
7. `tags`: List of relevant domain tags (e.g. ["NestJS", "AI/LLM", "PostgreSQL", "pgvector", "System Design", "Microservices", "TypeScript"]).
8. `target_audience`: Target engineering personas (e.g. ["Backend NestJS Engineer", "AI Systems Engineer", "Tech Lead"]).
9. `architectural_tradeoffs`:
   - `pros`: Practical technical advantages.
   - `cons`: Operational overhead, costs, or complexity.
   - `when_not_to_use`: Specific anti-patterns or scenarios where adopting this causes over-engineering.
   - `scalability_bottlenecks`: Performance bottlenecks under high load or massive scale.
10. `nestjs_blueprint`: Production-grade implementation blueprint in the NestJS / Node.js ecosystem:
   - `architectural_pattern`: Recommended pattern (e.g., "Hexagonal / Ports & Adapters", "CQRS with Event Sourcing", "Repository & Service Pattern").
   - `suggested_module_structure`: Recommended NestJS folder & file structure.
   - `code_snippet`: Concrete, clean TypeScript/NestJS production code sample (Module/Service/Guard/Interceptor/Prisma/BullMQ).
   - `database_integration`: Practical DB guidance (e.g., Prisma ORM with pgvector, TypeORM, Redis cache, BullMQ).
11. `learning_path`:
   - `prerequisites`: Required foundational knowledge.
   - `recommended_next_topics`: Advanced topics to explore next.
12. `cluster_topic_key`: Short kebab-case slug identifying the core subject/event for topic clustering (e.g., "deepseek-v3-release", "anthropic-claude-3-7", "postgresql-17-performance", "apple-iphone-launch").

CRITICAL LANGUAGE REQUIREMENT:
All textual values intended for users (`vietnamese_title`, `vietnamese_summary`, `key_takeaways`, `pros`, `cons`, etc.) MUST be written in 100% natural, fluent Vietnamese.
STRICTLY FORBIDDEN to output Chinese or Japanese characters in any field.

Return ONLY a valid JSON object matching this schema (do not wrap in markdown or extra commentary):
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
    # Limit timeout to 40.0s to avoid long blocking on slow tunnel/LLM responses
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL,
        api_key=settings.NINEROUTERS_API_KEY,
        timeout=40.0,
    )

    # Truncate content to max 3000 chars to avoid 9routers inference latency and 524 timeouts
    truncated_content = content[:3000] if content else title
    prompt = f"""ARTICLE TITLE: {title}
SOURCE URL: {url}
ARTICLE CONTENT:
{truncated_content}

Analyze the article according to your system instructions. Output ONLY the required JSON object.
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
        # Graceful fallback if 9router times out, is unreachable, or encounters parsing errors
        error_msg = str(e)
        is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
        fallback_summary = (
            f"Tóm tắt nhanh: Bài viết về '{title}' từ nguồn {url}. "
            f"(Hệ thống tự động lưu trữ dự phòng do {'xử lý AI bị timeout' if is_timeout else '9routers AI tạm thời bận'})."
        )
        return AIAnalysisResult(
            relevance_score=6.0,
            is_worth_reading=False,
            target_audience=["Developer", "Backend Engineer"],
            vietnamese_title=title,
            vietnamese_summary=fallback_summary,
            key_takeaways=[
                "Xem chi tiết toàn văn bài viết tại liên kết nguồn.",
                "Hệ thống đã lưu trữ an toàn bài viết để tránh gián đoạn tiến trình cào tin.",
            ],
            new_tech_stack=[],
            tags=["Engineering", "Tech"],
            architectural_tradeoffs=None,
            nestjs_blueprint=None,
            learning_path=None,
            cluster_topic_key=None,
        )


async def chat_with_article(
    article_title: str,
    article_content: str,
    user_message: str,
    history: list = None,
) -> dict:
    """Multi-turn technical Q&A with Senior NestJS & AI Architect persona."""
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL,
        api_key=settings.NINEROUTERS_API_KEY,
        timeout=35.0,
    )

    system_instruction = f"""You are a Principal Backend Engineer & AI Systems Architect, specialized in NestJS, TypeScript, Microservices, RAG, and AI Engineering.
You are assisting a software engineer in discussing and deep-diving into the following technical article:

ARTICLE TITLE: {article_title}
ARTICLE SUMMARY & CONTENT:
{article_content[:4000]}

Guidelines:
1. Answer technical questions directly and practically based on the article's core concepts.
2. When asked about implementation, provide clean, idiomatic NestJS & TypeScript code blueprints (utilizing Dependency Injection, DTOs, Decorators, Prisma, Redis, BullMQ, or Vercel AI SDK where appropriate).
3. Always explain architectural trade-offs, performance edge cases, and scalability bottlenecks.
4. Respond in professional, practical, natural Vietnamese. STRICTLY DO NOT use Chinese characters.
5. At the end of your response, provide 2-3 follow-up exploration questions in the following exact format:
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

    system_prompt = """You are a Chief Technology Officer (CTO) and Chief AI Architect.
Synthesize the provided technical articles into an executive, strategic "Tech Intelligence & Radar Heatmap" report for Backend NestJS & AI Engineers.

CRITICAL LANGUAGE REQUIREMENT:
All output values must be written in 100% natural, fluent Vietnamese.
STRICTLY FORBIDDEN to include Chinese or Japanese characters in the output.

Return ONLY a valid JSON object matching this schema:
{
  "week_label": "Báo cáo Radar Công nghệ Tuần này",
  "dominant_trends": [
    {
      "topic": "Tên chủ đề / Công nghệ",
      "status": "Adopt | Trial | Assess | Hold",
      "summary": "Tóm tắt ngắn 1-2 câu về đột phá",
      "relevance": "Ý nghĩa đối với Backend NestJS / Distributed Systems"
    }
  ],
  "architectural_shifts": [
    "Sự dịch chuyển kiến trúc 1...",
    "Sự dịch chuyển kiến trúc 2..."
  ],
  "actionable_recommendations": [
    "Khuyến nghị hành động thực chiến 1 cho backend team...",
    "Khuyến nghị hành động thực chiến 2..."
  ]
}
"""

    user_prompt = f"""Below is the list of top engineering articles from this week:

{articles_context}

Please generate the weekly tech radar digest JSON following the system instructions.
"""

    try:
        response = await client.chat.completions.create(
            model=settings.AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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


