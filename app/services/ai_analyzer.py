import json
import logging
import re
from typing import Tuple, List, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.schemas import AIAnalysisResult
from app.services.circuit_breaker import ai_circuit_breaker

logger = logging.getLogger(__name__)

# One client per timeout so the underlying HTTP connection pool is reused across calls
_ai_clients: dict = {}


def get_ai_client(timeout: float) -> AsyncOpenAI:
    client = _ai_clients.get(timeout)
    if client is None:
        client = AsyncOpenAI(
            base_url=settings.NINEROUTERS_BASE_URL,
            api_key=settings.NINEROUTERS_API_KEY,
            timeout=timeout,
            # Failover and the circuit breaker handle errors; SDK retries honour the
            # gateway's long retry-after (60s+) and stall the crawl's AI slots.
            max_retries=1,
        )
        _ai_clients[timeout] = client
    return client


def message_text(response) -> str:
    """Return the first choice's text, raising a clear error on empty/malformed responses."""
    choices = getattr(response, "choices", None)
    if not choices or choices[0].message is None or not choices[0].message.content:
        # Some upstreams (via 9router) return HTTP 200 with the failure in an "error" body field
        body_error = getattr(response, "error", None) or (getattr(response, "model_extra", None) or {}).get("error")
        if body_error:
            message = body_error.get("message") if isinstance(body_error, dict) else str(body_error)
            raise ValueError(f"Upstream error in 200 response: {message}")
        try:
            raw = response.model_dump_json()
        except Exception:
            raw = repr(response)
        logger.warning(f"Malformed completion from {getattr(response, 'model', '?')}: {raw[:2000]}")
        raise ValueError("Empty or malformed completion response")
    return choices[0].message.content.strip()


async def close_ai_clients():
    for client in _ai_clients.values():
        await client.close()
    _ai_clients.clear()

BASE_SYSTEM_PROMPT = """Bạn là biên tập viên kỹ thuật cho một bản tin công nghệ dành cho lập trình viên Việt Nam. Bài viết đầu vào đã được viết bằng tiếng Việt (tin công nghệ, blog lập trình, kiến trúc phần mềm, kỹ thuật lập trình).
Nhiệm vụ: đánh giá và tóm tắt bài viết CHỈ dựa trên nội dung được cung cấp.

NGUYÊN TẮC CHỐNG BỊA (ưu tiên cao nhất):
- Chỉ dùng thông tin có trong TIÊU ĐỀ và NỘI DUNG bài. Không dùng kiến thức bên ngoài để thêm sự kiện, số liệu, tên sản phẩm, phiên bản, ngày tháng, tên người hay công ty.
- Giữ nguyên số liệu, tên riêng, tên công nghệ và thuật ngữ như bài viết. Không làm tròn, không suy đoán, không "nghe có vẻ hợp lý".
- Nếu nội dung bị cắt ngắn hoặc thiếu ý, chỉ tóm tắt phần có thật. Tuyệt đối không đoán phần còn lại.
- Trường nào bài không đề cập thì để trống: danh sách rỗng [] hoặc null. Để trống luôn tốt hơn là bịa.
- Không biến ý kiến, dự đoán hay quảng cáo trong bài thành sự thật chắc chắn. Nếu bài chỉ nêu dự kiến/tin đồn, hãy giữ đúng sắc thái đó.

Tiêu chí và định dạng:
1. `relevance_score` (số thực 1.0 đến 10.0):
   - Cao (8.0 - 10.0): tin có tác động thực tế tới lập trình viên: bản phát hành, công cụ, nền tảng, mô hình AI mới, sự cố bảo mật lớn, thay đổi kiến trúc hoặc kỹ thuật có thể áp dụng ngay.
   - Trung bình (5.0 - 7.9): tin công nghệ chung, đánh giá sản phẩm, xu hướng có liên quan gián tiếp.
   - Thấp (dưới 5.0): quảng cáo, PR, bài tài trợ, khuyến mãi, tin giải trí/đời sống ít liên quan tới công nghệ hoặc lập trình, bài trùng lặp, nội dung nông.
2. `is_worth_reading` (boolean): true nếu `relevance_score` >= 7.0.
3. `vietnamese_title`: bài đã là tiếng Việt, nên GIỮ NGUYÊN tiêu đề gốc. Chỉ bỏ phần thừa (tên báo, ký hiệu rác) nếu có. Không viết lại theo hướng giật gân hay thêm ý mới.
4. `vietnamese_summary`: 2-4 câu tiếng Việt tự nhiên, nêu đúng điều bài viết nói: chuyện gì xảy ra, ai/cái gì liên quan, ý nghĩa mà bài đã nêu. Không thêm nhận định của riêng bạn.
5. `key_takeaways`: 0-4 ý chính, mỗi ý phải truy ngược được về một câu trong bài. Bài không có ý thực tế thì để [].
6. `new_tech_stack`: chỉ liệt kê công nghệ, framework, thư viện, cơ sở dữ liệu hoặc mẫu kiến trúc được NHẮC TÊN trong bài, kèm mô tả ngắn theo đúng bài. Không có thì [].
7. `tags`: 2-6 thẻ ngắn, chỉ lấy chủ đề thực sự xuất hiện trong bài (ví dụ "AI", "Bảo mật", "PostgreSQL", "Kiến trúc hệ thống", "Mobile").
8. `target_audience`: nhóm người đọc phù hợp suy ra từ nội dung (ví dụ ["Backend Engineer", "Tech Lead"]). Không rõ thì [].
9. `architectural_tradeoffs`: CHỈ điền khi bài thực sự bàn về kiến trúc/kỹ thuật và nêu ưu nhược điểm. Chỉ dùng điểm có trong bài (`pros`, `cons`, `when_not_to_use`, `scalability_bottlenecks`); mục nào bài không nói thì []. Bài tin tức, đánh giá sản phẩm hoặc bài không bàn kiến trúc thì đặt `architectural_tradeoffs` là null.
10. `learning_path`: CHỈ điền khi bài là hướng dẫn/bài học kỹ thuật và có nêu kiến thức nền hay chủ đề nên học tiếp. Ngược lại đặt null.
11. `cluster_topic_key`: slug kebab-case ngắn (chữ thường, không dấu) mô tả sự kiện/chủ đề cốt lõi để gom nhóm bài cùng chủ đề (ví dụ "openai-gpt-5-ra-mat", "postgresql-17-hieu-nang"). Dùng tên riêng có trong bài.

YÊU CẦU NGÔN NGỮ:
Mọi giá trị văn bản dành cho người đọc phải bằng tiếng Việt tự nhiên, trừ tên riêng và thuật ngữ kỹ thuật thường giữ nguyên (API, Kubernetes, PostgreSQL...). Không dùng chữ Hán, chữ Nhật hay ký tự nước ngoài khác.

Chỉ trả về MỘT đối tượng JSON hợp lệ theo schema sau (không bọc markdown, không giải thích thêm):
{
  "relevance_score": 7.5,
  "is_worth_reading": true,
  "target_audience": ["..."],
  "vietnamese_title": "...",
  "vietnamese_summary": "...",
  "cluster_topic_key": "slug-kebab-case",
  "key_takeaways": ["..."],
  "new_tech_stack": [{"name": "...", "category": "...", "desc": "..."}],
  "tags": ["..."],
  "architectural_tradeoffs": null,
  "learning_path": null
}
(`architectural_tradeoffs` khi được điền có dạng {"pros": [], "cons": [], "when_not_to_use": [], "scalability_bottlenecks": []}; `learning_path` có dạng {"prerequisites": [], "recommended_next_topics": []}.)
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


def _build_calibrated_system_prompt(few_shot_examples: Optional[List[dict]] = None) -> str:
    """Inject dynamically learned feedback examples into System Prompt (Module 6)"""
    if not few_shot_examples:
        return BASE_SYSTEM_PROMPT

    examples_text = "\n\nUser Scoring Calibration Examples (Learn from past feedback):\n"
    for ex in few_shot_examples[:3]:
        examples_text += f"- Title: {ex.get('title')}\n  User Preference: {'HIGH VALUE (Thumbs Up)' if ex.get('type') == 'like' else 'LOW RELEVANCE (Thumbs Down)'}\n  Note: {ex.get('note', '')}\n"

    return BASE_SYSTEM_PROMPT + examples_text


# ai_model_used for articles analyzed without an LLM: their text is copied from the
# source untranslated, so it may still be English
EXTRACTIVE_FALLBACK_MODEL = "extractive-fallback"


def extractive_heuristic_fallback(title: str, content: str, url: str) -> AIAnalysisResult:
    """Heuristic fallback when all AI models in the failover chain fail"""
    cleaned_content = re.sub(r"\s+", " ", content).strip()
    summary_words = cleaned_content.split()[:40]
    fallback_summary = " ".join(summary_words) if summary_words else title
    
    # Infer basic tags from title/content
    inferred_tags = []
    lower_text = f"{title} {content}".lower()
    for kw in ["ai", "llm", "backend", "python", "nestjs", "database", "postgres", "cloud", "security", "devops", "kubernetes", "rust", "golang"]:
        if kw in lower_text:
            inferred_tags.append(kw.upper() if kw in ["ai", "llm"] else kw.capitalize())
    if not inferred_tags:
        inferred_tags = ["Technology", "Engineering"]

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")[:50] or "tech-update"

    # Clean out any foreign Hanzi/Kanji script from fallback to uphold language invariant
    clean_summary = re.sub(r"[\u4e00-\u9fff]+", "", fallback_summary).strip()
    clean_title = re.sub(r"[\u4e00-\u9fff]+", "", title).strip() or "Tin tức công nghệ"
    summary_text = clean_summary if clean_summary else clean_title

    return AIAnalysisResult(
        relevance_score=5.5,
        is_worth_reading=False,
        target_audience=["Software Engineer", "Backend Developer"],
        vietnamese_title=clean_title,
        vietnamese_summary=f"Trích xuất tự động: {summary_text}...",
        key_takeaways=[
            "Xem bài viết đầy đủ tại đường dẫn nguồn.",
            "Bản trích xuất tự động đảm bảo dữ liệu không bị thất thoát khi đường truyền AI gặp sự cố.",
        ],
        new_tech_stack=[],
        tags=inferred_tags[:5],
        architectural_tradeoffs=None,
        nestjs_blueprint=None,
        learning_path=None,
        cluster_topic_key=slug,
    )


async def analyze_article_with_9router(
    title: str, content: str, url: str, few_shot_examples: Optional[List[dict]] = None
) -> Tuple[AIAnalysisResult, str]:
    """
    Multi-model AI analysis with Circuit Breaker and Failover Chain:
    Gemini 2.5 Flash -> Claude 3.5 Haiku -> GPT-4o-mini -> Extractive Fallback
    """
    client = get_ai_client(settings.AI_TIMEOUT_BACKGROUND)

    truncated_content = content[:3000] if content else title
    system_prompt = _build_calibrated_system_prompt(few_shot_examples)
    user_prompt = f"""TIÊU ĐỀ BÀI VIẾT: {title}
URL NGUỒN: {url}
NỘI DUNG BÀI VIẾT:
{truncated_content}

Phân tích bài viết theo hướng dẫn hệ thống, chỉ dùng thông tin trong bài trên. Chỉ trả về đối tượng JSON yêu cầu.
"""

    models_to_try = settings.fallback_models_list
    if not models_to_try:
        models_to_try = [settings.AI_MODEL, "gemini/gemini-3.8-flash", "groq/openai/gpt-oss-120b"]

    last_exception = None

    for model_name in models_to_try:
        if not ai_circuit_breaker.can_execute(model_name):
            logger.info(f"⚡ Circuit Breaker OPEN for {model_name}. Skipping to next fallback...")
            continue

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            raw_answer = message_text(response)
            data = _clean_json_response(raw_answer)
            analysis = AIAnalysisResult(**data)
            
            ai_circuit_breaker.record_success(model_name)
            return analysis, model_name

        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            last_exception = err
            logger.warning(f"⚠️ Model {model_name} failed ({err}). Triggering failover...")
            continue

    # All AI providers failed or circuit open -> Extractive Heuristic Fallback
    logger.error(f"🚨 All AI models in failover chain exhausted ({last_exception}). Using Extractive Fallback.")
    fallback_res = extractive_heuristic_fallback(title, content, url)
    return fallback_res, EXTRACTIVE_FALLBACK_MODEL


async def chat_with_article(
    article_title: str,
    article_content: str,
    user_message: str,
    history: list = None,
) -> dict:
    """Multi-turn technical Q&A with Senior NestJS & AI Architect persona."""
    client = get_ai_client(settings.AI_TIMEOUT_INTERACTIVE)

    system_instruction = f"""You are a senior software engineer and technical editor helping a Vietnamese developer understand the following Vietnamese tech article. You are knowledgeable across backend, frontend, mobile, data, AI and system design; do not assume the reader uses any particular stack unless the article does.

ARTICLE TITLE: {article_title}
ARTICLE SUMMARY & CONTENT:
{article_content[:4000]}

Guidelines:
1. Answer technical questions directly and practically based on the article's core concepts.
2. When asked about implementation, give concise, correct code in the language/stack the article or the user mentions (default to TypeScript only if neither does). Say clearly that code is an illustrative example, not taken from the article.
3. Discuss trade-offs and performance edge cases only when relevant to the question; label them as general engineering judgment ("Ngoài bài viết:") rather than claims from the article.
4. Only state facts that appear in the article above. If the user asks for something the article does not cover, say clearly "bài viết không đề cập" and then separate any general knowledge you add by prefixing it with "Ngoài bài viết:". Never invent numbers, versions, names or quotes.
5. Respond in professional, practical, natural Vietnamese. STRICTLY DO NOT use Chinese characters.
6. At the end of your response, provide 2-3 follow-up exploration questions in the following exact format:
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

    # Try models with circuit breaker
    models_to_try = settings.fallback_models_list or [
        settings.AI_MODEL,
        "gemini/gemini-3.8-flash",
        "groq/openai/gpt-oss-120b",
    ]

    for model_name in models_to_try:
        if not ai_circuit_breaker.can_execute(model_name):
            continue
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )
            raw_text = message_text(response)

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

            ai_circuit_breaker.record_success(model_name)
            return {
                "reply": reply_body,
                "suggested_followups": suggested_followups[:3],
            }
        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            continue

    return {
        "reply": "Hiện tại tất cả cổng AI đều đang bận hoặc quá tải. Vui lòng thử lại sau giây lát.",
        "suggested_followups": [
            "Làm sao áp dụng nội dung bài viết này vào dự án thực tế?",
            "Những rủi ro về hiệu năng khi áp dụng là gì?",
        ],
    }


async def generate_weekly_radar_digest(top_articles: list) -> dict:
    """Synthesize cross-article insights into a weekly tech radar intelligence report."""
    client = get_ai_client(settings.AI_TIMEOUT_BACKGROUND)

    articles_summary = []
    for a in top_articles[:15]:
        articles_summary.append(
            f"- [{a.get('source_name', 'Tech')}] {a.get('vietnamese_title') or a.get('title')}: {a.get('vietnamese_summary', '')} (Tags: {', '.join(a.get('tags') or [])})"
        )
    articles_context = "\n".join(articles_summary)

    system_prompt = """Bạn là biên tập viên công nghệ tổng hợp "Radar Công nghệ" hàng tuần cho lập trình viên Việt Nam, dựa trên các bài tiếng Việt đã được tóm tắt.

QUY TẮC CHỐNG BỊA:
- Chỉ rút ra xu hướng từ các bài trong danh sách. Không thêm sự kiện, số liệu, tên sản phẩm hay công ty không có trong dữ liệu, không dùng kiến thức bên ngoài.
- Chỉ nêu một xu hướng khi có ít nhất 2 bài cùng chủ đề hoặc 1 bài thật sự nổi bật; nói rõ khi bằng chứng mỏng. Không khẳng định xu hướng "toàn ngành" từ vài bài.
- Nếu dữ liệu không đủ cho một mục, trả về danh sách rỗng [] cho mục đó. Không viết cho đủ số lượng.
- "status" (Adopt | Trial | Assess | Hold) là đánh giá mức độ sẵn sàng áp dụng; chỉ dùng "Adopt"/"Hold" khi bài nêu rõ cơ sở, mặc định là "Assess".
- Khuyến nghị phải bám vào nội dung bài, không chung chung.

Mọi giá trị bằng tiếng Việt tự nhiên, không dùng chữ Hán/Nhật. Giữ nguyên tên riêng và thuật ngữ kỹ thuật.

Chỉ trả về MỘT đối tượng JSON hợp lệ theo schema:
{
  "week_label": "Báo cáo Radar Công nghệ Tuần này",
  "dominant_trends": [
    {
      "topic": "Tên chủ đề / công nghệ có trong bài",
      "status": "Adopt | Trial | Assess | Hold",
      "summary": "1-2 câu, dựa đúng vào các bài",
      "relevance": "Ý nghĩa với lập trình viên, dựa trên nội dung bài"
    }
  ],
  "architectural_shifts": ["Chỉ khi có bài bàn về thay đổi kiến trúc/kỹ thuật, nếu không để []"],
  "actionable_recommendations": ["Khuyến nghị bám sát nội dung bài; nếu không đủ cơ sở thì []"]
}
"""

    user_prompt = f"""Danh sách các bài nổi bật trong tuần:

{articles_context}

Hãy tạo JSON radar tuần theo hướng dẫn hệ thống, chỉ dựa trên các bài trên.
"""

    models_to_try = settings.fallback_models_list or [
        settings.AI_MODEL,
        "gemini/gemini-3.8-flash",
        "groq/openai/gpt-oss-120b",
    ]
    for model_name in models_to_try:
        if not ai_circuit_breaker.can_execute(model_name):
            continue
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            raw_text = message_text(response)
            data = _clean_json_response(raw_text)
            ai_circuit_breaker.record_success(model_name)
            return data
        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            continue

    # No LLM available: group by the most common tags instead of inventing trends
    tag_groups: dict = {}
    for a in top_articles[:15]:
        title = a.get("vietnamese_title") or a.get("title")
        for tag in (a.get("tags") or [])[:3]:
            tag_groups.setdefault(tag, []).append(title)
    ranked = sorted(tag_groups.items(), key=lambda kv: -len(kv[1]))[:5]
    return {
        "week_label": "Báo cáo Radar Công nghệ Tuần này",
        "dominant_trends": [
            {
                "topic": tag,
                "status": "Assess",
                "summary": f"{len(titles)} bài trong tuần, ví dụ: {titles[0]}",
                "relevance": "Tổng hợp tự động theo thẻ (AI không phản hồi), chưa có phân tích.",
            }
            for tag, titles in ranked
        ],
        "architectural_shifts": [],
        "actionable_recommendations": [],
    }


BLUEPRINT_SYSTEM_PROMPT = """Bạn là kỹ sư phần mềm cấp cao. Hãy đọc bài viết kỹ thuật (tiếng Việt) và đề xuất một bản thiết kế triển khai tham khảo bằng TypeScript/NestJS cho ý tưởng chính của bài.

QUY TẮC CHỐNG BỊA:
- Chỉ triển khai những ý tưởng thực sự có trong bài. Không thêm tính năng, thư viện hay số liệu mà bài không nhắc tới.
- Mã chỉ là ví dụ minh hoạ, không phải mã gốc của bài. Dùng API có thật và ổn định của NestJS/TypeScript; không bịa tên hàm, package hay tùy chọn cấu hình. Không chắc thì ghi chú "// Cần kiểm tra tài liệu chính thức" thay vì đoán.
- Nếu bài không phải bài kỹ thuật có thể triển khai (tin tức, đánh giá sản phẩm...), trả về giá trị ngắn, ghi rõ "Bài viết không có nội dung kiến trúc để triển khai" cho các trường, và để "code_snippet" là chuỗi rỗng.
- Văn bản mô tả bằng tiếng Việt; mã và tên kỹ thuật giữ nguyên tiếng Anh.

Chỉ trả về MỘT đối tượng JSON hợp lệ đúng schema:
{
  "architectural_pattern": "Mẫu kiến trúc phù hợp với nội dung bài (ví dụ Hexagonal / CQRS / Event-Driven / Modular Service)",
  "suggested_module_structure": "src/modules/...\\n├── controllers/\\n├── services/\\n└── ...",
  "code_snippet": "// Đoạn mã TypeScript/NestJS minh hoạ ý chính của bài",
  "database_integration": "Gợi ý về cơ sở dữ liệu/ORM/cache/hàng đợi, chỉ khi bài có liên quan; nếu không thì ghi rõ không áp dụng"
}
"""


async def generate_blueprint_on_demand(
    article_title: str,
    article_content: str,
    vietnamese_summary: Optional[str] = None,
) -> dict:
    """Generate on-demand NestJS code blueprint and architectural pattern for an article."""
    client = get_ai_client(settings.AI_TIMEOUT_BACKGROUND)


    truncated_content = article_content[:4000] if article_content else article_title
    user_prompt = f"""TIÊU ĐỀ: {article_title}
TÓM TẮT: {vietnamese_summary or ''}
NỘI DUNG:
{truncated_content}

Đề xuất bản thiết kế NestJS tham khảo, chỉ dựa trên nội dung bài trên. Chỉ trả về đối tượng JSON.
"""

    models_to_try = settings.fallback_models_list or [
        settings.AI_MODEL,
        "gemini/gemini-3.8-flash",
        "groq/openai/gpt-oss-120b",
    ]


    for model_name in models_to_try:
        if not ai_circuit_breaker.can_execute(model_name):
            continue
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": BLUEPRINT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            raw_text = message_text(response)
            data = _clean_json_response(raw_text)
            ai_circuit_breaker.record_success(model_name)
            return {
                "architectural_pattern": data.get(
                    "architectural_pattern",
                    "Modular Service Pattern with Hexagonal Adapters",
                ),
                "suggested_module_structure": data.get(
                    "suggested_module_structure", "src/modules/feature/"
                ),
                "code_snippet": data.get(
                    "code_snippet", "// Code implementation blueprint"
                ),
                "database_integration": data.get(
                    "database_integration", "PostgreSQL / Prisma ORM"
                ),
            }
        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            continue

    # Fallback if AI providers unavailable
    clean_mod_name = (
        re.sub(r"[^a-zA-Z0-9]", "", article_title.lower())[:15] or "feature"
    )
    return {
        "architectural_pattern": "Modular Service Pattern (Fallback Blueprint)",
        "suggested_module_structure": f"src/modules/{clean_mod_name}/\\n├── {clean_mod_name}.controller.ts\\n├── {clean_mod_name}.service.ts\\n├── dto/\\n└── entities/",
        "code_snippet": f"""import {{ Injectable, Logger }} from '@nestjs/common';

@Injectable()
export class {clean_mod_name.capitalize()}Service {{
  private readonly logger = new Logger('{clean_mod_name.capitalize()}Service');

  async executeOperation(): Promise<{{ success: boolean; data: string }}> {{
    this.logger.log('Executing {clean_mod_name} on-demand blueprint logic');
    return {{ success: true, data: 'Initialized {clean_mod_name}' }};
  }}
}}""",
        "database_integration": "PostgreSQL with Prisma ORM / pgvector extension.",
    }

