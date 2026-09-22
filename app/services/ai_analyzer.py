import json
import re
from typing import Tuple, List, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.schemas import AIAnalysisResult, FastTriageResult
from app.services.circuit_breaker import ai_circuit_breaker


def has_vietnamese_diacritics(text: str) -> bool:
    """Check if text contains Vietnamese diacritic characters."""
    vietnamese_chars = set("áààảãạăắằẳẵặâấầẩẫậéèèẻẽẹêễếềểễệíìỉĩịóòòỏõọôốồổỗộơờởợỡỡúùủũụưứừửữựýỳỷỹỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊỄẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỜỞỢỠỠÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ")
    return any(c in vietnamese_chars for c in text)


def make_vietnamese_title(title: str) -> str:
    """Ensure title is in Vietnamese. If English, prepend 'Tin: '."""
    if has_vietnamese_diacritics(title):
        return title
    return f"Tin: {title}"

BASE_SYSTEM_PROMPT = """You are a Principal Backend & AI Systems Engineer, specializing in NestJS, TypeScript, Microservices, Distributed Systems, and AI Infrastructure (LLMs, Vector DBs, RAG, Agentic Workflows).
Your task is to analyze the provided technical article and output an in-depth, production-oriented evaluation for software engineers.

Evaluation & Scoring Criteria:
1. `relevance_score` (float between 1.0 and 10.0):
   - HIGH RELEVANCE (8.0 - 10.0): Practical tech breakthroughs, new releases from leading AI labs / tech firms, emerging industry shifts (AI Agents, Reasoning Models, Multimodal, On-Device AI, Distributed Systems), actionable developer tools, and high-impact system patterns.
   - LOW RELEVANCE (below 5.0): Purely theoretical academic papers full of mathematical proofs without real-world software applicability, shallow marketing/PR noise, sponsored puff pieces, or duplicate fluff.
2. `is_worth_reading` (boolean): true if `relevance_score` >= 7.0, otherwise false.
3. `vietnamese_title`: A concise, professional, engaging Vietnamese title (100% natural Vietnamese). STRICTLY NO Chinese characters (no Hanzi/Kanji).
4. `vietnamese_summary`: A comprehensive, in-depth technical synthesis article (2-3 structured paragraphs, 6-10 sentences, ~200-350 words) in 100% natural, fluent Vietnamese crafted specifically for Senior Software & AI Engineers.
   - MUST provide complete, thorough, and high-density technical information with full context, architectural mechanics, and production metrics. Absolutely avoid shallow or truncated summaries.
   - In-Depth 3-Part Paragraph Structure:
     * Paragraph 1 - Context & Core Engineering Problem: Detailed analysis of the technical problem, system bottlenecks, or architectural limitations that this technology or article solves; compare context with traditional approaches.
     * Paragraph 2 - Deep Architectural Mechanism & Technical Solution: Detailed explanation of internal mechanics, underlying components, algorithms, data flows, and critical benchmark or quantitative metrics (if present).
     * Paragraph 3 - System Impact, Trade-offs & Production Recommendations: Quantitative evaluation of performance impact (latency, throughput, memory, IOPS), operational trade-offs (costs, complexity, maintenance overhead), and concrete, actionable production recommendations for backend and AI engineers.
   - Preserve standard international technical terms (e.g., latency, throughput, vacuum, failover, microservices, vector embedding, KV cache, sharding, zero-copy).
   - STRICTLY FORBIDDEN to include Chinese or Japanese characters.
5. `key_takeaways`: 3-5 practical, bulleted technical lessons for Backend & AI engineers (in 100% natural Vietnamese).
6. `new_tech_stack`: List of new technologies, frameworks, libraries, databases, or architectural patterns introduced, each with a brief description (in Vietnamese).
7. `tags`: List of relevant domain tags (e.g. ["NestJS", "AI/LLM", "PostgreSQL", "pgvector", "System Design", "Microservices", "TypeScript"]).
8. `target_audience`: Target engineering personas (e.g. ["Backend NestJS Engineer", "AI Systems Engineer", "Tech Lead"]).
9. `architectural_tradeoffs`:
   - `pros`: Practical technical advantages (in 100% natural Vietnamese).
   - `cons`: Operational overhead, costs, or complexity (in 100% natural Vietnamese).
   - `when_not_to_use`: Specific anti-patterns or scenarios where adopting this causes over-engineering (in 100% natural Vietnamese).
   - `scalability_bottlenecks`: Performance bottlenecks under high load or massive scale (in 100% natural Vietnamese).
10. `nestjs_blueprint`: Production-grade implementation blueprint in the NestJS / Node.js ecosystem:
   - `architectural_pattern`: Recommended pattern (e.g., "Hexagonal / Ports & Adapters", "CQRS with Event Sourcing", "Repository & Service Pattern").
   - `suggested_module_structure`: Recommended NestJS folder & file structure.
   - `code_snippet`: Concrete, clean TypeScript/NestJS production code sample (Module/Service/Guard/Interceptor/Prisma/BullMQ).
   - `database_integration`: Practical DB guidance (e.g., Prisma ORM with pgvector, TypeORM, Redis cache, BullMQ).
11. `learning_path`:
   - `prerequisites`: Required foundational knowledge (in Vietnamese).
   - `recommended_next_topics`: Advanced topics to explore next (in Vietnamese).
12. `cluster_topic_key`: Short kebab-case slug identifying the core subject/event for topic clustering (e.g., "deepseek-v3-release", "anthropic-claude-3-7", "postgresql-17-performance", "apple-iphone-launch").

CRITICAL LANGUAGE REQUIREMENT:
All explanation and textual values intended for users (`vietnamese_title`, `vietnamese_summary`, `key_takeaways`, `pros`, `cons`, `when_not_to_use`, `scalability_bottlenecks`, `prerequisites`, `recommended_next_topics`) MUST be written in 100% natural, fluent Vietnamese with rich technical depth.
Preserve standard international engineering terms (e.g. latency, throughput, vacuum, failover, microservices, KV cache, sharding, zero-copy, prompt engineering) in standard English.
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

FAST_SYSTEM_PROMPT = """You are a Fast Technical News & Architecture Intelligence Analyst.
Your task is to analyze the provided tech news/release and output a high-impact, accurate Vietnamese summary and key takeaways for engineers.

Evaluation & Scoring Criteria:
1. `relevance_score` (float between 1.0 and 10.0): Overall engineering relevance.
2. `is_worth_reading` (boolean): true if `relevance_score` >= 7.0, otherwise false.
3. `vietnamese_title`: A concise, engaging, professional Vietnamese title (100% natural Vietnamese). STRICTLY NO Chinese characters (no Hanzi/Kanji).
4. `vietnamese_summary`: A clear, comprehensive technical synthesis article (2-3 structured paragraphs, 5-8 sentences, ~150-250 words) in 100% natural, fluent Vietnamese.
   - Summarize the news event, the underlying tech update, key features/changes, and practical takeaways.
   - Preserve standard international technical terms (e.g. latency, throughput, vacuum, API, SDK, cache).
   - STRICTLY FORBIDDEN to include Chinese or Japanese characters.
5. `key_takeaways`: 2-4 practical bullet points in 100% natural Vietnamese.
6. `new_tech_stack`: List of technologies, tools, or frameworks introduced, each with a brief description (in Vietnamese).
7. `tags`: Domain tags (e.g. ["AI/LLM", "Cloud", "Database", "DevOps"]).
8. `target_audience`: Target engineering personas (e.g. ["Backend Developer", "DevOps Engineer"]).
9. `cluster_topic_key`: Short kebab-case slug for topic clustering.

CRITICAL LANGUAGE REQUIREMENT:
All explanation and textual values intended for users (`vietnamese_title`, `vietnamese_summary`, `key_takeaways`) MUST be written in 100% natural, fluent Vietnamese.
Preserve standard international engineering terms in English.
STRICTLY FORBIDDEN to output Chinese or Japanese characters in any field.

Return ONLY a valid JSON object matching this schema (do not wrap in markdown or extra commentary):
{
  "relevance_score": 6.5,
  "is_worth_reading": false,
  "target_audience": ["Software Engineer"],
  "vietnamese_title": "...",
  "vietnamese_summary": "...",
  "cluster_topic_key": "slug-kebab-case",
  "key_takeaways": ["...", "..."],
  "new_tech_stack": [{"name": "...", "category": "...", "desc": "..."}],
  "tags": ["..."],
  "architectural_tradeoffs": null,
  "nestjs_blueprint": null,
  "learning_path": null
}
"""


def _clean_json_response(raw_text: str) -> dict:
    """Extract and robustly clean JSON dictionary from LLM completion text"""
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
    try:
        return json.loads(cleaned)
    except Exception:
        # Regex search for first JSON object {...}
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError(f"Failed to parse valid JSON from AI response: {cleaned[:100]}...")


def sanitize_and_validate_analysis(raw_data: dict, fallback_title: str, fallback_content: str, url: str) -> AIAnalysisResult:
    """
    Partial-failure resilient builder: ensures that even if individual sub-objects (like NestJS blueprint)
    have syntax flaws, the high-value summary & takeaways remain intact without blowing up the analysis.
    """
    # 1. Ensure essential numeric and text types
    score = float(raw_data.get("relevance_score", 6.0) or 6.0)
    score = max(1.0, min(10.0, score))
    is_worth = bool(raw_data.get("is_worth_reading", score >= 7.0))
    vi_title = str(raw_data.get("vietnamese_title") or fallback_title).strip()
    vi_summary = str(raw_data.get("vietnamese_summary") or fallback_content[:300]).strip()

    # 2. Resilient nested models parsing
    tradeoffs_data = raw_data.get("architectural_tradeoffs")
    blueprint_data = raw_data.get("nestjs_blueprint")
    learning_data = raw_data.get("learning_path")

    tradeoffs = None
    if isinstance(tradeoffs_data, dict) and tradeoffs_data:
        try:
            from app.schemas.schemas import ArchitecturalTradeoffs
            tradeoffs = ArchitecturalTradeoffs(**tradeoffs_data)
        except Exception:
            tradeoffs = None

    blueprint = None
    if isinstance(blueprint_data, dict) and blueprint_data:
        try:
            from app.schemas.schemas import NestJSBlueprint
            blueprint = NestJSBlueprint(**blueprint_data)
        except Exception:
            blueprint = None

    learning = None
    if isinstance(learning_data, dict) and learning_data:
        try:
            from app.schemas.schemas import LearningPath
            learning = LearningPath(**learning_data)
        except Exception:
            learning = None

    raw_data["relevance_score"] = score
    raw_data["is_worth_reading"] = is_worth
    raw_data["vietnamese_title"] = vi_title
    raw_data["vietnamese_summary"] = vi_summary
    raw_data["architectural_tradeoffs"] = tradeoffs
    raw_data["nestjs_blueprint"] = blueprint
    raw_data["learning_path"] = learning

    return AIAnalysisResult(**raw_data)


def _build_calibrated_system_prompt(
    few_shot_examples: Optional[List[dict]] = None,
    triage_info: Optional[FastTriageResult] = None,
) -> Tuple[str, bool]:
    """Inject dynamically learned feedback examples and select prompt mode based on System-1 Triage."""
    # Determine mode: Deep Engineering (>= 6.0 or default) vs Fast Intelligence (< 6.0)
    deep_mode = True
    if triage_info is not None and triage_info.tech_depth_score < 6.0:
        deep_mode = False

    base_prompt = BASE_SYSTEM_PROMPT if deep_mode else FAST_SYSTEM_PROMPT

    if not few_shot_examples:
        return base_prompt, deep_mode

    examples_text = "\n\nUser Scoring Calibration Examples (Learn from past feedback):\n"
    for ex in few_shot_examples[:3]:
        examples_text += f"- Title: {ex.get('title')}\n  User Preference: {'HIGH VALUE (Thumbs Up)' if ex.get('type') == 'like' else 'LOW RELEVANCE (Thumbs Down)'}\n  Note: {ex.get('note', '')}\n"

    return base_prompt + examples_text, deep_mode


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

    return AIAnalysisResult(
        relevance_score=5.5,
        is_worth_reading=False,
        target_audience=["Software Engineer", "Backend Developer"],
        vietnamese_title=title,
        vietnamese_summary=f"Trích xuất tự động: {fallback_summary[:100]}..." if fallback_summary else "Tin được trích xuất tự động.",
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
    title: str,
    content: str,
    url: str,
    few_shot_examples: Optional[List[dict]] = None,
    triage_info: Optional[FastTriageResult] = None,
) -> Tuple[AIAnalysisResult, str]:
    """
    Multi-model AI analysis with System-1 Dynamic Steering, Circuit Breaker, and Tiered Failover Chain:
    - Deep Mode (tech_depth_score >= 6.0): full architecture synthesis, tradeoffs, and NestJS blueprint.
    - Fast Mode (tech_depth_score < 6.0): high-impact intelligence summary and key takeaways.
    """
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL,
        api_key=settings.NINEROUTERS_API_KEY,
        timeout=35.0,
    )

    truncated_content = content[:3000] if content else title
    system_prompt, deep_mode = _build_calibrated_system_prompt(
        few_shot_examples=few_shot_examples, triage_info=triage_info
    )

    triage_context = ""
    if triage_info:
        triage_context = f"""
SYSTEM-1 TRIAGE SIGNALS:
- Assessed Tech Depth Score: {triage_info.tech_depth_score:.1f}/10.0
- Urgency Level: {triage_info.urgency_level}
- Breaking News Flag: {triage_info.is_breaking_news}
- Suggested Priority: {triage_info.suggested_priority}
"""

    user_prompt = f"""ARTICLE TITLE: {title}
SOURCE URL: {url}
{triage_context}
ARTICLE CONTENT:
{truncated_content}

Analyze the article according to your system instructions. Output ONLY the required JSON object.
"""

    # Tiered model selection based on task mode
    if deep_mode:
        models_to_try = list(settings.deep_models_list)
    else:
        models_to_try = list(settings.fast_models_list)

    for m in settings.fallback_models_list:
        if m not in models_to_try:
            models_to_try.append(m)

    last_exception = None

    for model_name in models_to_try:
        if not ai_circuit_breaker.can_execute(model_name):
            print(f"⚡ Circuit Breaker OPEN for {model_name}. Skipping to next fallback...")
            continue

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            if not response.choices or not response.choices[0].message:
                raise ValueError(f"Empty choices in response from {model_name}")
            raw_answer = (response.choices[0].message.content or "").strip()
            if not raw_answer and hasattr(response.choices[0].message, "reasoning"):
                raw_answer = (getattr(response.choices[0].message, "reasoning", "") or "").strip()
            if not raw_answer:
                raise ValueError(f"Empty content from {model_name}")

            data = _clean_json_response(raw_answer)
            analysis = sanitize_and_validate_analysis(
                raw_data=data,
                fallback_title=title,
                fallback_content=content,
                url=url,
            )
            
            ai_circuit_breaker.record_success(model_name)
            return analysis, model_name

        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            last_exception = err
            print(f"⚠️ Model {model_name} failed ({err}). Triggering failover...")
            continue

    # All AI providers failed or circuit open -> Extractive Heuristic Fallback
    print(f"🚨 All AI models in failover chain exhausted ({last_exception}). Using Extractive Fallback.")
    fallback_res = extractive_heuristic_fallback(title, content, url)
    return fallback_res, "extractive-fallback"


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
4. Respond in professional, practical, natural Vietnamese. STRICTLY DO NOT use Chinese or Japanese characters. Preserve international technical terms in English.
5. At the end of your response, provide 2-3 follow-up exploration questions in Vietnamese in the following exact format:
FOLLOW_UPS:
- [Follow-up question 1 in Vietnamese]
- [Follow-up question 2 in Vietnamese]
- [Follow-up question 3 in Vietnamese]
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
    models_to_try = settings.fallback_models_list or [settings.AI_MODEL, "claude-3-5-haiku", "gpt-4o-mini"]

    for model_name in models_to_try:
        if not ai_circuit_breaker.can_execute(model_name):
            continue
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.4,
            )
            if not response.choices or not response.choices[0].message:
                raise ValueError(f"Empty choices from {model_name}")
            raw_text = (response.choices[0].message.content or "").strip()
            if not raw_text and hasattr(response.choices[0].message, "reasoning"):
                raw_text = (getattr(response.choices[0].message, "reasoning", "") or "").strip()

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
All explanation and textual output values in the JSON (dominant trend summaries, relevance, architectural shifts, actionable recommendations) MUST be written in 100% natural, fluent Vietnamese.
Preserve standard international technical terms (e.g., latency, throughput, vacuum, failover, microservices, KV cache, sharding, zero-copy) in English.
STRICTLY FORBIDDEN to include Chinese or Japanese characters in the output.

Return ONLY a valid JSON object matching this schema:
{
  "week_label": "Báo cáo Radar Công nghệ Tuần này",
  "dominant_trends": [
    {
      "topic": "Technology / Topic Name",
      "status": "Adopt | Trial | Assess | Hold",
      "summary": "1-2 sentence breakthrough summary in Vietnamese",
      "relevance": "Architectural significance for Backend & AI Systems in Vietnamese"
    }
  ],
  "architectural_shifts": [
    "Key architectural shift 1 in Vietnamese...",
    "Key architectural shift 2 in Vietnamese..."
  ],
  "actionable_recommendations": [
    "Actionable production recommendation 1 for backend/AI team in Vietnamese...",
    "Actionable production recommendation 2 for backend/AI team in Vietnamese..."
  ]
}
"""

    user_prompt = f"""Below is the list of top engineering articles from this week:

{articles_context}

Please generate the weekly tech radar digest JSON following the system instructions.
"""

    models_to_try = settings.fallback_models_list or [settings.AI_MODEL, "claude-3-5-haiku", "gpt-4o-mini"]
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
                temperature=0.3,
            )
            if not response.choices or not response.choices[0].message:
                raise ValueError(f"Empty choices from {model_name}")
            raw_text = (response.choices[0].message.content or "").strip()
            if not raw_text and hasattr(response.choices[0].message, "reasoning"):
                raw_text = (getattr(response.choices[0].message, "reasoning", "") or "").strip()
            if not raw_text:
                raise ValueError(f"Empty content from {model_name}")
            data = _clean_json_response(raw_text)
            ai_circuit_breaker.record_success(model_name)
            return data
        except Exception as err:
            ai_circuit_breaker.record_failure(model_name)
            continue

    return {
        "week_label": "Báo cáo Radar Công nghệ Tuần này",
        "dominant_trends": [
            {
                "topic": "Hạ tầng AI & Backend NestJS",
                "status": "Trial",
                "summary": "Tổng hợp cập nhật từ các nguồn tin công nghệ (Failover mode).",
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
