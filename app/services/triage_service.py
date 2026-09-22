import json
from typing import Optional, Tuple

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.schemas import FastTriageResult
from app.services.ai_analyzer import _clean_json_response


async def _triage_with_typesafe_jev(title: str, snippet: str, url: str) -> Optional[FastTriageResult]:
    """
    Calls TypeSafe AI Jev System One endpoint using official typed primitives:
    - noul: binary validation for is_relevant_tech & is_spam
    - choice: categorical routing for suggested_priority
    - score: quantitative score on article depth
    Endpoint: POST https://api.typesafe.ai/v1/systemone
    """
    if not settings.TYPESAFE_API_KEY:
        return None

    endpoint = f"{settings.TYPESAFE_BASE_URL.rstrip('/')}/systemone"
    headers = {
        "Authorization": f"Bearer {settings.TYPESAFE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Official TypeSafe System One schema format with structured state
    payload = {
        "model": "jev-latest",
        "state": {
            "title": title,
            "url": url,
            "snippet": snippet[:600],
            "source_domain": url.split("/")[2] if "//" in url else ""
        },
        "questions": {
            "is_relevant_tech": {
                "type": "noul",
                "instructions": "Does this article (referencing `title` and `snippet`) provide meaningful engineering value for Software, Backend, AI, Cloud, or DevOps professionals?"
            },
            "is_spam_or_marketing": {
                "type": "noul",
                "instructions": "Is this article purely consumer gadget marketing, discount codes, affiliate spam, or superficial PR without tech depth?"
            },
            "is_breaking_news": {
                "type": "noul",
                "instructions": "Does this article represent an urgent breaking technical event, critical zero-day security vulnerability, or major flagship AI model release?"
            },
            "urgency_level": {
                "type": "choice",
                "instructions": "Determine the urgency level of this news event for software engineers.",
                "criteria": {
                    "CRITICAL": "Critical breaking news, security CVE-critical emergency, paradigm-shifting AI model release.",
                    "HIGH": "Significant software release, major framework update, high-priority technical incident postmortem.",
                    "NORMAL": "Standard informative engineering blog post, tutorial, or routine update."
                }
            },
            "suggested_priority": {
                "type": "choice",
                "instructions": "Determine how the crawler pipeline should route this technical article for software engineers.",
                "criteria": {
                    "PROCESS_FULL_AI": "Any technical article, software engineering release, architecture pattern, backend/AI framework update, system design, or engineering tutorial to be analyzed and synthesized.",
                    "STORE_UNANALYZED": "General software industry news or minor software announcements.",
                    "DISCARD": "Consumer electronics, shopping deals, coupon codes, affiliate marketing, pure PR noise, or non-technical content."
                }
            },
            "tech_depth_score": {
                "type": "score",
                "instructions": "How deep and practically useful is the engineering content in this article for senior backend and AI engineers?",
                "criteria": [
                    "Shallow clickbait, generic tech news, or non-technical product promotion",
                    "High-level software news announcement or basic conceptual overview",
                    "Moderate engineering overview with general architecture insights",
                    "High-value in-depth technical analysis, system design breakdown, or production code lessons",
                    "Seminal breakthrough, production postmortem, or comprehensive engineering guide"
                ]
            }
        }
    }

    async with httpx.AsyncClient(timeout=4.0) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            answers = data.get("answers", {})

            # Parse Typed Primitives
            # 1. Noul: returns float in "noul" field (not "probability")
            p_tech_raw = answers.get("is_relevant_tech", {})
            p_tech = float(p_tech_raw.get("noul", 1.0)) if isinstance(p_tech_raw, dict) else float(p_tech_raw or 1.0)
            p_spam_raw = answers.get("is_spam_or_marketing", {})
            p_spam = float(p_spam_raw.get("noul", 0.0)) if isinstance(p_spam_raw, dict) else float(p_spam_raw or 0.0)
            p_breaking_raw = answers.get("is_breaking_news", {})
            p_breaking = float(p_breaking_raw.get("noul", 0.0)) if isinstance(p_breaking_raw, dict) else float(p_breaking_raw or 0.0)

            # 2. Choice: returns selected option + calibrated confidence
            choice_obj = answers.get("suggested_priority", {})
            priority = choice_obj.get("choice", "PROCESS_FULL_AI") if isinstance(choice_obj, dict) else str(choice_obj)
            confidence = choice_obj.get("confidence", 0.90) if isinstance(choice_obj, dict) else 0.90

            urgency_obj = answers.get("urgency_level", {})
            urgency = urgency_obj.get("choice", "NORMAL") if isinstance(urgency_obj, dict) else str(urgency_obj)

            # 3. Score: returns float score (0-4 mapped to 1-10 legend)
            score_obj = answers.get("tech_depth_score", {})
            depth_score = score_obj.get("score", 2.0) if isinstance(score_obj, dict) else float(score_obj or 2.0)

            return FastTriageResult(
                is_relevant_tech=(p_tech >= 0.50),
                is_spam_or_marketing=(p_spam >= 0.50),
                confidence=float(confidence),
                suggested_priority=priority,
                is_breaking_news=(p_breaking >= 0.65 or urgency == "CRITICAL"),
                urgency_level=urgency,
                tech_depth_score=float(depth_score),
                reason=f"Jev System-1 Primitive (TechProb: {p_tech:.2f}, SpamProb: {p_spam:.2f}, Urgency: {urgency})"
            )
        else:
            print(f"[TypeSafe Jev Notice] Status {resp.status_code}: {resp.text}")
            return None


async def _triage_with_fast_llm_fallback(title: str, snippet: str, url: str) -> FastTriageResult:
    """Fallback classifier using existing 9routers fast pass with structured schema"""
    client = AsyncOpenAI(
        base_url=settings.NINEROUTERS_BASE_URL,
        api_key=settings.NINEROUTERS_API_KEY,
    )

    triage_prompt = f"""You are a high-speed technical article classifier (System One).
Classify the following article and return STRICT JSON with schema:
{{
  "is_relevant_tech": boolean,
  "is_spam_or_marketing": boolean,
  "is_breaking_news": boolean,
  "urgency_level": "CRITICAL" | "HIGH" | "NORMAL",
  "tech_depth_score": float (1.0 to 10.0),
  "confidence": float (0.0 to 1.0),
  "suggested_priority": "DISCARD" | "STORE_UNANALYZED" | "PROCESS_FULL_AI",
  "reason": "short string"
}}

Article Title: {title}
URL: {url}
Snippet: {snippet[:400]}

Rules:
- High-value technical content (Backend, AI/ML, Cloud, Databases, Distributed Systems) -> "PROCESS_FULL_AI"
- Consumer tech reviews, phone specs, non-tech news, pure PR -> "DISCARD"
- Urgent 0-day security flaw or flagship LLM breakthrough -> is_breaking_news: true, urgency_level: "CRITICAL"
- Generic software news with moderate relevance -> "STORE_UNANALYZED"
"""

    for model_name in settings.fallback_models_list:
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a fast JSON classification engine. Return ONLY valid JSON."},
                    {"role": "user", "content": triage_prompt}
                ],
                temperature=0.1,
                max_tokens=180,
                timeout=12.0
            )
            if completion.choices and completion.choices[0].message:
                raw_text = completion.choices[0].message.content or "{}"
                parsed = _clean_json_response(raw_text)
                return FastTriageResult(
                    is_relevant_tech=bool(parsed.get("is_relevant_tech", True)),
                    is_spam_or_marketing=bool(parsed.get("is_spam_or_marketing", False)),
                    confidence=float(parsed.get("confidence", 0.85)),
                    suggested_priority=str(parsed.get("suggested_priority", "PROCESS_FULL_AI")),
                    is_breaking_news=bool(parsed.get("is_breaking_news", False)),
                    urgency_level=str(parsed.get("urgency_level", "NORMAL")),
                    tech_depth_score=float(parsed.get("tech_depth_score", 5.0)),
                    reason=parsed.get("reason", f"{model_name} fast classifier")
                )
        except Exception as err:
            print(f"[Fast Triage Fallback] {model_name} failed: {err}")
            continue

    # Final safe fallback if all providers fail
    return FastTriageResult(
        is_relevant_tech=True,
        is_spam_or_marketing=False,
        confidence=0.0,
        suggested_priority="PROCESS_FULL_AI",
        is_breaking_news=False,
        urgency_level="NORMAL",
        tech_depth_score=5.0,
        reason="Fallback due to all triage models failing"
    )


async def fast_triage_article(title: str, snippet: str, url: str) -> Tuple[FastTriageResult, str]:
    """
    Tier-1 System One Triage.
    Priority 1: TypeSafe AI Jev (https://api.typesafe.ai/v1/systemone)
    Priority 2: Fast LLM Structured Fallback (9routers)
    """
    # 1. Try Jev System One
    try:
        jev_result = await _triage_with_typesafe_jev(title=title, snippet=snippet, url=url)
        if jev_result:
            return jev_result, "typesafe-jev"
    except Exception as jev_err:
        print(f"[TypeSafe Jev Warning] {jev_err}, falling back to fast LLM")

    # 2. Fallback to Fast LLM
    llm_result = await _triage_with_fast_llm_fallback(title=title, snippet=snippet, url=url)
    return llm_result, "fast-llm-classifier"
