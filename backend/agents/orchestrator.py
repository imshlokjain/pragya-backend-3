"""
Minimal agent orchestrator (TRD sections 26-28).

This is a keyword-based intent router standing in for the real LLM
tool-calling orchestrator. It exists so the API contract for /api/v1/chat
is real and testable before an LLM is wired in.

The guardrail that matters most is already enforced structurally: every
number in a response comes from a call into backend/agents/tools.py, never
typed inline in this file. When this gets replaced by a real LLM
orchestrator, that invariant should carry over unchanged -- the LLM
composes prose *around* tool outputs, it does not generate the numbers
itself (TRD section 28).
"""

import re
from typing import Optional

from sqlalchemy.orm import Session

from backend.agents import tools

DEFAULT_SCENARIO_MULTIPLIER = 1.3  # matches the MVP demo script's "+30%"


def classify_intent(message: str) -> str:
    m = message.lower()
    if "what if" in m or ("increase" in m and "%" in m):
        return "scenario"
    if "which zone" in m or "highest risk" in m or "most at risk" in m:
        return "rank_zones"
    if "what should" in m or "action" in m or "recommend" in m or "sop" in m:
        return "actions"
    if "why" in m or "risk" in m:
        return "explain_risk"
    return "unknown"


def extract_rainfall_multiplier(message: str) -> float:
    match = re.search(r"(\d+(\.\d+)?)\s*%", message)
    if match:
        pct = float(match.group(1))
        return round(1 + pct / 100, 2)
    return DEFAULT_SCENARIO_MULTIPLIER


def handle_chat(
    db: Session, message: str, zone_id: Optional[str] = None, district_id: Optional[str] = None
) -> dict:
    intent = classify_intent(message)
    tool_calls = []

    def call(tool_name, fn, *args, **kwargs):
        output = fn(*args, **kwargs)
        tool_calls.append({"tool": tool_name, "output": output})
        return output

    if intent == "scenario":
        if not zone_id:
            return _needs_zone_response(tool_calls)
        multiplier = extract_rainfall_multiplier(message)
        scenario = call("run_scenario", tools.run_scenario, db, zone_id, multiplier)
        if "error" in scenario:
            return _error_response(scenario["error"], tool_calls)
        pct = round((multiplier - 1) * 100)
        answer = (
            f"Running a +{pct}% rainfall scenario: baseline risk is "
            f"{scenario['baseline_risk']} ({scenario['baseline_category']}), scenario risk is "
            f"{scenario['scenario_risk']} ({scenario['scenario_category']}), a change of "
            f"{scenario['risk_change']} points. This is a HYPOTHETICAL result, not an "
            f"observed condition."
        )
        return _build_response(answer, tool_calls, evidence=[scenario])

    if intent == "rank_zones":
        return _build_response(
            "Ranking zones by risk needs a district-wide call. Hook this up to "
            "GET /api/v1/risk/district/{district_id} once chat context carries a "
            "district_id — the endpoint already exists, this orchestrator just "
            "doesn't call it yet.",
            tool_calls,
            evidence=[],
        )

    if intent == "explain_risk":
        if not zone_id:
            return _needs_zone_response(tool_calls)
        risk = call("get_current_risk", tools.get_current_risk, db, zone_id)
        if "error" in risk:
            return _error_response(risk["error"], tool_calls)
        rainfall = call("get_rainfall", tools.get_rainfall, db, zone_id)
        river = call("get_river_level", tools.get_river_level, db, zone_id)
        satellite = call("get_satellite_evidence", tools.get_satellite_evidence, db, zone_id)

        drivers_text = "; ".join(_format_driver(driver) for driver in risk["drivers"])
        prototype_note = "prototype" if risk["is_prototype"] else "validated"
        answer = (
            f"This zone is currently classified {risk['risk_category']} "
            f"({risk['risk_score']}/100, confidence {risk['confidence']}). "
            f"Main drivers: {drivers_text}. "
            f"Prediction is labelled '{prototype_note}' (model version {risk['model_version']})."
        )
        return _build_response(answer, tool_calls, evidence=[risk, rainfall, river, satellite])

    if intent == "actions":
        sop = call("search_sop", tools.search_sop, message, {"zone_id": zone_id, "district_id": district_id})
        risk = call("get_current_risk", tools.get_current_risk, db, zone_id) if zone_id else None

        clauses = sop.get("clauses", [])
        sources = sop.get("sources", [])

        if clauses:
            action_lines = []
            for c in clauses:
                action_lines.append(f"• **{c.get('title', 'Action Protocol')}**: {c.get('content')}")

            risk_header = ""
            if risk and not "error" in risk:
                risk_header = f" for {zone_id} (Risk: **{risk.get('risk_category', 'ASSESSED')}**, Score: {risk.get('risk_score', 0)}/100)"

            citations_text = ", ".join(sources) if sources else "NDMA SOP Framework"

            answer = (
                f"🚨 **INCIDENT ACTION PLAN**{risk_header}\n\n"
                f"**Mandated Directives**:\n"
                + "\n".join(action_lines) +
                f"\n\n**SOP Citations**: {citations_text}\n\n"
                f"⚖️ *DRAFT ACTION PLAN — HUMAN REVIEW & DISTRICT MAGISTRATE SIGN-OFF REQUIRED BEFORE DEPLOYMENT.*"
            )
        else:
            answer = (
                "No specific SOP section directly matched your query. Recommended standard protocol: "
                "continuous river monitoring, drainage inspections, and alerting frontline NDRF rescue teams.\n\n"
                "⚖️ *HUMAN REVIEW REQUIRED.*"
            )

        evidence = [sop] + ([risk] if risk else [])
        return _build_response(answer, tool_calls, evidence=evidence)

    # General / Policy inquiry fallback — query RAG documents
    rag_docs = call("search_rag_documents", tools.search_rag_documents, message)
    if rag_docs.get("status") == "FOUND":
        chunks = rag_docs.get("chunks", [])
        sources = rag_docs.get("sources", [])
        context_text = "\n\n".join(f"[{c.get('document')} p.{c.get('page')}]: {c.get('content')[:300]}..." for c in chunks)
        citations_text = ", ".join(sources)
        answer = (
            f"Grounding information from disaster management documents:\n\n"
            f"{context_text}\n\n"
            f"**Citations**: {citations_text}"
        )
        return _build_response(answer, tool_calls, evidence=[rag_docs])

    return _build_response(
        "I can answer questions about zone risk, rainfall, river levels, what-if "
        "rainfall scenarios, and NDMA SOP action plans. Try "
        "asking 'What action should we take?', 'Why is this zone at risk?', or 'What if rainfall increases by 30%?'.",
        tool_calls,
        evidence=[],
    )



def _needs_zone_response(tool_calls):
    return _build_response(
        "I need a zone_id in the chat context to answer this — pass context.zone_id "
        "with your request.",
        tool_calls,
        evidence=[],
    )


def _format_driver(driver) -> str:
    """Keep chat compatible with both the current string drivers and future SHAP-style objects."""
    if isinstance(driver, dict):
        feature = driver.get("feature", "risk signal")
        direction = str(driver.get("direction", "observed")).replace("_", " ").lower()
        contribution = driver.get("contribution")
        return f"{feature} ({direction}{f', contributes {contribution}' if contribution is not None else ''})"
    return str(driver)


def _error_response(error, tool_calls):
    return _build_response(error, tool_calls, evidence=[])


def _build_response(answer, tool_calls, evidence):
    return {"answer": answer, "tool_calls": tool_calls, "evidence": evidence}
