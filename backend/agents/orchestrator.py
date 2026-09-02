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

        drivers_text = "; ".join(
            f"{d['feature']} ({d['direction'].replace('_', ' ').lower()}, "
            f"contributes {d['contribution']})"
            for d in risk["drivers"]
        )
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
        answer = (
            "SOP retrieval isn't wired up yet (Phase 4), so I can't cite a specific clause. "
            "Once it is, this will return the matched SOP section/page and a draft response "
            "plan labelled 'DRAFT — HUMAN REVIEW REQUIRED'."
        )
        evidence = [sop] + ([risk] if risk else [])
        return _build_response(answer, tool_calls, evidence=evidence)

    return _build_response(
        "I can answer questions about zone risk, rainfall, river levels, and what-if "
        "rainfall scenarios. SOP-based recommendations are coming in a later phase. Try "
        "asking why a specific zone is at risk, or what happens if rainfall increases by 30%.",
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


def _error_response(error, tool_calls):
    return _build_response(error, tool_calls, evidence=[])


def _build_response(answer, tool_calls, evidence):
    return {"answer": answer, "tool_calls": tool_calls, "evidence": evidence}