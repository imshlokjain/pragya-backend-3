from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class SopProvider(ABC):

    @abstractmethod
    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


# --------------------------------------------------
# PROTOTYPE FLOOD SOP KNOWLEDGE BASE
# --------------------------------------------------

SOP_CLAUSES = [

    {
        "id": "SOP-001",
        "title": "Critical Flood Risk Response",
        "keywords": [
            "critical",
            "critical risk",
            "flood risk",
            "emergency",
            "danger",
        ],
        "content": (
            "When flood risk is classified as CRITICAL, "
            "immediately activate emergency response procedures. "
            "Notify district authorities and disaster management "
            "teams, prepare evacuation resources, issue public "
            "warnings, and continuously monitor river levels."
        ),
        "source": "Prototype Flood Emergency SOP",
    },

    {
        "id": "SOP-002",
        "title": "High River Level Monitoring",
        "keywords": [
            "river",
            "river level",
            "rising",
            "water level",
            "danger level",
        ],
        "content": (
            "When river levels are rising or approaching danger "
            "levels, increase monitoring frequency and notify "
            "relevant authorities. Prepare emergency response "
            "teams and review evacuation readiness."
        ),
        "source": "Prototype Flood Monitoring SOP",
    },

    {
        "id": "SOP-003",
        "title": "Heavy Rainfall Response",
        "keywords": [
            "rain",
            "rainfall",
            "heavy rain",
            "monsoon",
        ],
        "content": (
            "During heavy rainfall, monitor rainfall accumulation "
            "and river levels closely. Inspect vulnerable drainage "
            "areas and prepare response teams for possible flooding."
        ),
        "source": "Prototype Rainfall Response SOP",
    },

    {
        "id": "SOP-004",
        "title": "Evacuation Preparation",
        "keywords": [
            "evacuate",
            "evacuation",
            "leave",
            "safe place",
            "shelter",
        ],
        "content": (
            "Prepare evacuation routes and shelters for vulnerable "
            "communities. Prioritize elderly people, children, "
            "persons with disabilities, and residents in low-lying "
            "areas."
        ),
        "source": "Prototype Evacuation SOP",
    },

    {
        "id": "SOP-005",
        "title": "Public Warning Procedure",
        "keywords": [
            "warning",
            "alert",
            "public",
            "notify",
            "announcement",
        ],
        "content": (
            "Issue clear and timely public warnings through "
            "available communication channels. Provide information "
            "about flood risk, affected areas, evacuation routes, "
            "and emergency contacts."
        ),
        "source": "Prototype Public Warning SOP",
    },
]


class PrototypeSopProvider(SopProvider):

    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        query_lower = query.lower()

        matches = []

        for clause in SOP_CLAUSES:

            for keyword in clause["keywords"]:

                if keyword in query_lower:
                    matches.append(
                        {
                            "id": clause["id"],
                            "title": clause["title"],
                            "content": clause["content"],
                        }
                    )

                    break

        # Remove duplicates
        unique_matches = []

        seen_ids = set()

        for match in matches:

            if match["id"] not in seen_ids:

                unique_matches.append(match)

                seen_ids.add(match["id"])

        sources = list(
            {
                clause["source"]
                for clause in SOP_CLAUSES
                if any(
                    match["id"] == clause["id"]
                    for match in unique_matches
                )
            }
        )

        return {
            "query": query,
            "context": context or {},
            "status": (
                "FOUND"
                if unique_matches
                else "NO_MATCH"
            ),
            "clauses": unique_matches,
            "sources": sources,
            "note": (
                "Prototype keyword-based SOP retrieval. "
                "Human review is required."
            ),
        }


sop_provider = PrototypeSopProvider()
