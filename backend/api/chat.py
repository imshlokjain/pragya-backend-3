from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.agents.orchestrator import handle_chat
from backend.core.auth import require_roles


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
)


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
            "VIEWER",
        )
    ),
):
    zone_id = request.context.zone_id if request.context else None
    district_id = request.context.district_id if request.context else None

    result = handle_chat(
        db,
        request.message,
        zone_id=zone_id,
        district_id=district_id,
    )

    return result
