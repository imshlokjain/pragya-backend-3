from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.auth import require_roles
from backend.database.session import get_db
from backend.schemas.risk import ScenarioRequest, ScenarioResult
from backend.services import scenario_service


router = APIRouter(
    prefix="/api/v1/scenario",
    tags=["scenario"],
)


@router.post(
    "",
    response_model=ScenarioResult,
)
def run_scenario(
    request: ScenarioRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
        )
    ),
):
    result = scenario_service.run_scenario(
        db=db,
        zone_id=request.zone_id,
        rainfall_multiplier=request.rainfall_multiplier,
        river_level_increase=request.river_level_increase,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{request.zone_id}' not found",
        )

    return result
