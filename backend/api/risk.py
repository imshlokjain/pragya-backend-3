from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.core.auth import require_roles
from backend.database.session import get_db
from backend.schemas.risk import RiskPredictionOut
from backend.services import risk_service


router = APIRouter(
    prefix="/api/v1/risk",
    tags=["risk"],
)


@router.get(
    "/district/{district_id}",
    response_model=list[RiskPredictionOut],
)
def get_district_risk(
    district_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
            "VIEWER",
        )
    ),
):
    return risk_service.get_district_risk(
        db,
        district_id,
    )


@router.get(
    "/{zone_id}/history",
    response_model=list[RiskPredictionOut],
)
def get_zone_risk_history(
    zone_id: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
    current_user=Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
            "VIEWER",
        )
    ),
):
    result = risk_service.get_risk_history(
        db,
        zone_id,
        limit,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown zone_id: {zone_id}",
        )

    return result


@router.get(
    "/{zone_id}",
    response_model=RiskPredictionOut,
)
def get_zone_risk(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
            "VIEWER",
        )
    ),
):
    result = risk_service.get_risk(
        db,
        zone_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown zone_id: {zone_id}",
        )

    return result


@router.post(
    "/{zone_id}/predict",
    response_model=RiskPredictionOut,
)
def create_risk_prediction(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
        )
    ),
):
    result = risk_service.save_risk_prediction(
        db,
        zone_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown zone_id: {zone_id}",
        )

    return result
