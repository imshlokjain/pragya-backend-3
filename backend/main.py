from datetime import datetime

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.api import (
    risk,
    rainfall,
    river,
    scenario,
    chat,
    auth,
    satellite,
    flood,
    features,
)
from backend.database.models import AuditEvent
from backend.database.session import SessionLocal, get_db
from backend.core.security import decode_access_token
from backend.core.config import settings
from backend.services import zone_service


app = FastAPI(
    title="PRAGYA Backend",
    description=(
        "Predictive Risk Assessment & Geo-Intelligence "
        "for Adaptive Governance and Action"
    ),
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routers
# ---------------------------------------------------------

app.include_router(auth.router)
app.include_router(risk.router)
app.include_router(rainfall.router)
app.include_router(river.router)
app.include_router(scenario.router)
app.include_router(chat.router)
app.include_router(satellite.router)
app.include_router(flood.router)
app.include_router(features.router)


# ---------------------------------------------------------
# Audit Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def audit_middleware(request, call_next):
    start_time = datetime.utcnow()

    response = await call_next(request)

    if request.url.path not in ["/", "/health"]:
        db = SessionLocal()

        try:
            user_id = None

            authorization = request.headers.get("Authorization")

            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ", 1)[1]

                try:
                    payload = decode_access_token(token)
                    user_id = payload.get("sub")
                except ValueError:
                    user_id = None

            event = AuditEvent(
                user_id=user_id,
                timestamp=start_time,
                action=f"{request.method} {request.url.path}",
                input_reference={
                    "path": request.url.path,
                    "method": request.method,
                },
                output_reference={
                    "status_code": response.status_code,
                },
            )

            db.add(event)
            db.commit()

        except Exception:
            db.rollback()

        finally:
            db.close()

    return response


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "PRAGYA Backend is running",
        "version": "0.1.0",
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ---------------------------------------------------------
# Zones
# ---------------------------------------------------------

@app.get("/api/v1/zones")
def get_zones(db: Session = Depends(get_db)):
    return zone_service.list_zones(db)
