from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import AuditEvent


def log_event(
    db: Session,
    action: str,
    user_id: Optional[str] = None,
    input_reference: Optional[dict] = None,
    output_reference: Optional[dict] = None,
    model_version: Optional[str] = None,
    document_versions: Optional[dict] = None,
) -> AuditEvent:
    event = AuditEvent(
        user_id=user_id,
        action=action,
        input_reference=input_reference,
        output_reference=output_reference,
        model_version=model_version,
        document_versions=document_versions,
    )

    db.add(event)
    db.commit()
    db.refresh(event)

    return event
