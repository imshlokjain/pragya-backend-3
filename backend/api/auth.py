from fastapi import APIRouter, HTTPException

from backend.core.security import (
    create_access_token,
    verify_password,
)


router = APIRouter(
    prefix="/api/v1/auth",
    tags=["auth"],
)


# Temporary MVP users.
# Passwords are stored as bcrypt hashes.
# Replace this with a users table for production.
USERS = {
    "admin": {
        "password_hash": "$2b$12$QGayEa/.K4f9YYcX4oI4xuqIMwIh2UDC7LnuxWfKuZabj06BaBQWe",
        "role": "ADMIN",
    },
    "operator": {
        "password_hash": "$2b$12$ePZAye3cXMQC6owAm1g6aO3sqmsRa5wEzgeHAlMLeFW7P43OMl/TO",
        "role": "OPERATOR",
    },
    "analyst": {
        "password_hash": "$2b$12$uagfkO/vHZP45vUbSzq0d.4NzUhC58tjhhyA9po8h.bSNzvfSMOMm",
        "role": "ANALYST",
    },
    "viewer": {
        "password_hash": "$2b$12$tzNrqRx94PlCo54L/BUfE.hnqpP69KcYED1QZTzKtEI0YC0032HQW",
        "role": "VIEWER",
    },
}


@router.post("/login")
def login(username: str, password: str):
    user = USERS.get(username)

    if user is None or not verify_password(
        password,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    token = create_access_token(
        user_id=username,
        role=user["role"],
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": username,
        "role": user["role"],
    }
