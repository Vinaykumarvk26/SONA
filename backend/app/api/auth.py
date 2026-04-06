from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.schemas import (
    AuthGoogleRequest,
    AuthResponse,
    AuthSigninRequest,
    AuthSignupRequest,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    UserProfileResponse,
    UserProfileUpdateRequest,
    ResetPasswordRequest,
)
from app.services.auth_service import (
    create_password_reset_token,
    create_session,
    reset_password,
    signin_google,
    signin,
    signup,
    update_profile,
    validate_token,
)
from app.services.mongo_app_service import mongo_app

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return parts[1]


@router.post("/signup", response_model=AuthResponse)
def auth_signup(payload: AuthSignupRequest, db: Session = Depends(get_db)):
    user = signup(db, payload.username, payload.email, payload.password)
    token = create_session(db, user)
    return AuthResponse(token=token, email=user.email, username=user.username)


@router.post("/signin", response_model=AuthResponse)
def auth_signin(payload: AuthSigninRequest, db: Session = Depends(get_db)):
    identifier = (payload.identifier or payload.email or "").strip()
    if not identifier:
        raise HTTPException(status_code=422, detail="identifier or email is required")

    user = signin(db, identifier, payload.password)
    token = create_session(db, user)
    return AuthResponse(token=token, email=user.email, username=user.username)


@router.post("/google", response_model=AuthResponse)
def auth_google(payload: AuthGoogleRequest, db: Session = Depends(get_db)):
    user = signin_google(db, payload.access_token)
    token = create_session(db, user)
    return AuthResponse(token=token, email=user.email, username=user.username)


@router.get("/me", response_model=UserProfileResponse)
def auth_me(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    token = _extract_bearer(authorization)
    user = validate_token(db, token)
    profile = mongo_app.get_user_profile(user=user)
    return UserProfileResponse(
        email=user.email,
        username=user.username,
        full_name=profile.get("full_name", ""),
        phone=profile.get("phone", ""),
        location=profile.get("location", ""),
        bio=profile.get("bio", ""),
    )


@router.patch("/me", response_model=UserProfileResponse)
def auth_update_me(
    payload: UserProfileUpdateRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    token = _extract_bearer(authorization)
    user = validate_token(db, token)
    user, profile = update_profile(
        db,
        user,
        username=payload.username,
        full_name=payload.full_name,
        phone=payload.phone,
        location=payload.location,
        bio=payload.bio,
    )
    return UserProfileResponse(
        email=user.email,
        username=user.username,
        full_name=profile.get("full_name", ""),
        phone=profile.get("phone", ""),
        location=profile.get("location", ""),
        bio=profile.get("bio", ""),
    )


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def auth_forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    token = create_password_reset_token(db, payload.identifier)

    # Keep response generic in production to avoid account enumeration.
    if settings.expose_reset_token_in_response and token is not None:
        return ForgotPasswordResponse(
            message="Password reset token generated. Use it to reset your password.",
            reset_token=token,
        )
    return ForgotPasswordResponse(message="If this account exists, password reset instructions were generated.")


@router.post("/reset-password")
def auth_reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    reset_password(db, payload.reset_token, payload.new_password)
    return {"ok": True, "message": "Password reset successful. Please log in with your new password."}
