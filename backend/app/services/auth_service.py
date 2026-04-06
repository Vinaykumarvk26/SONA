import hashlib
import secrets
from datetime import datetime, timedelta

import requests
from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.auth_session import AuthSession
from app.models.password_reset_token import PasswordResetToken
from app.models.user import User
from app.services.mongo_app_service import mongo_app


SESSION_TTL_HOURS = 72
PASSWORD_RESET_TTL_MINUTES = 20
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _normalize_identifier(identifier: str) -> str:
    return (identifier or "").strip().lower()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, _digest_hex = encoded.split(":")
    except ValueError:
        return False
    test = _hash_password(password, bytes.fromhex(salt_hex))
    return secrets.compare_digest(test, encoded)


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _base_username_from_email(email: str) -> str:
    local = (email.split("@", 1)[0] if "@" in email else email).strip().lower()
    cleaned = "".join(ch for ch in local if ch.isalnum() or ch in ("_", "."))
    return cleaned[:48] or "sonauser"


def _unique_username(db: Session, base: str) -> str:
    candidate = base[:64]
    if db.query(User).filter(User.username == candidate).one_or_none() is None:
        return candidate

    for _ in range(20):
        suffix = secrets.token_hex(2)
        with_suffix = f"{base[:59]}_{suffix}"
        if db.query(User).filter(User.username == with_suffix).one_or_none() is None:
            return with_suffix
    return f"user_{secrets.token_hex(4)}"


def signup(db: Session, username: str, email: str, password: str) -> User:
    email = _normalize_email(email)
    username = _normalize_username(username)

    if db.query(User).filter(User.email == email).one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Email is already registered")
    if db.query(User).filter(User.username == username).one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Username is already taken")

    encoded = _hash_password(password)
    user = User(username=username, email=email, hashed_password=encoded, password_hash=encoded)
    db.add(user)
    db.commit()
    db.refresh(user)
    try:
        mongo_app.upsert_user(user=user, provider="password")
        mongo_app.log_audit(action="user_signup", user=user, metadata={"provider": "password"})
    except Exception:
        pass
    return user


def signin(db: Session, identifier: str, password: str) -> User:
    normalized = _normalize_identifier(identifier)
    user = (
        db.query(User)
        .filter(or_(User.email == normalized, User.username == normalized))
        .one_or_none()
    )

    if user is None:
        raise HTTPException(status_code=404, detail="User not registered. Please signup")
    if not _verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect password")
    try:
        mongo_app.upsert_user(user=user, provider="password")
        mongo_app.log_audit(action="user_signin", user=user, metadata={"provider": "password"})
    except Exception:
        pass
    return user


def signin_google(db: Session, access_token: str) -> User:
    token = (access_token or "").strip()
    if not token:
        raise HTTPException(status_code=422, detail="Google access token is required")

    try:
        response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Google verification service unavailable") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    payload = response.json() or {}
    email = _normalize_email(payload.get("email", ""))
    email_verified = bool(payload.get("email_verified"))
    if not email or not email_verified:
        raise HTTPException(status_code=401, detail="Google account email is not verified")

    user = db.query(User).filter(User.email == email).one_or_none()
    if user is not None:
        try:
            mongo_app.upsert_user(user=user, provider="google")
            mongo_app.log_audit(action="user_signin", user=user, metadata={"provider": "google"})
        except Exception:
            pass
        return user

    username_base = _base_username_from_email(email)
    username = _unique_username(db, username_base)
    encoded = _hash_password(secrets.token_urlsafe(24))
    user = User(username=username, email=email, hashed_password=encoded, password_hash=encoded)
    db.add(user)
    db.commit()
    db.refresh(user)
    try:
        mongo_app.upsert_user(user=user, provider="google")
        mongo_app.log_audit(action="user_signup", user=user, metadata={"provider": "google"})
    except Exception:
        pass
    return user


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)
    session = AuthSession(user_id=user.id, token=token, expires_at=expires_at)
    db.add(session)
    db.commit()
    try:
        mongo_app.create_session(user=user, token=token, expires_at=expires_at)
    except Exception:
        pass
    return token


def validate_token(db: Session, token: str) -> User:
    session = (
        db.query(AuthSession)
        .filter(AuthSession.token == token, AuthSession.expires_at > datetime.utcnow())
        .one_or_none()
    )
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == session.user_id).one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    try:
        mongo_app.touch_session(token)
    except Exception:
        pass
    return user


def create_password_reset_token(db: Session, identifier: str) -> str | None:
    normalized = _normalize_identifier(identifier)
    user = (
        db.query(User)
        .filter(or_(User.email == normalized, User.username == normalized))
        .one_or_none()
    )
    if user is None:
        return None

    token = secrets.token_urlsafe(40)
    token_hash = _hash_reset_token(token)

    # Invalidate older active tokens for this user.
    active_tokens = (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > datetime.utcnow(),
        )
        .all()
    )
    for row in active_tokens:
        row.used_at = datetime.utcnow()

    row = PasswordResetToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_TTL_MINUTES),
    )
    db.add(row)
    db.commit()
    return token


def reset_password(db: Session, reset_token: str, new_password: str) -> None:
    token_hash = _hash_reset_token(reset_token)
    row = (
        db.query(PasswordResetToken)
        .filter(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > datetime.utcnow(),
        )
        .one_or_none()
    )

    if row is None:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user = db.query(User).filter(User.id == row.user_id).one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    encoded = _hash_password(new_password)
    user.hashed_password = encoded
    user.password_hash = encoded
    row.used_at = datetime.utcnow()

    # Force re-login on all devices after password reset.
    db.query(AuthSession).filter(AuthSession.user_id == user.id).delete(synchronize_session=False)
    db.commit()
    try:
        mongo_app.revoke_user_sessions(user.id)
        mongo_app.log_audit(action="password_reset", user=user, metadata={})
    except Exception:
        pass


def update_profile(
    db: Session,
    user: User,
    *,
    username: str | None = None,
    full_name: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    bio: str | None = None,
) -> tuple[User, dict]:
    if username is not None:
        next_username = _normalize_username(username)
        if not next_username:
            raise HTTPException(status_code=422, detail="Username cannot be empty")
        if next_username != user.username:
            exists = db.query(User).filter(User.username == next_username, User.id != user.id).one_or_none()
            if exists is not None:
                raise HTTPException(status_code=409, detail="Username is already taken")
            user.username = next_username

    db.add(user)
    db.commit()
    db.refresh(user)

    profile = {
        "full_name": (full_name or "").strip() if full_name is not None else "",
        "phone": (phone or "").strip() if phone is not None else "",
        "location": (location or "").strip() if location is not None else "",
        "bio": (bio or "").strip() if bio is not None else "",
    }
    try:
        mongo_app.upsert_user(user=user, provider="password")
        profile = mongo_app.save_user_profile(
            user=user,
            full_name=full_name,
            phone=phone,
            location=location,
            bio=bio,
        )
        mongo_app.log_audit(action="profile_updated", user=user, metadata=profile)
    except Exception:
        pass

    return user, profile
