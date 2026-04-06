import re
import secrets

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.user import User


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", (value or "guest").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "guest"


def resolve_user(db: Session, identifier: str) -> User | None:
    if not identifier:
        return None

    text = identifier.strip()
    if not text:
        return None

    query = db.query(User)
    if text.isdigit():
        user = query.filter(User.id == int(text)).one_or_none()
        if user is not None:
            return user

    lowered = text.lower()
    return query.filter(or_(User.username == lowered, User.email == lowered)).one_or_none()


def get_or_create_user(db: Session, identifier: str) -> User:
    existing = resolve_user(db, identifier)
    if existing is not None:
        return existing

    base = _slug(identifier)
    username = base
    email = f"{base}@guest.local"

    while db.query(User).filter(or_(User.username == username, User.email == email)).first() is not None:
        suffix = secrets.token_hex(2)
        username = f"{base}_{suffix}"
        email = f"{base}_{suffix}@guest.local"

    placeholder = secrets.token_hex(32)
    user = User(username=username, email=email, hashed_password=placeholder, password_hash=placeholder)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
