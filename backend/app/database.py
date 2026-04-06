from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

# Naming convention keeps Alembic diffs deterministic across environments.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)
Base = declarative_base(metadata=metadata)

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db():
    """FastAPI dependency that yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Example bootstrap for local/dev usage; Alembic should own prod migrations."""
    import app.models  # noqa: F401

    from sqlalchemy.exc import OperationalError

    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        # SQLite can sometimes report existing indexes on redeploy; ignore those.
        if "already exists" not in str(exc):
            raise
    _apply_sqlite_compat_fixes()


def _apply_sqlite_compat_fixes() -> None:
    """Backfill essential columns for legacy local SQLite databases."""
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as conn:
        tables = {row[0] for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
        if "users" not in tables:
            return

        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()}

        if "hashed_password" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN hashed_password VARCHAR")
            columns.add("hashed_password")

        if "updated_at" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN updated_at DATETIME")
            columns.add("updated_at")

        if "username" not in columns:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN username VARCHAR(64)")
            columns.add("username")

        if "password_hash" in columns:
            conn.exec_driver_sql(
                "UPDATE users SET hashed_password = password_hash "
                "WHERE (hashed_password IS NULL OR TRIM(hashed_password) = '') AND password_hash IS NOT NULL"
            )
            conn.exec_driver_sql(
                "UPDATE users SET password_hash = hashed_password "
                "WHERE (password_hash IS NULL OR TRIM(password_hash) = '') AND hashed_password IS NOT NULL"
            )

        conn.exec_driver_sql(
            "UPDATE users SET hashed_password = hex(randomblob(32)) "
            "WHERE hashed_password IS NULL OR TRIM(hashed_password) = ''"
        )
        conn.exec_driver_sql(
            "UPDATE users SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP) "
            "WHERE updated_at IS NULL"
        )
        conn.exec_driver_sql(
            "UPDATE users SET username = lower(substr(email, 1, instr(email, '@') - 1)) || '_' || id "
            "WHERE username IS NULL OR TRIM(username) = ''"
        )
