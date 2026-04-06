from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from app.config import get_settings


@lru_cache
def get_mongo_client() -> MongoClient | None:
    settings = get_settings()
    mongodb_url = (settings.mongodb_url or "").strip()
    if not mongodb_url:
        return None

    return MongoClient(
        mongodb_url,
        serverSelectionTimeoutMS=4000,
        connectTimeoutMS=4000,
        socketTimeoutMS=4000,
        retryWrites=True,
    )


def get_mongo_db() -> Database | None:
    client = get_mongo_client()
    if client is None:
        return None
    return client[get_settings().mongodb_db_name]
