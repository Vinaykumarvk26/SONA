import base64
import hashlib
import logging
import random
import time
from typing import Dict
from urllib.parse import quote_plus

import requests

from app.config import get_settings


class SpotifyService:
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    RECOMMENDATION_URL = "https://api.spotify.com/v1/recommendations"
    TRACK_SEARCH_URL = "https://api.spotify.com/v1/search"
    ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

    EMOTION_GENRES = {
        "happy": ["pop,dance,indie-pop", "dance,pop,electropop"],
        # SONA leans toward emotional regulation for lower moods.
        "sad": ["pop,indie-pop,acoustic", "indie-pop,acoustic,folk"],
        "angry": ["acoustic,ambient,chill", "acoustic,study,indie"],
        "fear": ["ambient,chill,acoustic", "acoustic,study,indie"],
        "disgust": ["acoustic,chill,indie", "ambient,acoustic,study"],
        "surprise": ["dance,pop,electropop", "pop,dance,indie-pop"],
        "neutral": ["chill,pop,indie", "acoustic,pop,indie-pop"],
    }

    EMOTION_SEARCH_TERMS = {
        "happy": [
            "bollywood party hits",
            "punjabi dance songs",
            "feel good hindi songs",
            "tamil kuthu hits",
        ],
        "sad": [
            "motivational hindi songs",
            "healing bollywood songs",
            "feel good hindi songs",
            "uplifting indian songs",
        ],
        "angry": [
            "calm hindi songs",
            "peaceful bollywood songs",
            "soothing indian acoustic",
            "relaxing hindi songs",
        ],
        "fear": [
            "calm indian songs",
            "healing bollywood songs",
            "peaceful hindi songs",
            "instrumental indian calm",
        ],
        "disgust": [
            "soft hindi songs",
            "calm bollywood songs",
            "indian acoustic songs",
            "relaxing bollywood",
        ],
        "surprise": [
            "trending bollywood hits",
            "latest hindi songs",
            "indian pop hits",
            "trending tamil songs",
        ],
        "neutral": [
            "indian chill mix",
            "bollywood soft hits",
            "hindi indie songs",
            "telugu chill songs",
        ],
    }

    INDIAN_HINTS = (
        "hindi",
        "bollywood",
        "punjabi",
        "tamil",
        "telugu",
        "malayalam",
        "arijit",
        "shreya",
        "atif",
        "rahat",
        "pritam",
        "a.r. rahman",
        "anirudh",
        "jubin",
    )

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger("spotify")
        self._token = None

    def _get_token(self) -> str:
        if self._token:
            return self._token

        client_id = self.settings.spotify_client_id
        client_secret = self.settings.spotify_client_secret
        if (
            not client_id
            or not client_secret
            or client_id.startswith("your_")
            or client_secret.startswith("your_")
        ):
            self.logger.warning("Spotify credentials are not configured; using fallback recommendations")
            return ""

        auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        try:
            resp = requests.post(
                self.TOKEN_URL,
                headers={"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
                timeout=10,
            )
            resp.raise_for_status()
            self._token = resp.json().get("access_token", "")
            return self._token
        except requests.RequestException as exc:
            self.logger.warning("Spotify auth failed (%s); using fallback recommendations", exc)
            self._token = ""
            return ""

    def recommend_tracks(self, target_features: Dict[str, float], emotion: str = "neutral", user_id: str = "default-user", limit: int = 10):
        emotion = (emotion or "neutral").strip().lower()
        token = self._get_token()
        curated_tracks = self._static_indian_fallback(emotion=emotion, limit=max(limit * 3, 20))
        merged_tracks = self._rotate_for_user(curated_tracks, user_id=user_id, emotion=emotion)
        if token:
            # Prioritize Indian-focused search first, then fill with feature-based recommendations.
            merged_tracks.extend(self._spotify_search_tracks(token=token, emotion=emotion, limit=max(limit * 2, 12)))

        merged_tracks.extend(self._fallback_tracks(emotion=emotion, user_id=user_id, limit=max(limit * 2, 12)))

        # Generic recommendations are used as backfill only, to keep Indian songs dominant.
        if token:
            merged_tracks.extend(
                self._spotify_recommendation_tracks(
                    token=token,
                    target_features=target_features,
                    emotion=emotion,
                    limit=max(limit, 8),
                )
            )

        unique = self._dedupe_tracks(merged_tracks)
        if not unique:
            unique = self._static_indian_fallback(emotion=emotion, limit=max(limit * 2, 12))
        return unique[:limit]

    def _spotify_recommendation_tracks(self, token: str, target_features: Dict[str, float], emotion: str, limit: int):
        genre_candidates = list(self.EMOTION_GENRES.get(emotion, self.EMOTION_GENRES["neutral"]))
        genre_candidates.extend(["pop,chill,dance", "dance,pop,indie-pop"])

        for genres in genre_candidates:
            params = {
                "limit": limit,
                "market": self.settings.spotify_market or "IN",
                "seed_genres": genres,
                "target_tempo": target_features.get("target_tempo", 110),
                "target_valence": target_features.get("target_valence", 0.5),
                "target_energy": target_features.get("target_energy", 0.5),
            }
            try:
                resp = requests.get(self.RECOMMENDATION_URL, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=10)
                if resp.status_code >= 400:
                    self.logger.warning("Spotify recommendation API failed with %s for genres=%s", resp.status_code, genres)
                    continue

                items = resp.json().get("tracks", [])
                tracks = []
                for item in items:
                    tracks.append(
                        {
                            "id": item.get("id", ""),
                            "name": item.get("name", "Unknown"),
                            "artist": ", ".join([a.get("name", "") for a in item.get("artists", [])]),
                            "preview_url": item.get("preview_url"),
                            "external_url": item.get("external_urls", {}).get("spotify"),
                            "image_url": (item.get("album", {}).get("images") or [{}])[0].get("url"),
                            "embed_url": f"https://open.spotify.com/embed/track/{item.get('id', '')}" if item.get("id") else None,
                        }
                    )
                if tracks:
                    return tracks
            except requests.RequestException as exc:
                self.logger.warning("Spotify recommendation request failed (%s); trying fallback genres", exc)

        return []

    def _spotify_search_tracks(self, token: str, emotion: str, limit: int):
        terms = self.EMOTION_SEARCH_TERMS.get(emotion, self.EMOTION_SEARCH_TERMS["neutral"])
        tracks = []
        for term in terms:
            try:
                resp = requests.get(
                    self.TRACK_SEARCH_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "q": f"{term} hindi OR bollywood OR punjabi OR tamil OR telugu",
                        "type": "track",
                        "market": self.settings.spotify_market or "IN",
                        "limit": max(8, limit // 2),
                    },
                    timeout=10,
                )
                if resp.status_code >= 400:
                    self.logger.warning("Spotify track search API failed with %s for term=%s", resp.status_code, term)
                    continue

                items = resp.json().get("tracks", {}).get("items", [])
                for item in items:
                    tracks.append(
                        {
                            "id": item.get("id", ""),
                            "name": item.get("name", "Unknown"),
                            "artist": ", ".join([a.get("name", "") for a in item.get("artists", [])]),
                            "preview_url": item.get("preview_url"),
                            "external_url": item.get("external_urls", {}).get("spotify"),
                            "image_url": (item.get("album", {}).get("images") or [{}])[0].get("url"),
                            "embed_url": f"https://open.spotify.com/embed/track/{item.get('id', '')}" if item.get("id") else None,
                        }
                    )
            except requests.RequestException as exc:
                self.logger.warning("Spotify search request failed (%s) for term=%s", exc, term)

        return tracks

    def _fallback_tracks(self, emotion: str, user_id: str, limit: int):
        # Keyless Indian catalogue fallback with preview clips.
        try:
            terms = self.EMOTION_SEARCH_TERMS.get(emotion, self.EMOTION_SEARCH_TERMS["neutral"])
            term = random.choice(terms)
            resp = requests.get(
                self.ITUNES_SEARCH_URL,
                params={
                    "term": term,
                    "media": "music",
                    "entity": "song",
                    "country": "IN",
                    "limit": max(20, limit * 3),
                },
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            preferred = []
            for item in results:
                blob = " ".join(
                    str(x or "").lower()
                    for x in [item.get("trackName"), item.get("artistName"), item.get("collectionName")]
                )
                if any(hint in blob for hint in self.INDIAN_HINTS):
                    preferred.append(item)
            rows_source = preferred if preferred else results

            rows = []
            for item in rows_source:
                track_name = item.get("trackName")
                artist_name = item.get("artistName")
                if not track_name or not artist_name:
                    continue

                track_id = item.get("trackId", f"{track_name}-{artist_name}")
                artwork = item.get("artworkUrl100")
                image_url = artwork.replace("100x100bb", "512x512bb") if isinstance(artwork, str) else None

                rows.append(
                    {
                        "id": f"itunes-{track_id}",
                        "name": track_name,
                        "artist": artist_name,
                        "preview_url": item.get("previewUrl"),
                        "external_url": item.get("trackViewUrl"),
                        "image_url": image_url,
                        "embed_url": None,
                    }
                )

            if rows:
                seed = hashlib.sha256(
                    f"{user_id}:{emotion}:{len(rows)}:{int(time.time())}:{random.random()}".encode("utf-8")
                ).hexdigest()
                random.Random(seed).shuffle(rows)
                return rows[:limit]
        except requests.RequestException as exc:
            self.logger.warning("iTunes fallback failed (%s); using static backup catalog", exc)

        return self._static_indian_fallback(emotion=emotion, limit=limit)

    def _static_indian_fallback(self, emotion: str, limit: int):
        catalog_by_emotion = {
            "happy": [
                ("Kesariya", "Arijit Singh"),
                ("Malang Sajna", "Sachet-Parampara"),
                ("Kala Chashma", "Amar Arshi"),
                ("What Jhumka", "Arijit Singh"),
                ("Badtameez Dil", "Pritam"),
            ],
            "sad": [
                ("Love You Zindagi", "Amit Trivedi"),
                ("Ilahi", "Arijit Singh"),
                ("Safarnama", "Lucky Ali"),
                ("Phir Se Ud Chala", "Mohit Chauhan"),
                ("Kar Har Maidaan Fateh", "Sukhwinder Singh"),
            ],
            "angry": [
                ("Kun Faya Kun", "A. R. Rahman"),
                ("Iktara", "Kavita Seth"),
                ("Shaam", "Amit Trivedi"),
                ("Kabira", "Arijit Singh"),
                ("Phir Le Aya Dil", "Arijit Singh"),
            ],
            "fear": [
                ("Kho Gaye Hum Kahan", "Jasleen Royal"),
                ("Iktara", "Kavita Seth"),
                ("Raabta", "Arijit Singh"),
                ("Kun Faya Kun", "A. R. Rahman"),
                ("Khaabon Ke Parinday", "Alyssa Mendonsa"),
            ],
            "disgust": [
                ("Aao Milo Chale", "Shaan"),
                ("Hawayein", "Arijit Singh"),
                ("Safarnama", "Lucky Ali"),
                ("Khairiyat", "Arijit Singh"),
                ("Ranjha", "B Praak"),
            ],
            "surprise": [
                ("Naatu Naatu", "Rahul Sipligunj"),
                ("Jhoome Jo Pathaan", "Arijit Singh"),
                ("Ghungroo", "Arijit Singh"),
                ("Param Sundari", "Shreya Ghoshal"),
                ("Nachde Ne Saare", "Jasleen Royal"),
            ],
            "neutral": [
                ("Heeriye", "Jasleen Royal"),
                ("Tere Vaaste", "Sachin-Jigar"),
                ("Raatan Lambiyan", "Jubin Nautiyal"),
                ("Tu Hai Kahan", "AUR"),
                ("Kesariya", "Arijit Singh"),
            ],
        }

        tracks = []
        rows = list(catalog_by_emotion.get(emotion, catalog_by_emotion["neutral"]))
        random.shuffle(rows)
        for index, (name, artist) in enumerate(rows[:limit], start=1):
            query = quote_plus(f"{name} {artist}")
            tracks.append(
                {
                    "id": f"fallback-{emotion}-{index}",
                    "name": name,
                    "artist": artist,
                    "preview_url": None,
                    "external_url": f"https://open.spotify.com/search/{query}",
                    "image_url": None,
                    "embed_url": None,
                }
            )
        return tracks

    def _dedupe_tracks(self, tracks):
        unique = []
        seen = set()
        for track in tracks:
            key = (
                str(track.get("id", "")).strip().lower(),
                str(track.get("name", "")).strip().lower(),
                str(track.get("artist", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(track)
        return unique

    def _rotate_for_user(self, tracks, user_id: str, emotion: str):
        if not tracks:
            return []
        ordered = list(tracks)
        bucket = int(time.time() // 20)  # rotate every ~20 seconds in active sessions
        seed = hashlib.sha256(f"{user_id}:{emotion}:{bucket}:{len(ordered)}".encode("utf-8")).hexdigest()
        offset = int(seed[:8], 16) % len(ordered)
        return ordered[offset:] + ordered[:offset]
