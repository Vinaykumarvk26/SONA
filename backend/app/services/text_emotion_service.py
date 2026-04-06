from __future__ import annotations

import re
from typing import Dict

from app.ml.constants import FER_EMOTIONS


NEGATIONS = {
    "not",
    "no",
    "never",
    "dont",
    "don't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "cant",
    "can't",
    "couldnt",
    "couldn't",
    "wont",
    "won't",
    "without",
    "hardly",
}

INTENSIFIERS = {
    "very",
    "really",
    "so",
    "too",
    "quite",
    "extremely",
    "super",
    "totally",
    "absolutely",
    "much",
}

EMOTION_KEYWORDS = {
    "happy": {
        "happy", "glad", "joy", "joyful", "cheerful", "great", "good", "awesome", "amazing",
        "excited", "delighted", "smile", "smiling", "fine", "fantastic", "wonderful", "love",
        "loved", "loving", "pleased", "thrilled", "yay",
    },
    "sad": {
        "sad", "down", "upset", "depressed", "cry", "crying", "hurt", "lonely", "miserable",
        "unhappy", "bad", "heartbroken", "tired", "low", "pain", "broken", "miss", "empty",
    },
    "angry": {
        "angry", "mad", "furious", "annoyed", "irritated", "rage", "raging", "hate", "hated",
        "frustrated", "frustrating", "shouting", "fight", "fighting", "pissed",
    },
    "fear": {
        "fear", "afraid", "scared", "nervous", "worried", "anxious", "panic", "panicking",
        "terrified", "frightened", "stress", "stressed", "tense",
    },
    "disgust": {
        "disgust", "disgusted", "disgusting", "gross", "nasty", "awful", "sick", "dirty", "toxic", "creepy",
        "horrible", "dislike", "disliked", "revolting", "filthy", "stinks", "smelly", "yuck", "ew",
        "cringe", "cringy", "vomit", "repulsive", "unpleasant", "icky",
    },
    "surprise": {
        "surprise", "surprised", "shocked", "wow", "unexpected", "suddenly", "sudden",
        "unbelievable", "amazed", "astonished", "startled",
    },
    "neutral": {
        "okay", "ok", "fine", "normal", "regular", "calm", "steady", "neutral", "alright",
    },
}

EMOTION_PHRASES = {
    "happy": [
        "i am happy", "im happy", "i feel happy", "good day", "great day", "feeling good",
        "feeling great", "i am excited", "im excited", "i feel amazing",
    ],
    "sad": [
        "i am sad", "im sad", "i feel sad", "feeling low", "not feeling good", "very upset",
        "i want to cry",
    ],
    "angry": [
        "i am angry", "im angry", "i feel angry", "very mad", "so annoyed", "very frustrated",
    ],
    "fear": [
        "i am scared", "im scared", "i feel scared", "very afraid", "i am nervous", "panic attack",
    ],
    "disgust": [
        "this is disgusting", "i feel disgusted", "i am disgusted", "so gross", "very gross", "i hate this",
        "this makes me sick", "this is nasty", "this is gross", "this is horrible", "it smells bad", "this stinks",
        "ew this is gross", "yuck this is gross",
    ],
    "surprise": [
        "i am surprised", "im surprised", "what just happened", "i did not expect that", "oh my god",
    ],
}

NEGATED_BOOST = {
    "happy": "sad",
    "sad": "neutral",
    "angry": "neutral",
    "fear": "neutral",
    "disgust": "neutral",
    "surprise": "neutral",
    "neutral": "sad",
}


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in scores.values())
    if total <= 0:
        return {emotion: (1.0 / len(FER_EMOTIONS)) for emotion in FER_EMOTIONS}
    return {emotion: max(0.0, scores.get(emotion, 0.0)) / total for emotion in FER_EMOTIONS}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def analyze_text_emotion(transcript: str) -> dict | None:
    raw = (transcript or "").strip()
    if not raw:
        return None

    text = raw.lower()
    tokens = _tokenize(text)
    if not tokens:
        return None

    scores = {emotion: 0.12 for emotion in FER_EMOTIONS}
    scores["neutral"] = 0.18
    evidence = 0

    for emotion, phrases in EMOTION_PHRASES.items():
        for phrase in phrases:
            if phrase in text:
                scores[emotion] += 2.2
                evidence += 2

    for idx, token in enumerate(tokens):
        window = tokens[max(0, idx - 3):idx]
        negated = any(word in NEGATIONS for word in window)
        intensity_hits = sum(1 for word in window[-2:] if word in INTENSIFIERS)
        intensity = 1.0 + (0.35 * intensity_hits)

        for emotion, keywords in EMOTION_KEYWORDS.items():
            if token not in keywords:
                continue
            evidence += 1
            if negated:
                scores[emotion] -= 0.5 * intensity
                scores[NEGATED_BOOST[emotion]] += 1.15 * intensity
            else:
                scores[emotion] += 1.0 * intensity

    if "but" in tokens or "however" in tokens:
        last_clause = text.split("but")[-1].split("however")[-1]
        clause_tokens = _tokenize(last_clause)
        for token in clause_tokens:
            for emotion, keywords in EMOTION_KEYWORDS.items():
                if token in keywords:
                    scores[emotion] += 0.6

    normalized = _normalize(scores)
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    top_label, top_conf = ranked[0]
    second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = min(0.98, top_conf + max(0.0, top_conf - second_conf) * 0.8 + min(0.16, evidence * 0.015))

    return {
        "transcript": raw,
        "scores": normalized,
        "label": top_label,
        "confidence": float(confidence),
        "evidence": evidence,
    }


def fuse_voice_and_text_emotion(voice_scores: Dict[str, float], text_analysis: dict | None) -> dict:
    if not text_analysis:
        normalized = _normalize(voice_scores)
        label, conf = max(normalized.items(), key=lambda item: item[1])
        return {
            "scores": normalized,
            "label": label,
            "confidence": float(conf),
            "weights": {"voice": 1.0, "text": 0.0},
            "text_analysis": None,
        }

    voice_norm = _normalize(voice_scores)
    text_scores = text_analysis["scores"]
    text_conf = float(text_analysis.get("confidence", 0.0))
    evidence = int(text_analysis.get("evidence", 0))
    text_weight = min(0.7, max(0.25, 0.25 + ((text_conf - 0.25) * 0.7) + min(0.15, evidence * 0.01)))
    voice_weight = 1.0 - text_weight

    fused_scores = {
        emotion: float((voice_weight * voice_norm.get(emotion, 0.0)) + (text_weight * text_scores.get(emotion, 0.0)))
        for emotion in FER_EMOTIONS
    }

    # Give transcript evidence a stronger say for disgust-related language because
    # the acoustic model tends to confuse disgust with sad/angry in live speech.
    if text_analysis["label"] == "disgust" and text_conf >= 0.42:
        fused_scores["disgust"] += 0.12 + min(0.08, evidence * 0.01)
        if voice_norm.get("sad", 0.0) > 0.18:
            fused_scores["sad"] *= 0.84
        if voice_norm.get("angry", 0.0) > 0.18:
            fused_scores["angry"] *= 0.9

    fused_scores = _normalize(fused_scores)
    label, conf = max(fused_scores.items(), key=lambda item: item[1])

    return {
        "scores": fused_scores,
        "label": label,
        "confidence": float(conf),
        "weights": {"voice": float(voice_weight), "text": float(text_weight)},
        "text_analysis": text_analysis,
    }
