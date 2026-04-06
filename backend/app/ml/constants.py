FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# RAVDESS label ids -> names
RAVDESS_EMOTION_MAP = {
    "01": "neutral",  # neutral
    "02": "neutral",  # calm
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",  # fearful
    "07": "disgust",
    "08": "surprise",  # surprised
}

SER_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
SER_TO_FER = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}
