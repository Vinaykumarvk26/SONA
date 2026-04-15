import { useEffect, useMemo, useRef, useState } from "react";
import Webcam from "react-webcam";
import ReactGA from "react-ga4";
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";
import EmotionTimeline from "./components/EmotionTimeline";
import {
  getMetricsOverview,
  getRecommendations,
  inferFrame,
  inferSpeech,
  logMetricsEvent,
  runModelEvaluation,
  submitEmotionFeedback,
} from "./api/emotionApi";
import { getCurrentUser, getMyPreferences, saveMyPreferences, signinWithGoogleAccessToken, signinWithIdentifier, signupWithDetails, signout, updateMyProfile } from "./api/authApi";

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend);

const EMOJI_BY_EMOTION = {
  happy: "\u{1F604}",
  sad: "\u{1F614}",
  angry: "\u{1F620}",
  fear: "\u{1F628}",
  disgust: "\u{1F922}",
  surprise: "\u{1F62E}",
  neutral: "\u{1F610}",
};

const STICKER_BY_EMOTION = {
  happy: "/stickers/happy.jpeg",
  sad: "/stickers/sad.jpeg",
  angry: "/stickers/angry.jpeg",
  fear: "/stickers/fear.jpeg",
  disgust: "/stickers/disgust.jpeg",
  surprise: "/stickers/surprise.jpeg",
  neutral: "/stickers/neutral.jpeg",
  idle: "/stickers/idle.jpeg",
};

const ALL_EMOTIONS = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"];
const LANGUAGE_OPTIONS = ["Hindi", "English", "Telugu", "Tamil", "Kannada", "Malayalam"];
const SOURCE_OPTIONS = ["Spotify", "Local", "Import"];
const SOURCE_DESCRIPTIONS = {
  Spotify: "Fetches online preview tracks based on detected mood.",
  Local: "Prioritizes locally available songs in your playlist logic.",
  Import: "Uses imported library tracks with your mood + language order.",
};
const DEFAULT_PROFILE_FORM = { username: "", full_name: "", phone: "", location: "", bio: "" };

const NAV_ITEMS = [
  { key: "home", label: "Home", icon: "O" },
  { key: "metrics", label: "Metrics", icon: "#" },
  { key: "profile", label: "Settings", icon: "*" },
  { key: "contacts", label: "Contacts", icon: "@" },
];
const APP_PAGE_KEYS = new Set(NAV_ITEMS.map((item) => item.key));
const GOOGLE_CLIENT_ID = (import.meta.env.VITE_GOOGLE_CLIENT_ID || "").trim();
const SETTINGS_STORAGE_KEY = "sona_user_settings_v1";
const LANDING_MARQUEE_ITEMS = [
  "Neural Listening",
  "Face Signal",
  "Voice Pulse",
  "Silent Read",
  "Emotional Match",
  "Midnight Playback",
  "Adaptive Queue",
  "Living Frequency",
  "Aura Sync",
  "Sentiment Flow",
  "Human Tone",
  "Signal to Sound",
  "Reactive Audio",
  "Cinematic Mode",
  "Deep Focus",
  "Soft Static",
  "Echo Memory",
];
const LANDING_MARQUEE_THEMES = [
  "sage-garden",
  "soft-pop",
  "solar-dusk",
  "starry-night",
  "sunset-horizon",
  "supabase",
  "t3-chat",
  "tangerine",
  "twitter",
  "vintage-paper",
  "mocha-mousse",
  "nature",
  "neo-brutalism",
  "ocean-breeze",
  "pastel-dreams",
  "perpetuity",
  "quantum-rose",
];
const LANDING_MARQUEE_ROWS = [
  LANDING_MARQUEE_ITEMS.slice(0, 6),
  LANDING_MARQUEE_ITEMS.slice(6, 12),
  LANDING_MARQUEE_ITEMS.slice(12),
];
const LANDING_STEPS = [
  {
    index: "01",
    title: "Capture Emotion",
    body: "Start with live camera or voice input so SONA reads the strongest emotional signal before playback begins.",
  },
  {
    index: "02",
    title: "Lock the Mood",
    body: "Once an emotion is detected, the result stays locked until you manually retake. No drifting state, no noisy refresh.",
  },
  {
    index: "03",
    title: "Play the Match",
    body: "SONA builds a playlist around that locked emotion, then playback follows your source, mode, and language priorities.",
  },
];
const LANDING_TESTIMONIALS = [
  {
    name: "Aarav Menon",
    handle: "@mooddriven",
    quote: "The flow feels intentional. Face lock, voice capture, and playlist response all read like one continuous product story.",
  },
  {
    name: "Ritika Shah",
    handle: "@uxritika",
    quote: "What stands out is control. Retake actually resets the player, and that makes the recommendation loop feel trustworthy.",
  },
  {
    name: "Kiran Dev",
    handle: "@audiosystems",
    quote: "The dark interface and live modules make it feel closer to an audio instrument than a generic music dashboard.",
  },
  {
    name: "Naveen Rao",
    handle: "@neuralbeats",
    quote: "Language priority plus manual playback mode makes the product feel personal instead of algorithmic for the sake of it.",
  },
  {
    name: "Ishita Verma",
    handle: "@sonicish",
    quote: "The playlist lock is the right call. It gives the recommendation a clear beginning and end instead of a restless live feed.",
  },
  {
    name: "Rahul Sinha",
    handle: "@rahulsounds",
    quote: "Voice mode feels like an instrument panel, not a toy feature. The waveform area makes the whole flow easier to trust.",
  },
  {
    name: "Meera Kapoor",
    handle: "@meeraui",
    quote: "The landing and dashboard finally feel like the same product. That continuity matters more than adding more visual tricks.",
  },
  {
    name: "Dev Patel",
    handle: "@depthsignal",
    quote: "Manual mode is useful. Sometimes I want the emotion read and the queue prepared without having playback fire instantly.",
  },
  {
    name: "Sana Iqbal",
    handle: "@sanaframes",
    quote: "The face and voice panels feel purposeful now. One takes focus, one stays out of the way. That decision cleaned up the experience.",
  },
  {
    name: "Vikram Joshi",
    handle: "@auramotion",
    quote: "The product feels more premium when the player, playlist, and detected state all move as one locked system.",
  },
  {
    name: "Pooja Nair",
    handle: "@poojalistens",
    quote: "Retake stopping playback is a strong UX rule. It avoids confusion and makes the new recommendation cycle feel deliberate.",
  },
  {
    name: "Aditya Roy",
    handle: "@tonelab",
    quote: "This works because it does not over-explain itself. Detect, lock, and play is a simple loop users understand immediately.",
  },
];
const LANDING_FAQS = [
  {
    question: "How does SONA choose songs?",
    answer: "SONA uses the locked emotional result from your face or voice session, then shapes the playlist around your playback mode, source choice, and language order.",
  },
  {
    question: "Does the detected emotion keep changing live?",
    answer: "No. Once the result is locked, it stays fixed. Only a manual retake unlocks the state and generates a new recommendation cycle.",
  },
  {
    question: "Can I use only face or only voice input?",
    answer: "Yes. Each module works independently. You can stay in facial mode, stay in voice mode, and retake either one without forcing the other.",
  },
  {
    question: "What happens in manual playback mode?",
    answer: "The playlist is still generated, but audio does not start automatically. You decide when to play, pause, or switch tracks.",
  },
];
const TRACK_LANGUAGE_MAP = {
  kesariya: "Hindi",
  "malang sajna": "Hindi",
  "kala chashma": "Hindi",
  "what jhumka": "Hindi",
  "badtameez dil": "Hindi",
  "channa mereya": "Hindi",
  "agar tum saath ho": "Hindi",
  "tum hi ho": "Hindi",
  shayad: "Hindi",
  khairiyat: "Hindi",
  zinda: "Hindi",
  "kar har maidaan fateh": "Hindi",
  "apna bana le": "Hindi",
  ilahi: "Hindi",
  "love you zindagi": "Hindi",
  "phir le aya dil": "Hindi",
  iktara: "Hindi",
  raabta: "Hindi",
  "kun faya kun": "Hindi",
  kabira: "Hindi",
  safarnama: "Hindi",
  "aao milo chale": "Hindi",
  hawayein: "Hindi",
  ranjha: "Hindi",
  heeriye: "Hindi",
  "tere vaaste": "Hindi",
  "raatan lambiyan": "Hindi",
  satranga: "Hindi",
  "tu hai kahan": "Hindi",
  "naatu naatu": "Telugu",
  yaani: "Tamil",
  hukum: "Tamil",
  manasilayo: "Malayalam",
  "deva deva": "Hindi",
};

function normalizeTrackText(value) {
  return String(value || "").trim().toLowerCase();
}

function inferTrackLanguage(track) {
  const title = normalizeTrackText(track?.name);
  if (TRACK_LANGUAGE_MAP[title]) return TRACK_LANGUAGE_MAP[title];
  if (/\b(yaani|hukum|jailer|anirudh)\b/.test(title)) return "Tamil";
  if (/\b(naatu|telugu)\b/.test(title)) return "Telugu";
  if (/\b(malayalam|manasilayo)\b/.test(title)) return "Malayalam";
  return "Hindi";
}

function shapePlaylistWithSettings(tracks, source, languageOrder) {
  const normalized = (tracks || []).map((track, index) => ({
    ...track,
    _originalIndex: index,
    language: inferTrackLanguage(track),
  }));

  const ranked = normalized.sort((a, b) => {
    const aLangRank = languageOrder.indexOf(a.language);
    const bLangRank = languageOrder.indexOf(b.language);
    const aRank = aLangRank === -1 ? 999 : aLangRank;
    const bRank = bLangRank === -1 ? 999 : bLangRank;
    if (aRank !== bRank) return aRank - bRank;
    const aPlayable = a.preview_url ? 0 : 1;
    const bPlayable = b.preview_url ? 0 : 1;
    if (aPlayable !== bPlayable) return aPlayable - bPlayable;
    return a._originalIndex - b._originalIndex;
  });

  if (source === "Local") {
    const localPlayable = ranked.filter((track) => track.preview_url);
    return (localPlayable.length >= 3 ? localPlayable : ranked).slice(0, 10);
  }

  if (source === "Import") {
    const imported = [...ranked].sort((a, b) => {
      const aHasLink = a.external_url || a.embed_url ? 0 : 1;
      const bHasLink = b.external_url || b.embed_url ? 0 : 1;
      if (aHasLink !== bHasLink) return aHasLink - bHasLink;
      return a._originalIndex - b._originalIndex;
    });
    return imported.slice(0, 10);
  }

  return ranked.slice(0, 10);
}

function selectPreferredTrackIndex(tracks, previousTrackId = "") {
  const list = tracks || [];
  if (!list.length) return -1;

  if (previousTrackId) {
    const sameTrackIndex = list.findIndex((track) => (track?.id || "") === previousTrackId);
    if (sameTrackIndex >= 0) return sameTrackIndex;
  }

  const firstPlayableIndex = list.findIndex((track) => !!track?.preview_url);
  if (firstPlayableIndex >= 0) return firstPlayableIndex;
  return 0;
}

function ThemeIcon({ isLightMode }) {
  return isLightMode ? (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M21 12.8A9 9 0 1 1 11.2 3a7.2 7.2 0 1 0 9.8 9.8Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ) : (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 2.8v2.4M12 18.8v2.4M21.2 12h-2.4M5.2 12H2.8M18.5 5.5l-1.7 1.7M7.2 16.8l-1.7 1.7M18.5 18.5l-1.7-1.7M7.2 7.2 5.5 5.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function ProfileIcon() {
  return (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="8" r="3.6" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M5.5 19.2a6.8 6.8 0 0 1 13 0"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M19.4 15a1 1 0 0 0 .2 1.1l.1.1a1.6 1.6 0 1 1-2.3 2.3l-.1-.1a1 1 0 0 0-1.1-.2 1 1 0 0 0-.6.9V19a1.6 1.6 0 1 1-3.2 0v-.2a1 1 0 0 0-.7-.9 1 1 0 0 0-1.1.2l-.1.1a1.6 1.6 0 1 1-2.3-2.3l.1-.1A1 1 0 0 0 8 15a1 1 0 0 0-.9-.6H7a1.6 1.6 0 1 1 0-3.2h.2A1 1 0 0 0 8 10a1 1 0 0 0-.2-1.1l-.1-.1a1.6 1.6 0 1 1 2.3-2.3l.1.1A1 1 0 0 0 11.2 7h.2a1 1 0 0 0 .9-.7V6a1.6 1.6 0 1 1 3.2 0v.2a1 1 0 0 0 .6.9 1 1 0 0 0 1.1-.2l.1-.1a1.6 1.6 0 1 1 2.3 2.3l-.1.1A1 1 0 0 0 19 10c0 .4.4.8.9.8h.2a1.6 1.6 0 1 1 0 3.2h-.2a1 1 0 0 0-.9.6Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M14 6V4.8A1.8 1.8 0 0 0 12.2 3H6.8A1.8 1.8 0 0 0 5 4.8v14.4A1.8 1.8 0 0 0 6.8 21h5.4A1.8 1.8 0 0 0 14 19.2V18"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10 12h9M16 8l4 4-4 4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ArrowUpIcon() {
  return (
    <svg className="ui-icon small" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 18V6M12 6l-5 5M12 6l5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ArrowDownIcon() {
  return (
    <svg className="ui-icon small" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 6v12M12 18l-5-5M12 18l5-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MusicNoteIcon() {
  return (
    <svg className="ui-icon artwork-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M15 5v9.2a2.8 2.8 0 1 1-1.6-2.5V7.8L8 9.2v7a2.8 2.8 0 1 1-1.6-2.5V8l8.6-3Z" fill="currentColor" />
    </svg>
  );
}

function PrevTrackIcon() {
  return (
    <svg className="ui-icon player-control-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 6v12M18 7l-7 5 7 5V7ZM11 7l-7 5 7 5V7Z" fill="currentColor" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg className="ui-icon player-control-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 6.8v10.4c0 .8.9 1.3 1.6.8l8-5.2a1 1 0 0 0 0-1.6l-8-5.2c-.7-.5-1.6 0-1.6.8Z" fill="currentColor" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg className="ui-icon player-control-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 6h3v12H8zM13 6h3v12h-3z" fill="currentColor" />
    </svg>
  );
}

function NextTrackIcon() {
  return (
    <svg className="ui-icon player-control-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M17 6v12M6 7l7 5-7 5V7Zm7 0 7 5-7 5V7Z" fill="currentColor" />
    </svg>
  );
}

function dataURLToBlob(dataURL) {
  const arr = dataURL.split(",");
  const mime = arr[0].match(/:(.*?);/)?.[1] || "image/jpeg";
  const bstr = atob(arr[1]);
  const n = bstr.length;
  const u8arr = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) u8arr[i] = bstr.charCodeAt(i);
  return new Blob([u8arr], { type: mime });
}

function normalizeEmotionPayload(payload) {
  const label = payload?.emotion || payload?.label || payload?.fused?.label || "neutral";
  const confidence = Number(payload?.confidence ?? payload?.fused?.confidence ?? 0);
  return {
    label: String(label).toLowerCase(),
    confidence: Number.isFinite(confidence) ? confidence : 0,
    transcript: String(payload?.transcript || "").trim(),
    textScores: payload?.text_scores || null,
    sourceWeights: payload?.source_weights || null,
  };
}

function extractApiError(err, fallbackMessage) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    if (typeof first?.msg === "string" && first.msg.trim()) return first.msg;
  }
  const message = err?.message;
  if (typeof message === "string" && message.trim()) return message;
  return fallbackMessage;
}

function statusTone(status) {
  if (status === "locked") return "ok";
  if (status === "detecting") return "info";
  if (status === "analyzing") return "warn";
  return "idle";
}

function statusLabel(status) {
  if (status === "locked") return "Result Locked";
  if (status === "analyzing") return "Analyzing";
  if (status === "detecting") return "Detecting";
  return "Idle";
}

function formatClockTime(value) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const totalSeconds = Math.floor(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function buildRecommendationUrl(base, userId, emotion, confidence) {
  const params = new URLSearchParams({
    user_id: String(userId || "default-user"),
    emotion: String(emotion || "neutral"),
    confidence: String(Number.isFinite(confidence) ? confidence : 0.5),
  });
  const root = String(base || "").replace(/\/+$/, "");
  return `${root}/recommendations?${params.toString()}`;
}

function normalizeUserIdentity(user) {
  if (!user) return null;
  const normalizedEmail = String(user.email || "").trim().toLowerCase();
  const normalizedUsername = String(user.username || "").trim().toLowerCase();
  return {
    ...user,
    email: normalizedEmail || user.email || "",
    username: normalizedUsername || user.username || "",
    full_name: String(user.full_name || "").trim(),
    phone: String(user.phone || "").trim(),
    location: String(user.location || "").trim(),
    bio: String(user.bio || "").trim(),
  };
}

function formatPercent(value) {
  const numeric = Number(value || 0);
  return `${Math.round(numeric * 100)}%`;
}

function getPageFromPath(pathname) {
  const path = String(pathname || "/").toLowerCase();
  if (path === "/" || path === "/landing") return "landing";
  if (path.startsWith("/app/")) {
    const page = path.replace("/app/", "").split("/")[0];
    if (APP_PAGE_KEYS.has(page)) return page;
  }
  return "home";
}

function getPathForPage(page, isAuthed) {
  if (!isAuthed) return "/";
  return `/app/${page}`;
}

function getTrackArtwork(track) {
  if (!track) return "";
  return track.image_url || track.image || track.artwork_url || track.album_image || "";
}

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function encodeWav(samples, sampleRate) {
  const pcm = floatTo16BitPCM(samples);
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, pcm.length * 2, true);
  let offset = 44;
  for (let i = 0; i < pcm.length; i += 1) {
    view.setInt16(offset, pcm[i], true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

export default function App() {
  const webcamRef = useRef(null);
  const imageInputRef = useRef(null);
  const audioRef = useRef(null);
  const mediaUnlockedRef = useRef(false);

  const processorRef = useRef(null);
  const pcmChunksRef = useRef([]);
  const sampleRateRef = useRef(16000);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const waveformRafRef = useRef(null);
  const waveformCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const googleTokenClientRef = useRef(null);
  const speechRecognitionRef = useRef(null);

  const [activePage, setActivePage] = useState(() => getPageFromPath(window.location.pathname));
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [isLightMode, setIsLightMode] = useState(false);
  const [activeDetectionPanel, setActiveDetectionPanel] = useState("face");
  const [authReady, setAuthReady] = useState(false);
  const [authUser, setAuthUser] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState("signin");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [authForm, setAuthForm] = useState({ username: "", email: "", identifier: "", password: "" });
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [openFaqIndex, setOpenFaqIndex] = useState(0);
  const [preferencesReady, setPreferencesReady] = useState(false);
  const [profileForm, setProfileForm] = useState(DEFAULT_PROFILE_FORM);
  const [profileBusy, setProfileBusy] = useState(false);
  const [profileNotice, setProfileNotice] = useState("");
  const [profileNoticeTone, setProfileNoticeTone] = useState("info");

  const [faceMode, setFaceMode] = useState("live");
  const [isCamOn, setIsCamOn] = useState(false);
  const [facePreview, setFacePreview] = useState("");

  const [faceStatus, setFaceStatus] = useState("idle");
  const [speechStatus, setSpeechStatus] = useState("idle");
  const [faceLock, setFaceLock] = useState(null);
  const [speechLock, setSpeechLock] = useState(null);
  const [uiError, setUiError] = useState("");
  const [speechTranscript, setSpeechTranscript] = useState("");
  const [speechInterimTranscript, setSpeechInterimTranscript] = useState("");

  const [rawRecommendations, setRawRecommendations] = useState([]);
  const [playlist, setPlaylist] = useState([]);
  const [currentTrackIndex, setCurrentTrackIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [seek, setSeek] = useState(0);
  const [duration, setDuration] = useState(0);
  const [sourceLabel, setSourceLabel] = useState("None");

  const [playbackMode, setPlaybackMode] = useState("autoplay");
  const [musicSource, setMusicSource] = useState("Spotify");
  const [languages, setLanguages] = useState([...LANGUAGE_OPTIONS]);

  const [sessionStats, setSessionStats] = useState({ emotionsDetected: 0, songsPlayed: 0, retakes: 0 });
  const [timelinePoints, setTimelinePoints] = useState([]);
  const [metricsOverview, setMetricsOverview] = useState({
    mongo_enabled: false,
    overall_model: { accuracy: 0, precision: 0, recall: 0, f1_score: 0 },
    facial_model: {},
    speech_model: {},
    live: { emotions_detected: 0, songs_played: 0, retakes: 0, recommendations_generated: 0, top_emotions: [], timeline: [], average_confidence: 0 },
    evaluation: { by_input_type: { face: {}, voice: {} }, emotion_distribution: [] },
  });
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [feedbackState, setFeedbackState] = useState({ face: "", speech: "" });

  const currentTrack = currentTrackIndex >= 0 ? playlist[currentTrackIndex] : null;
  const userIdentifier = useMemo(() => {
    const email = String(authUser?.email || "").trim().toLowerCase();
    const username = String(authUser?.username || "").trim().toLowerCase();
    return email || username || "default-user";
  }, [authUser]);
  const activeLock = useMemo(() => (activeDetectionPanel === "face" ? faceLock : speechLock), [activeDetectionPanel, faceLock, speechLock]);
  const activeSticker = STICKER_BY_EMOTION[activeLock?.label] || STICKER_BY_EMOTION.idle;
  const speechRecognitionSupported = typeof window !== "undefined" && !!(window.SpeechRecognition || window.webkitSpeechRecognition);

  const navigateToPage = (page, options = {}) => {
    const nextPage = APP_PAGE_KEYS.has(page) ? page : "home";
    const path = getPathForPage(nextPage, true);
    if (window.location.pathname !== path) {
      if (options.replace) window.history.replaceState({}, "", path);
      else window.history.pushState({}, "", path);
    }
    setActivePage(nextPage);
  };

  const navigateToLanding = (options = {}) => {
    if (window.location.pathname !== "/") {
      if (options.replace) window.history.replaceState({}, "", "/");
      else window.history.pushState({}, "", "/");
    }
  };

  useEffect(() => {
    ReactGA.send({ hitType: "pageview", page: window.location.pathname });
  }, [activePage]);


  const handleBrandClick = () => {
    setShowProfileMenu(false);
    if (authUser) {
      navigateToPage("home");
    } else {
      navigateToLanding();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const drawWaveform = () => {
    const canvas = waveformCanvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) {
      waveformRafRef.current = requestAnimationFrame(drawWaveform);
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      waveformRafRef.current = requestAnimationFrame(drawWaveform);
      return;
    }
    const width = canvas.clientWidth || 700;
    const height = canvas.clientHeight || 320;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(2, 6, 23, 0.8)";
    ctx.fillRect(0, 0, width, height);

    const barCount = 64;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const barWidth = width / barCount;
    for (let i = 0; i < barCount; i += 1) {
      const v = data[i] / 255;
      const barHeight = Math.max(6, v * (height - 24));
      const x = i * barWidth;
      const y = height - barHeight;
      const g = ctx.createLinearGradient(0, y, 0, height);
      g.addColorStop(0, "#38bdf8");
      g.addColorStop(1, "#0ea5e9");
      ctx.fillStyle = g;
      ctx.fillRect(x + 1, y, Math.max(2, barWidth - 2), barHeight);
    }
    waveformRafRef.current = requestAnimationFrame(drawWaveform);
  };

  const stopWaveform = () => {
    if (waveformRafRef.current) {
      cancelAnimationFrame(waveformRafRef.current);
      waveformRafRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    const canvas = waveformCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const stopSpeechCapture = () => {
    if (speechRecognitionRef.current) {
      try {
        speechRecognitionRef.current.onresult = null;
        speechRecognitionRef.current.onerror = null;
        speechRecognitionRef.current.onend = null;
        speechRecognitionRef.current.stop();
      } catch {
        // Ignore recognition stop failures during cleanup.
      }
      speechRecognitionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    stopWaveform();
  };

  const stopAndResetPlayer = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
    setSeek(0);
    setDuration(0);
    setPlaylist([]);
    setCurrentTrackIndex(-1);
  };

  const handleRetakeReset = (channel = "general") => {
    stopAndResetPlayer();
    setSessionStats((prev) => ({ ...prev, retakes: prev.retakes + 1 }));
    logMetricsEvent({
      user_id: userIdentifier,
      category: "ui",
      action: channel === "face" ? "retake_face" : "retake_voice",
      metadata: { panel: channel },
    }).catch(() => {});
  };

  const primeMediaPlayback = async () => {
    if (mediaUnlockedRef.current) return true;
    try {
      const silentClip = "data:audio/wav;base64,UklGRlIAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YS4AAAAA";
      if (audioRef.current) {
        const player = audioRef.current;
        const previousSrc = player.src || "";
        const previousMuted = player.muted;
        const previousVolume = player.volume;
        player.muted = true;
        player.volume = 0;
        player.src = silentClip;
        player.load();
        await player.play();
        player.pause();
        player.currentTime = 0;
        player.removeAttribute("src");
        if (previousSrc) player.src = previousSrc;
        player.muted = previousMuted;
        player.volume = previousVolume;
        player.load();
      } else {
        const unlockAudio = new Audio(silentClip);
        unlockAudio.muted = true;
        unlockAudio.volume = 0;
        await unlockAudio.play();
        unlockAudio.pause();
        unlockAudio.currentTime = 0;
      }
      mediaUnlockedRef.current = true;
      return true;
    } catch {
      return false;
    }
  };

  const refreshPlaylist = async (emotion) => {
    let rec;
    try {
      rec = await getRecommendations(userIdentifier, emotion.label, emotion.confidence);
    } catch (error) {
      if (userIdentifier !== "default-user") {
        try {
          rec = await getRecommendations("default-user", emotion.label, emotion.confidence);
        } catch {
          rec = null;
        }
      } else {
        rec = null;
      }
    }

    if (!rec?.tracks?.length) {
      const runtimeHost = typeof window !== "undefined" ? (window.location.hostname || "127.0.0.1") : "127.0.0.1";
      const candidates = [
        buildRecommendationUrl("/api/v1", userIdentifier, emotion.label, emotion.confidence),
        buildRecommendationUrl("/api/v1", "default-user", emotion.label, emotion.confidence),
        buildRecommendationUrl(`http://${runtimeHost}:8010/api/v1`, userIdentifier, emotion.label, emotion.confidence),
        buildRecommendationUrl(`http://${runtimeHost}:8010/api/v1`, "default-user", emotion.label, emotion.confidence),
        buildRecommendationUrl("http://127.0.0.1:8010/api/v1", userIdentifier, emotion.label, emotion.confidence),
        buildRecommendationUrl("http://127.0.0.1:8010/api/v1", "default-user", emotion.label, emotion.confidence),
      ];

      for (const url of candidates) {
        try {
          const resp = await fetch(url, { method: "GET" });
          if (!resp.ok) continue;
          const payload = await resp.json();
          if (payload?.tracks?.length) {
            rec = payload;
            break;
          }
        } catch {
          // try next candidate
        }
      }
    }

    if (!rec?.tracks?.length) {
      throw new Error("No recommendations available right now.");
    }
    const tracks = rec?.tracks || [];
    const shapedTracks = shapePlaylistWithSettings(tracks, musicSource, languages);
    if (!shapedTracks.length) {
      throw new Error("No playable tracks available right now.");
    }
    setRawRecommendations(tracks);
    setPlaylist(shapedTracks);
    setSourceLabel(musicSource);
    setCurrentTrackIndex(selectPreferredTrackIndex(shapedTracks));
  };

  const lockEmotion = async (channel, emotion) => {
    if (!emotion) return;
    ReactGA.event({ category: "Engagement", action: "analysis_complete", label: emotion?.label });
    if (channel === "face") setFaceLock(emotion);
    if (channel === "speech") setSpeechLock(emotion);
    setFeedbackState((prev) => ({ ...prev, [channel]: "" }));
    setSessionStats((prev) => ({ ...prev, emotionsDetected: prev.emotionsDetected + 1 }));
    setTimelinePoints((prev) => [...prev.slice(-24), { timestamp: new Date().toISOString(), confidence: emotion.confidence }]);
    try {
      await refreshPlaylist(emotion);
      setUiError("");
    } catch (error) {
      const message = extractApiError(error, "Playlist refresh failed.");
      console.warn("Playlist refresh failed after emotion lock:", message);
      setUiError((prev) => prev || "Emotion locked, but playlist could not be refreshed.");
    }
  };

  const handleFaceDetect = async () => {
    setUiError("");
    try {
      ReactGA.event({ category: "Engagement", action: "start_facial_scan" });
      await primeMediaPlayback();
      setFaceStatus("detecting");
      let imageBlob;
      if (faceMode === "live") {
        if (!isCamOn) throw new Error("Start camera first.");
        const shot = webcamRef.current?.getScreenshot();
        if (!shot) throw new Error("Could not capture camera frame.");
        imageBlob = dataURLToBlob(shot);
      } else {
        const file = imageInputRef.current?.files?.[0];
        if (!file) {
          imageInputRef.current?.click();
          setFaceStatus("idle");
          setUiError("Please choose an image before submitting.");
          return;
        }
        if (!String(file.type || "").startsWith("image/")) {
          setFaceStatus("idle");
          setUiError("Invalid file type. Please choose an image file.");
          return;
        }
        imageBlob = file;
      }
      setFaceStatus("analyzing");
      const response = await inferFrame(imageBlob, userIdentifier);
      const emotion = normalizeEmotionPayload(response);
      await lockEmotion("face", emotion);
      setFaceStatus("locked");
    } catch (error) {
      setFaceStatus("idle");
      setUiError(extractApiError(error, "Face detection failed."));
    }
  };

  const handleSpeechStart = async () => {
    setUiError("");
    setSpeechTranscript("");
    setSpeechInterimTranscript("");
    try {
      ReactGA.event({ category: "Engagement", action: "start_audio_scan" });
      await primeMediaPlayback();
      setSpeechStatus("detecting");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(audioContext.destination);

      pcmChunksRef.current = [];
      sampleRateRef.current = audioContext.sampleRate || 16000;
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(input));
      };

      audioContextRef.current = audioContext;
      sourceRef.current = source;
      analyserRef.current = analyser;
      processorRef.current = processor;
      drawWaveform();

      const RecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (RecognitionCtor) {
        const recognition = new RecognitionCtor();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        recognition.onresult = (event) => {
          let finalText = "";
          let interimText = "";
          for (let i = event.resultIndex; i < event.results.length; i += 1) {
            const segment = event.results[i]?.[0]?.transcript || "";
            if (event.results[i].isFinal) finalText += `${segment} `;
            else interimText += `${segment} `;
          }
          if (finalText.trim()) {
            setSpeechTranscript((prev) => `${prev} ${finalText}`.trim());
          }
          setSpeechInterimTranscript(interimText.trim());
        };
        recognition.onerror = () => {
          setSpeechInterimTranscript("");
        };
        recognition.onend = () => {
          speechRecognitionRef.current = null;
        };
        recognition.start();
        speechRecognitionRef.current = recognition;
      }
    } catch {
      setSpeechStatus("idle");
      setUiError("Microphone permission denied or unavailable.");
    }
  };

  const handleSpeechStop = async () => {
    try {
      const chunks = pcmChunksRef.current;
      if (!chunks || chunks.length === 0) {
        stopSpeechCapture();
        setSpeechStatus("idle");
        setUiError("No voice captured. Please speak and try again.");
        return;
      }

      setSpeechStatus("analyzing");
      const total = chunks.reduce((acc, arr) => acc + arr.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      const wavBlob = encodeWav(merged, sampleRateRef.current || 16000);
      const transcriptForAnalysis = `${speechTranscript} ${speechInterimTranscript}`.trim();
      stopSpeechCapture();
      const response = await inferSpeech(wavBlob, transcriptForAnalysis, userIdentifier);
      const emotion = normalizeEmotionPayload(response);
      if (emotion.transcript) {
        setSpeechTranscript(emotion.transcript);
      }
      setSpeechInterimTranscript("");
      await lockEmotion("speech", emotion);
      setSpeechStatus("locked");
    } catch (error) {
      setSpeechStatus("idle");
      setUiError(error?.message || "Speech detection failed.");
    } finally {
      pcmChunksRef.current = [];
    }
  };

  const handleRetakeFace = () => {
    setFaceStatus("idle");
    setFaceLock(null);
    setFeedbackState((prev) => ({ ...prev, face: "" }));
    handleRetakeReset("face");
  };

  const handleReuploadPicture = () => {
    const input = imageInputRef.current;
    if (!input) return;
    input.value = "";
    setFacePreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return "";
    });
    input.click();
  };

  const handleRetakeSpeech = () => {
    stopSpeechCapture();
    setSpeechStatus("idle");
    setSpeechLock(null);
    setFeedbackState((prev) => ({ ...prev, speech: "" }));
    pcmChunksRef.current = [];
    setSpeechTranscript("");
    setSpeechInterimTranscript("");
    handleRetakeReset("voice");
  };

  const handleEmotionFeedback = async (channel, isCorrect) => {
    const active = channel === "face" ? faceLock : speechLock;
    if (!active) return;
    try {
      await submitEmotionFeedback({
        user_id: userIdentifier,
        input_type: channel,
        predicted_label: active.label,
        confidence_score: active.confidence || 0,
        is_correct: Boolean(isCorrect),
      });
      setFeedbackState((prev) => ({ ...prev, [channel]: isCorrect ? "correct" : "incorrect" }));
    } catch {
      setUiError("Could not save feedback right now.");
    }
  };

  const handleRunEvaluation = async (inputType = "all") => {
    setEvaluationBusy(true);
    try {
      await runModelEvaluation(inputType);
      const data = await getMetricsOverview(userIdentifier);
      setMetricsOverview(data);
    } catch (error) {
      setUiError(extractApiError(error, "Model evaluation failed."));
    } finally {
      setEvaluationBusy(false);
    }
  };

  const togglePlayPause = () => {
    if (!currentTrack) return;
    if (!audioRef.current || !currentTrack?.preview_url) {
      if (currentTrack.external_url) {
        window.open(currentTrack.external_url, "_blank", "noopener,noreferrer");
      }
      return;
    }
    if (audioRef.current.paused) {
      audioRef.current.play().then(() => {
        setUiError("");
        setIsPlaying(true);
        setSessionStats((prev) => ({ ...prev, songsPlayed: prev.songsPlayed + 1 }));
        logMetricsEvent({
          user_id: userIdentifier,
          category: "playback",
          action: "track_start",
          track_id: currentTrack?.id || "",
          track_name: currentTrack?.name || "",
          artist: currentTrack?.artist || "",
          emotion: activeLock?.label || "neutral",
          source: sourceLabel,
        }).catch(() => {});
      }).catch(() => {
        setIsPlaying(false);
        setUiError("Track loaded, but playback was blocked. Press play again once.");
      });
    } else {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const playPrev = () => {
    if (!playlist.length) return;
    setCurrentTrackIndex((prev) => (prev <= 0 ? playlist.length - 1 : prev - 1));
  };

  const playNext = () => {
    if (!playlist.length) return;
    setCurrentTrackIndex((prev) => (prev >= playlist.length - 1 ? 0 : prev + 1));
  };

  const moveLanguage = (from, to) => {
    if (to < 0 || to >= languages.length) return;
    const copy = [...languages];
    const [item] = copy.splice(from, 1);
    copy.splice(to, 0, item);
    setLanguages(copy);
  };

  const updateProfileField = (field, value) => {
    setProfileForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleProfileReset = () => {
    if (!authUser) {
      setProfileForm(DEFAULT_PROFILE_FORM);
      return;
    }
    setProfileForm({
      username: authUser.username || "",
      full_name: authUser.full_name || "",
      phone: authUser.phone || "",
      location: authUser.location || "",
      bio: authUser.bio || "",
    });
    setProfileNoticeTone("info");
    setProfileNotice("Profile form reset.");
  };

  const handleProfileSave = async () => {
    if (!authUser) return;
    setProfileNotice("");
    setProfileNoticeTone("info");
    setProfileBusy(true);
    try {
      const nextUsername = profileForm.username.trim().toLowerCase() || authUser.username;
      const payload = {
        username: nextUsername,
        full_name: profileForm.full_name.trim(),
        phone: profileForm.phone.trim(),
        location: profileForm.location.trim(),
        bio: profileForm.bio.trim(),
      };
      const updated = normalizeUserIdentity(await updateMyProfile(payload));
      setAuthUser(updated);
      setProfileNoticeTone("success");
      setProfileNotice("Profile updated successfully.");
      setUiError("");
    } catch (error) {
      const message = extractApiError(error, "Could not update profile right now.");
      setProfileNoticeTone("error");
      setProfileNotice(message);
      setUiError(message);
    } finally {
      setProfileBusy(false);
    }
  };

  const handleResetSettingsDefaults = () => {
    setPlaybackMode("autoplay");
    setMusicSource("Spotify");
    setLanguages([...LANGUAGE_OPTIONS]);
    setIsLightMode(false);
    setProfileNoticeTone("info");
    setProfileNotice("Settings reset to defaults.");
  };

  useEffect(() => {
    if (!authUser) {
      setProfileForm(DEFAULT_PROFILE_FORM);
      setProfileNoticeTone("info");
      setProfileNotice("");
      return;
    }
    setProfileForm({
      username: authUser.username || "",
      full_name: authUser.full_name || "",
      phone: authUser.phone || "",
      location: authUser.location || "",
      bio: authUser.bio || "",
    });
  }, [authUser]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed?.playbackMode === "autoplay" || parsed?.playbackMode === "manual") {
        setPlaybackMode(parsed.playbackMode);
      }
      if (["Spotify", "Local", "Import"].includes(parsed?.musicSource)) {
        setMusicSource(parsed.musicSource);
        setSourceLabel(parsed.musicSource);
      }
      if (Array.isArray(parsed?.languages) && parsed.languages.length === LANGUAGE_OPTIONS.length) {
        setLanguages(parsed.languages);
      }
    } catch {
      // Ignore invalid persisted settings and fall back to defaults.
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      SETTINGS_STORAGE_KEY,
      JSON.stringify({
        playbackMode,
        musicSource,
        languages,
      })
    );
  }, [playbackMode, musicSource, languages]);

  useEffect(() => {
    if (!authUser) {
      setPreferencesReady(false);
      return;
    }

    let cancelled = false;
    getMyPreferences()
      .then((prefs) => {
        if (cancelled || !prefs) return;
        if (prefs.playback_mode === "autoplay" || prefs.playback_mode === "manual") {
          setPlaybackMode(prefs.playback_mode);
        }
        if (["Spotify", "Local", "Import"].includes(prefs.music_source)) {
          setMusicSource(prefs.music_source);
          setSourceLabel(prefs.music_source);
        }
        if (Array.isArray(prefs.languages) && prefs.languages.length === LANGUAGE_OPTIONS.length) {
          setLanguages(prefs.languages);
        }
        if (prefs.theme === "light") setIsLightMode(true);
        if (prefs.theme === "dark") setIsLightMode(false);
      })
      .catch(() => {
        // Keep local settings when backend preferences are unavailable.
      })
      .finally(() => {
        if (!cancelled) setPreferencesReady(true);
      });

    return () => {
      cancelled = true;
    };
  }, [authUser]);

  useEffect(() => {
    if (!authUser || !preferencesReady) return;
    saveMyPreferences({
      playback_mode: playbackMode,
      music_source: musicSource,
      languages,
      theme: isLightMode ? "light" : "dark",
    }).catch(() => {
      // Local storage remains the fallback if preference sync fails.
    });
  }, [authUser, preferencesReady, playbackMode, musicSource, languages, isLightMode]);

  useEffect(() => {
    if (!authUser) return;

    let cancelled = false;
    const loadMetrics = async () => {
      try {
        const data = await getMetricsOverview(userIdentifier);
        if (cancelled) return;
        setMetricsOverview(data);
      } catch {
        // Ignore transient metrics polling failures in the UI.
      }
    };

    loadMetrics();
    const timer = setInterval(loadMetrics, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [authUser, userIdentifier]);

  useEffect(() => {
    if (!rawRecommendations.length) return;
    const shapedTracks = shapePlaylistWithSettings(rawRecommendations, musicSource, languages);
    setPlaylist(shapedTracks);
    setSourceLabel(musicSource);
    setCurrentTrackIndex((prev) => {
      const previousTrackId = prev >= 0 ? playlist[prev]?.id || "" : "";
      return selectPreferredTrackIndex(shapedTracks, previousTrackId);
    });
  }, [musicSource, languages, rawRecommendations]);

  useEffect(() => {
    if (!currentTrack?.preview_url || !audioRef.current) return;
    audioRef.current.load();
    if (playbackMode === "autoplay") {
      audioRef.current.play().then(() => {
        setUiError("");
        setIsPlaying(true);
        setSessionStats((prev) => ({ ...prev, songsPlayed: prev.songsPlayed + 1 }));
        logMetricsEvent({
          user_id: userIdentifier,
          category: "playback",
          action: "track_start",
          track_id: currentTrack?.id || "",
          track_name: currentTrack?.name || "",
          artist: currentTrack?.artist || "",
          emotion: activeLock?.label || "neutral",
          source: sourceLabel,
        }).catch(() => {});
      }).catch(() => {
        setIsPlaying(false);
        setUiError((prev) => prev || "Track loaded. Press play once to enable audio in this browser.");
      });
    } else {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setSeek(0);
      setIsPlaying(false);
    }
  }, [currentTrack?.id, currentTrack?.preview_url, playbackMode]);

  useEffect(() => {
    if (faceMode !== "upload") return;
    const input = imageInputRef.current;
    if (!input) return;
    if (!input.files?.[0] && !facePreview) {
      setTimeout(() => input.click(), 0);
    }
    const onChange = () => {
      const file = input.files?.[0];
      if (!file) return;
      if (!String(file.type || "").startsWith("image/")) {
        setUiError("Invalid file type. Please choose an image file.");
        return;
      }
      const url = URL.createObjectURL(file);
      setFacePreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      setUiError("");
    };
    input.addEventListener("change", onChange);
    return () => input.removeEventListener("change", onChange);
  }, [faceMode]);

  useEffect(() => {
    return () => {
      if (facePreview) URL.revokeObjectURL(facePreview);
      stopSpeechCapture();
    };
  }, [facePreview]);

  useEffect(() => {
    document.documentElement.classList.toggle("theme-light", isLightMode);
  }, [isLightMode]);

  useEffect(() => {
    const handlePopState = () => {
      const routePage = getPageFromPath(window.location.pathname);
      if (routePage !== "landing") {
        setActivePage(routePage);
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    let active = true;
    getCurrentUser()
      .then((user) => {
        if (!active) return;
        setAuthUser(normalizeUserIdentity(user) || null);
        const routePage = getPageFromPath(window.location.pathname);
        if (user) {
          if (routePage === "landing") {
            navigateToPage("home", { replace: true });
          } else {
            setActivePage(routePage);
          }
        } else {
          navigateToLanding({ replace: true });
        }
      })
      .catch(() => {
        if (!active) return;
        setAuthUser(null);
        navigateToLanding({ replace: true });
      })
      .finally(() => {
        if (!active) return;
        setAuthReady(true);
      });
    return () => {
      active = false;
    };
  }, []);

  const openAuthModal = (mode) => {
    setAuthMode(mode);
    setAuthError("");
    setShowAuthModal(true);
  };

  const closeAuthModal = () => {
    if (authBusy) return;
    setShowAuthModal(false);
    setAuthError("");
  };

  const handleAuthSubmit = async (event) => {
    event.preventDefault();
    setAuthError("");
    setAuthBusy(true);
    try {
      let authResponse = null;
      if (authMode === "signup") {
        if (!authForm.username.trim() || !authForm.email.trim() || !authForm.password.trim()) {
          throw new Error("Please fill username, email, and password.");
        }
        authResponse = await signupWithDetails({
          username: authForm.username.trim(),
          email: authForm.email.trim(),
          password: authForm.password,
        });
      } else {
        if (!authForm.identifier.trim() || !authForm.password.trim()) {
          throw new Error("Please fill email/username and password.");
        }
        authResponse = await signinWithIdentifier(authForm.identifier.trim(), authForm.password);
      }
      const user = normalizeUserIdentity((await getCurrentUser()) || authResponse || { username: authForm.username || "User" });
      setAuthUser(user);
      navigateToPage("home");
      setShowAuthModal(false);
      setAuthForm({ username: "", email: "", identifier: "", password: "" });
    } catch (error) {
      setAuthError(extractApiError(error, authMode === "signup" ? "Signup failed." : "Login failed."));
    } finally {
      setAuthBusy(false);
    }
  };

  const handleLogout = () => {
    signout();
    setAuthUser(null);
    setShowProfileMenu(false);
    setActivePage("home");
    navigateToLanding();
  };
  const handleGoogleAuth = async () => {
    setAuthError("");
    if (!GOOGLE_CLIENT_ID) {
      setAuthError("Add your Google Web Client ID in frontend/.env as VITE_GOOGLE_CLIENT_ID=... then restart the frontend.");
      return;
    }
    setAuthBusy(true);

    const finishWithToken = async (accessToken) => {
      try {
        const authResponse = await signinWithGoogleAccessToken(accessToken);
        const user = normalizeUserIdentity((await getCurrentUser()) || authResponse || { username: "User" });
        setAuthUser(user);
        navigateToPage("home");
        setShowAuthModal(false);
        setAuthForm({ username: "", email: "", identifier: "", password: "" });
      } catch (error) {
        setAuthError(extractApiError(error, "Google sign-in failed."));
      } finally {
        setAuthBusy(false);
      }
    };

    try {
      if (!window.google?.accounts?.oauth2) {
        await new Promise((resolve, reject) => {
          const existing = document.querySelector('script[data-google-identity="1"]');
          if (existing && window.google?.accounts?.oauth2) {
            resolve();
            return;
          }
          const script = document.createElement("script");
          script.src = "https://accounts.google.com/gsi/client";
          script.async = true;
          script.defer = true;
          script.dataset.googleIdentity = "1";
          script.onload = () => resolve();
          script.onerror = () => reject(new Error("Failed to load Google Sign-In SDK"));
          document.head.appendChild(script);
        });
      }

      if (!googleTokenClientRef.current) {
        googleTokenClientRef.current = window.google.accounts.oauth2.initTokenClient({
          client_id: GOOGLE_CLIENT_ID,
          scope: "openid email profile",
          callback: (tokenResponse) => {
            const accessToken = tokenResponse?.access_token || "";
            if (!accessToken) {
              setAuthBusy(false);
              setAuthError("Google authorization was cancelled.");
              return;
            }
            finishWithToken(accessToken);
          },
        });
      }

      googleTokenClientRef.current.requestAccessToken({ prompt: "consent" });
    } catch (error) {
      setAuthBusy(false);
      setAuthError(extractApiError(error, "Google sign-in is unavailable right now."));
    }
  };

  const renderFaceWorkspace = () => (
    <section className="single-panel card-glass face-workspace-panel">
      <div className="panel-title-row">
        <h3>Face Emotion</h3>
        <span className={`status-inline ${statusTone(faceStatus)}`}>
          <span className={`status-icon ${statusTone(faceStatus)}`} aria-hidden />
          <span>{statusLabel(faceStatus)}</span>
        </span>
      </div>
      <div className="workspace-grid">
        <div className="workspace-main">
          <div className="workspace-top-actions">
            <button
              className={`face-mode-btn live-btn ${faceMode === "live" ? "active" : ""} ${isCamOn ? "cam-on" : "cam-off"}`}
              onClick={() => setFaceMode("live")}
            >
              Live
            </button>
            <button
              className={`face-mode-btn upload-btn ${faceMode === "upload" ? "active" : ""}`}
              onClick={() => setFaceMode("upload")}
            >
              Upload
            </button>
          </div>
          {uiError ? <p className="inline-panel-error">{uiError}</p> : null}
          <div className="stage face-stage">
            {faceMode === "live" ? (
              isCamOn ? (
                <Webcam ref={webcamRef} screenshotFormat="image/jpeg" className="camera-feed" />
              ) : (
                <div className="placeholder-msg">Camera is stopped.</div>
              )
            ) : (
              <div className="upload-zone">
                <input ref={imageInputRef} type="file" accept="image/*" className="hidden-upload-input" />
                {facePreview ? (
                  <img src={facePreview} alt="Uploaded preview" className="preview-image" />
                ) : (
                  <p>Select an image and submit to analyze.</p>
                )}
              </div>
            )}
          </div>
          <div className="workspace-bottom-actions">
            {faceMode === "live" ? (
              <>
                <button className="primary-btn" onClick={() => setIsCamOn((prev) => !prev)}>
                  {isCamOn ? "Stop Cam" : "Start Cam"}
                </button>
                <button className="primary-btn" onClick={handleFaceDetect}>Detect / Analyze</button>
                <button className="ghost-btn" onClick={handleRetakeFace}>Retake</button>
              </>
            ) : (
              <>
                <button className="primary-btn" onClick={handleFaceDetect}>Submit Picture</button>
                <button className="ghost-btn" onClick={handleReuploadPicture}>Reupload Picture</button>
              </>
            )}
          </div>
        </div>
        <div className="emotion-side-card face-emotion-side-card">
          <h4>Emotion</h4>
          <img className="result-sticker" src={activeSticker} alt={activeLock?.label || "idle"} />
          <p className="result-label">{activeLock ? activeLock.label : "No lock yet"}</p>
          <p className="confidence-text">Confidence Rate</p>
          <div className="confidence-track"><span style={{ width: `${Math.round((activeLock?.confidence || 0) * 100)}%` }} /></div>
          <p className="confidence-text">{Math.round((activeLock?.confidence || 0) * 100)}%</p>
          {activeLock ? (
            <div className="emotion-feedback-box">
              <p className="confidence-text">Was this detected correctly?</p>
              <div className="emotion-feedback-actions">
                <button
                  type="button"
                  className={feedbackState.face === "correct" ? "emotion-feedback-btn active positive" : "emotion-feedback-btn"}
                  onClick={() => handleEmotionFeedback("face", true)}
                  aria-label="Face emotion correct"
                  title="Correct"
                >
                  👍
                </button>
                <button
                  type="button"
                  className={feedbackState.face === "incorrect" ? "emotion-feedback-btn active negative" : "emotion-feedback-btn"}
                  onClick={() => handleEmotionFeedback("face", false)}
                  aria-label="Face emotion incorrect"
                  title="Incorrect"
                >
                  👎
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <div className="emotion-chip-row">
        {ALL_EMOTIONS.map((emotion) => (
          <span key={emotion} className={activeLock?.label === emotion ? "emotion-chip active" : "emotion-chip"}>{EMOJI_BY_EMOTION[emotion]} {emotion}</span>
        ))}
      </div>
    </section>
  );

  const renderVoiceWorkspace = () => (
    <section className="single-panel card-glass voice-workspace-panel">
      <div className="panel-title-row">
        <h3>Voice Emotion</h3>
        <span className={`status-inline ${statusTone(speechStatus)}`}>
          <span className={`status-icon ${statusTone(speechStatus)}`} aria-hidden />
          <span>{statusLabel(speechStatus)}</span>
        </span>
      </div>
      <div className="workspace-grid">
        <div className="workspace-main">
          {uiError ? <p className="inline-panel-error">{uiError}</p> : null}
          <div className="stage voice-stage">
            <canvas ref={waveformCanvasRef} className="waveform-canvas" />
            {speechStatus === "idle" ? <div className="placeholder-overlay">Start mic to visualize your voice frequency.</div> : null}
          </div>
          <div className="voice-transcript-card">
            <p className="voice-transcript-title">Live Subtitles</p>
            <p className="voice-transcript-text">
              {speechTranscript || speechInterimTranscript || (speechRecognitionSupported
                ? "What you say will appear here while recording."
                : "Subtitles are unavailable in this browser, but voice emotion detection still works.")}
            </p>
          </div>
          <div className="workspace-bottom-actions">
            <button className="primary-btn" onClick={handleSpeechStart}>Start Mic</button>
            <button className="primary-btn" onClick={handleSpeechStop}>Stop Mic</button>
            <button className="ghost-btn" onClick={handleRetakeSpeech}>Retake Speech</button>
          </div>
        </div>
        <div className="emotion-side-card voice-emotion-side-card">
          <h4>Emotion</h4>
          <img className="result-sticker" src={activeSticker} alt={activeLock?.label || "idle"} />
          <p className="result-label">{activeLock ? activeLock.label : "No lock yet"}</p>
          <p className="confidence-text">Confidence Rate</p>
          <div className="confidence-track"><span style={{ width: `${Math.round((activeLock?.confidence || 0) * 100)}%` }} /></div>
          <p className="confidence-text">{Math.round((activeLock?.confidence || 0) * 100)}%</p>
          {activeLock?.sourceWeights ? (
            <div className="fusion-bars">
              <p className="confidence-text">Combined Result</p>
              <div className="fusion-bar-group">
                <div className="fusion-bar-label-row">
                  <span className="confidence-text">Tone</span>
                  <span className="confidence-text">{Math.round((activeLock.sourceWeights.voice || 0) * 100)}%</span>
                </div>
                <div className="confidence-track fusion-track tone">
                  <span style={{ width: `${Math.round((activeLock.sourceWeights.voice || 0) * 100)}%` }} />
                </div>
              </div>
              <div className="fusion-bar-group">
                <div className="fusion-bar-label-row">
                  <span className="confidence-text">Words</span>
                  <span className="confidence-text">{Math.round((activeLock.sourceWeights.text || 0) * 100)}%</span>
                </div>
                <div className="confidence-track fusion-track words">
                  <span style={{ width: `${Math.round((activeLock.sourceWeights.text || 0) * 100)}%` }} />
                </div>
              </div>
            </div>
          ) : null}
          {activeLock ? (
            <div className="emotion-feedback-box">
              <p className="confidence-text">Was this detected correctly?</p>
              <div className="emotion-feedback-actions">
                <button
                  type="button"
                  className={feedbackState.speech === "correct" ? "emotion-feedback-btn active positive" : "emotion-feedback-btn"}
                  onClick={() => handleEmotionFeedback("voice", true)}
                  aria-label="Voice emotion correct"
                  title="Correct"
                >
                  👍
                </button>
                <button
                  type="button"
                  className={feedbackState.speech === "incorrect" ? "emotion-feedback-btn active negative" : "emotion-feedback-btn"}
                  onClick={() => handleEmotionFeedback("voice", false)}
                  aria-label="Voice emotion incorrect"
                  title="Incorrect"
                >
                  👎
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <div className="emotion-chip-row">
        {ALL_EMOTIONS.map((emotion) => (
          <span key={emotion} className={activeLock?.label === emotion ? "emotion-chip active" : "emotion-chip"}>{EMOJI_BY_EMOTION[emotion]} {emotion}</span>
        ))}
      </div>
    </section>
  );

  const renderHome = () => (
    <div className="home-single-layout">
      <div className="home-panel-switch">
        <button className={activeDetectionPanel === "face" ? "active" : ""} onClick={() => setActiveDetectionPanel("face")}>Facial Panel</button>
        <button className={activeDetectionPanel === "voice" ? "active" : ""} onClick={() => setActiveDetectionPanel("voice")}>Voice Panel</button>
      </div>
      {activeDetectionPanel === "face" ? renderFaceWorkspace() : renderVoiceWorkspace()}
    </div>
  );

  const renderMetrics = () => {
    const evaluation = metricsOverview.evaluation || {};
    const overall = metricsOverview.overall_model || {};
    const live = metricsOverview.live || {};
    const facial = evaluation.by_input_type?.face || metricsOverview.facial_model || {};
    const speech = evaluation.by_input_type?.voice || metricsOverview.speech_model || {};
    const timeline = live.timeline || timelinePoints;
    const topEmotionLabel = live.top_emotions?.[0]?.emotion || "None yet";
    const confusionMatrix = facial.confusion_matrix || speech.confusion_matrix || [];
    const confusionLabels = facial.labels || speech.labels || ALL_EMOTIONS;
    const maxConfusionValue = confusionMatrix.flat().reduce((max, value) => Math.max(max, value), 0) || 1;
    const distribution = evaluation.emotion_distribution?.length
      ? evaluation.emotion_distribution
      : live.top_emotions || [];
    const palette = ["#8d5cff", "#59bfff", "#56d27b", "#ffb24d", "#ff688a", "#ffe26d", "#b6bccd"];
    const modelBarData = {
      labels: ["Accuracy", "Precision", "Recall", "F1 Score"],
      datasets: [
        {
          label: "Face",
          data: [facial.accuracy || 0, facial.precision || 0, facial.recall || 0, facial.f1_score || 0],
          backgroundColor: "rgba(141, 92, 255, 0.78)",
          borderRadius: 8,
        },
        {
          label: "Voice",
          data: [speech.accuracy || 0, speech.precision || 0, speech.recall || 0, speech.f1_score || 0],
          backgroundColor: "rgba(89, 191, 255, 0.72)",
          borderRadius: 8,
        },
        {
          label: "Overall",
          data: [overall.accuracy || 0, overall.precision || 0, overall.recall || 0, overall.f1_score || 0],
          backgroundColor: "rgba(255, 255, 255, 0.76)",
          borderRadius: 8,
        },
      ],
    };
    const modelBarOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "rgba(255,255,255,0.78)",
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "rgba(255,255,255,0.7)" },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            color: "rgba(255,255,255,0.7)",
            callback: (value) => `${Math.round(Number(value) * 100)}%`,
          },
          grid: {
            color: "rgba(255,255,255,0.08)",
          },
        },
      },
    };
    const distributionChartData = {
      labels: distribution.map((item) => item.emotion || item.label),
      datasets: [
        {
          data: distribution.map((item) => item.count || item.value || 0),
          backgroundColor: palette,
          borderWidth: 0,
        },
      ],
    };
    const distributionChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      cutout: "62%",
    };

    return (
      <div className="metrics-stack">
        <div className="metrics-toolbar panel-card">
          <div>
            <h3>Model Evaluation</h3>
            <p className="metrics-subtext">Run evaluation on FER2013 and RAVDESS, store prediction logs, and compare model quality against live usage analytics.</p>
          </div>
          <div className="metrics-toolbar-actions">
            <button className="ghost-btn" onClick={() => handleRunEvaluation("face")} disabled={evaluationBusy}>
              {evaluationBusy ? "Running..." : "Evaluate Face"}
            </button>
            <button className="ghost-btn" onClick={() => handleRunEvaluation("voice")} disabled={evaluationBusy}>
              {evaluationBusy ? "Running..." : "Evaluate Voice"}
            </button>
            <button className="primary-btn" onClick={() => handleRunEvaluation("all")} disabled={evaluationBusy}>
              {evaluationBusy ? "Running Evaluation..." : "Run Full Evaluation"}
            </button>
          </div>
        </div>
        <div className="metrics-grid">
          <div className="stat-tile"><p>Accuracy</p><h4>{formatPercent(overall.accuracy)}</h4></div>
          <div className="stat-tile"><p>Precision</p><h4>{formatPercent(overall.precision)}</h4></div>
          <div className="stat-tile"><p>Recall</p><h4>{formatPercent(overall.recall)}</h4></div>
          <div className="stat-tile"><p>F1-Score</p><h4>{formatPercent(overall.f1_score)}</h4></div>
          <div className="chart-card">
            <h3>Confusion Matrix</h3>
            <div className="confusion-matrix-grid">
              {confusionMatrix.length ? (
                <>
                  <div className="confusion-matrix-header">
                    <span className="confusion-matrix-label">Actual \\ Predicted</span>
                    <div className="confusion-matrix-cells">
                      {confusionLabels.map((label) => (
                        <span className="confusion-matrix-axis" key={`head-${label}`}>{label}</span>
                      ))}
                    </div>
                  </div>
                  {confusionMatrix.map((row, rowIndex) => (
                    <div className="confusion-matrix-row" key={`row-${confusionLabels[rowIndex] || rowIndex}`}>
                      <span className="confusion-matrix-label">{confusionLabels[rowIndex] || `Row ${rowIndex + 1}`}</span>
                      <div className="confusion-matrix-cells">
                        {row.map((value, cellIndex) => (
                          <span
                            className="confusion-matrix-cell"
                            key={`cell-${rowIndex}-${cellIndex}`}
                            style={{ background: `rgba(141, 92, 255, ${0.08 + ((value || 0) / maxConfusionValue) * 0.7})` }}
                          >
                            {value}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </>
              ) : <p className="metrics-empty">Confusion matrix will appear once model metrics are available.</p>}
            </div>
          </div>
          <div className="chart-card">
            <h3>Model Accuracy Comparison</h3>
            <div className="chart-shell">
              <Bar data={modelBarData} options={modelBarOptions} />
            </div>
          </div>
          <div className="chart-card">
            <h3>Emotion Distribution</h3>
            {distribution.length ? (
              <>
                <div className="chart-shell chart-shell-donut">
                  <Doughnut data={distributionChartData} options={distributionChartOptions} />
                </div>
                <div className="distribution-legend">
                  {distribution.map((item, index) => (
                    <span key={`emotion-dist-${item.emotion || item.label}`}>
                      <i style={{ backgroundColor: palette[index % palette.length] }} />
                      {item.emotion || item.label} ({item.count || item.value || 0})
                    </span>
                  ))}
                </div>
              </>
            ) : <p className="metrics-empty">Emotion distribution will appear after evaluation data or live sessions are available.</p>}
          </div>
          <div className="chart-card">
            <h3>Live Confidence Timeline</h3>
            <EmotionTimeline points={timeline.map((p) => ({ timestamp: p.timestamp, confidence: p.confidence }))} />
          </div>
          <div className="chart-card">
            <h3>Live Session Stats</h3>
            <div className="metrics-session-list">
              <p><span>Emotions detected</span><strong>{live.emotions_detected || 0}</strong></p>
              <p><span>Songs played</span><strong>{live.songs_played || 0}</strong></p>
              <p><span>Retakes</span><strong>{live.retakes || 0}</strong></p>
              <p><span>Recommendations</span><strong>{live.recommendations_generated || 0}</strong></p>
              <p><span>Top emotion</span><strong>{topEmotionLabel}</strong></p>
              <p><span>Dataset samples</span><strong>{(facial.sample_count || 0) + (speech.sample_count || 0)}</strong></p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderProfile = () => (
    <div className="settings-stack">
      <section className="panel-card settings-card settings-profile-card">
        <div className="settings-header-row">
          <div>
            <h3>Profile Details</h3>
            <p className="settings-subtext">Edit your account information. Changes are saved to your user database profile.</p>
          </div>
          <span className="settings-tag">Account</span>
        </div>
        <div className="settings-form-grid">
          <label className="settings-field">
            <span>Username</span>
            <input
              type="text"
              value={profileForm.username}
              onChange={(e) => updateProfileField("username", e.target.value)}
              placeholder="username"
              autoComplete="username"
            />
          </label>
          <label className="settings-field">
            <span>Email</span>
            <input type="email" value={authUser?.email || ""} disabled />
          </label>
          <label className="settings-field">
            <span>Full Name</span>
            <input
              type="text"
              value={profileForm.full_name}
              onChange={(e) => updateProfileField("full_name", e.target.value)}
              placeholder="Your full name"
            />
          </label>
          <label className="settings-field">
            <span>Phone</span>
            <input
              type="text"
              value={profileForm.phone}
              onChange={(e) => updateProfileField("phone", e.target.value)}
              placeholder="Phone number"
            />
          </label>
          <label className="settings-field settings-field-wide">
            <span>Location</span>
            <input
              type="text"
              value={profileForm.location}
              onChange={(e) => updateProfileField("location", e.target.value)}
              placeholder="City, Country"
            />
          </label>
          <label className="settings-field settings-field-wide">
            <span>Bio</span>
            <textarea
              rows={3}
              value={profileForm.bio}
              onChange={(e) => updateProfileField("bio", e.target.value)}
              placeholder="Short bio for your SONA profile"
            />
          </label>
        </div>
        <div className="settings-actions-row">
          <button className="settings-btn primary" onClick={handleProfileSave} disabled={profileBusy}>
            {profileBusy ? "Saving..." : "Save Profile"}
          </button>
          <button className="settings-btn secondary" onClick={handleProfileReset} disabled={profileBusy}>
            Reset Form
          </button>
          {profileNotice ? (
            <span className={`settings-notice ${profileNoticeTone}`}>
              {profileNotice}
            </span>
          ) : null}
        </div>
      </section>

      <section className="panel-card settings-card">
        <div className="settings-header-row">
          <div>
            <h3>Playback Mode</h3>
            <p className="settings-subtext">Choose how tracks should start after a new emotion lock.</p>
          </div>
          <span className="settings-tag">{playbackMode === "autoplay" ? "Auto" : "Manual"}</span>
        </div>
        <div className="settings-segmented">
          <button className={playbackMode === "autoplay" ? "active" : ""} onClick={() => setPlaybackMode("autoplay")}>Autoplay</button>
          <button className={playbackMode === "manual" ? "active" : ""} onClick={() => setPlaybackMode("manual")}>Manual</button>
        </div>
      </section>

      <section className="panel-card settings-card">
        <div className="settings-header-row">
          <div>
            <h3>Music Source</h3>
            <p className="settings-subtext">Switch source preference for generated playlists.</p>
          </div>
          <span className="settings-tag">{musicSource}</span>
        </div>
        <div className="settings-source-grid">
          {SOURCE_OPTIONS.map((source) => (
            <button
              key={source}
              className={musicSource === source ? "settings-source-card active" : "settings-source-card"}
              onClick={() => setMusicSource(source)}
            >
              <strong>{source}</strong>
              <small>{SOURCE_DESCRIPTIONS[source]}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="panel-card settings-card">
        <div className="settings-header-row">
          <div>
            <h3>Language Priority</h3>
            <p className="settings-subtext">Higher-ranked languages get stronger recommendation weight.</p>
          </div>
          <span className="settings-tag">Ranked</span>
        </div>
        <div className="settings-language-list">
          {languages.map((lang, idx) => (
            <div className="settings-language-item" key={lang}>
              <div className="settings-language-left">
                <span className="settings-rank">{String(idx + 1).padStart(2, "0")}</span>
                <span className="settings-language-name">{lang}</span>
              </div>
              <div className="settings-language-actions">
                <button
                  className="settings-icon-btn"
                  onClick={() => moveLanguage(idx, idx - 1)}
                  aria-label={`Move ${lang} up`}
                  disabled={idx === 0}
                >
                  <ArrowUpIcon />
                </button>
                <button
                  className="settings-icon-btn"
                  onClick={() => moveLanguage(idx, idx + 1)}
                  aria-label={`Move ${lang} down`}
                  disabled={idx === languages.length - 1}
                >
                  <ArrowDownIcon />
                </button>
              </div>
            </div>
          ))}
        </div>
        <div className="settings-language-footer">
          <p>{languages.join(" → ")}</p>
          <button className="settings-btn tertiary" onClick={() => setLanguages([...LANGUAGE_OPTIONS])}>
            Restore Default Order
          </button>
        </div>
      </section>

      <section className="panel-card settings-card settings-quick-card">
        <div className="settings-header-row">
          <div>
            <h3>Quick Actions</h3>
            <p className="settings-subtext">Shortcuts for theme and preferences reset.</p>
          </div>
          <span className="settings-tag">Tools</span>
        </div>
        <div className="settings-actions-row">
          <button className="settings-btn secondary" onClick={() => setIsLightMode((prev) => !prev)}>
            Switch to {isLightMode ? "Dark" : "Light"} Theme
          </button>
          <button className="settings-btn tertiary" onClick={handleResetSettingsDefaults}>
            Reset All Preferences
          </button>
        </div>
      </section>
    </div>
  );

  const renderContacts = () => (
    <div className="contacts-grid">
      <section className="panel-card"><h3>Support</h3><p>Product Support: support@sona.app</p><p>Priority Channel: enterprise@sona.app</p><p>Response Window: 9:00 AM - 9:00 PM IST</p></section>
      <section className="panel-card"><h3>Team Contacts</h3><p>Design Lead - ui@sona.app</p><p>Product Team - product@sona.app</p><p>Operations - ops@sona.app</p></section>
    </div>
  );

  const renderAuthModal = () => (
    <div className="auth-modal-backdrop" onClick={closeAuthModal} role="presentation">
      <div className="auth-modal-card" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <h3>{authMode === "signup" ? "Create your account" : "Welcome back"}</h3>
        <p>{authMode === "signup" ? "Sign up to start using SONA." : "Login to continue to SONA."}</p>
        <form className="auth-form" onSubmit={handleAuthSubmit}>
          {authMode === "signup" ? (
            <input
              type="text"
              placeholder="Username"
              value={authForm.username}
              onChange={(e) => setAuthForm((prev) => ({ ...prev, username: e.target.value }))}
            />
          ) : null}
          {authMode === "signup" ? (
            <input
              type="email"
              placeholder="Email"
              value={authForm.email}
              onChange={(e) => setAuthForm((prev) => ({ ...prev, email: e.target.value }))}
            />
          ) : (
            <input
              type="text"
              placeholder="Email or Username"
              value={authForm.identifier}
              onChange={(e) => setAuthForm((prev) => ({ ...prev, identifier: e.target.value }))}
            />
          )}
          <input
            type="password"
            placeholder="Password"
            value={authForm.password}
            onChange={(e) => setAuthForm((prev) => ({ ...prev, password: e.target.value }))}
          />
          {authError ? <p className="auth-error">{authError}</p> : null}
          <div className="auth-actions">
            <button type="button" className="auth-link-btn" onClick={handleGoogleAuth} disabled={authBusy}>
              Continue with Google
            </button>
            <button type="submit" className="auth-primary-btn" disabled={authBusy}>
              {authBusy ? "Please wait..." : authMode === "signup" ? "Sign Up" : "Login"}
            </button>
            <button type="button" className="auth-link-btn" onClick={() => setAuthMode((prev) => (prev === "signup" ? "signin" : "signup"))}>
              {authMode === "signup" ? "Have an account? Login" : "Need an account? Sign Up"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );

  const renderLanding = () => (
    <div className="landing-shell">
      <header className="landing-topbar">
        <button type="button" className="brand brand-button" onClick={handleBrandClick} aria-label="Go to home page">
          <img className="brand-logo" src="/assets/logo/sona-logo.jpeg" alt="SONA logo" />
          <span className="brand-text">Sona</span>
        </button>
        <nav className="landing-center-nav" aria-label="Landing navigation">
          <button type="button" className="landing-nav-item">Features</button>
          <button type="button" className="landing-nav-item">Detection</button>
          <button type="button" className="landing-nav-item">Metrics</button>
          <button type="button" className="landing-nav-item">Support</button>
          <button type="button" className="landing-nav-item">FAQ</button>
        </nav>
        <div className="landing-top-actions">
          <button
            type="button"
            className={`theme-toggle nav-theme-toggle ${isLightMode ? "active" : ""}`}
            onClick={() => setIsLightMode((prev) => !prev)}
            title="Toggle theme"
            aria-label="Toggle theme"
          >
            <ThemeIcon isLightMode={isLightMode} />
          </button>
          <button className="landing-link-btn" onClick={() => openAuthModal("signin")}>Login</button>
          <button className="landing-primary-btn" onClick={() => openAuthModal("signup")}>Sign Up</button>
        </div>
      </header>
      <section className="landing-hero">
        <p className="landing-kicker">Welcome to SONA, <em>Sentimental oriented neural audio</em></p>
        <h1>
          <span>Feel It.</span>
          <span className="accent">Hear It.</span>
        </h1>
        <p className="landing-subtitle">
          Discover music through facial and voice emotion detection with a cinematic SONA interface tuned for mood, playback, and live recommendation flow.
        </p>
        <div className="landing-cta-row">
          <button className="landing-primary-btn big" onClick={() => openAuthModal("signup")}>Get Started</button>
          <button className="landing-ghost-btn" onClick={() => openAuthModal("signin")}>I already have an account</button>
        </div>
        <div className="landing-proof-row">
          <span className="landing-proof-pill">Real-time face analysis</span>
          <span className="landing-proof-pill">Voice frequency detection</span>
          <span className="landing-proof-pill">Emotion-locked playlists</span>
        </div>
      </section>
      <section className="landing-marquee-band" aria-label="SONA highlights">
        <div className="landing-marquee-stack">
          {LANDING_MARQUEE_ROWS.map((row, rowIndex) => (
            <div className={`landing-marquee row-${rowIndex + 1}`} key={`row-${rowIndex}`}>
              <div className={`landing-marquee-track ${rowIndex % 2 === 1 ? "reverse" : ""}`}>
                {[...row, ...row].map((item, index) => (
                  <span
                    className={`landing-marquee-pill theme-${LANDING_MARQUEE_THEMES[(rowIndex * 6 + index) % LANDING_MARQUEE_THEMES.length]}`}
                    key={`${rowIndex}-${item}-${index}`}
                  >
                    <span className="landing-marquee-dots" aria-hidden>
                      <i />
                      <i />
                      <i />
                    </span>
                    <span>{item}</span>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
      <section className="landing-section landing-steps-section">
        <div className="landing-section-head">
          <p className="landing-section-kicker">Flow</p>
          <h2>Three Steps to Perfection</h2>
          <p>SONA keeps the loop simple: detect, lock, then listen.</p>
        </div>
        <div className="landing-steps-grid">
          {LANDING_STEPS.map((step) => (
            <article className="landing-step-card" key={step.index}>
              <span className="landing-step-index">{step.index}</span>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </article>
          ))}
        </div>
      </section>
      <section className="landing-section landing-testimonials-section">
        <div className="landing-section-head centered">
          <p className="landing-section-kicker">Proof</p>
          <h2>Loved by listeners building intentional experiences</h2>
          <p>Designed to feel cinematic on first view and dependable when the interaction gets real.</p>
        </div>
        <div className="landing-testimonials-marquee">
          <div className="landing-testimonials-track">
            {[...LANDING_TESTIMONIALS, ...LANDING_TESTIMONIALS].map((item, index) => (
              <article className="landing-testimonial-card" key={`${item.handle}-${index}`}>
                <div className="landing-testimonial-top">
                  <div className="landing-testimonial-avatar">{item.name.charAt(0)}</div>
                  <div>
                    <h3>{item.name}</h3>
                    <p>{item.handle}</p>
                  </div>
                </div>
                <blockquote>{item.quote}</blockquote>
              </article>
            ))}
          </div>
        </div>
      </section>
      <section className="landing-section landing-faq-section">
        <div className="landing-faq-copy">
          <p className="landing-section-kicker">FAQ</p>
          <h2>Questions before you start</h2>
          <p>Everything important about detection flow, locked state, and playback behavior in one place.</p>
        </div>
        <div className="landing-faq-list">
          {LANDING_FAQS.map((item, index) => {
            const isOpen = openFaqIndex === index;
            return (
              <article className={`landing-faq-item ${isOpen ? "open" : ""}`} key={item.question}>
                <button
                  type="button"
                  className="landing-faq-trigger"
                  onClick={() => setOpenFaqIndex((prev) => (prev === index ? -1 : index))}
                >
                  <span>{item.question}</span>
                  <span className="landing-faq-icon" aria-hidden>{isOpen ? "−" : "+"}</span>
                </button>
                {isOpen ? <p className="landing-faq-answer">{item.answer}</p> : null}
              </article>
            );
          })}
        </div>
      </section>
      <section className="landing-footer-cta">
        <div className="landing-footer-cta-inner">
          <h2>Ready to turn emotion into playback?</h2>
          <p>Start with the landing flow, sign in, and let SONA lock a mood before the first track even begins.</p>
          <div className="landing-cta-row">
            <button className="landing-primary-btn big" onClick={() => openAuthModal("signup")}>Create Account</button>
            <button className="landing-ghost-btn" onClick={() => openAuthModal("signin")}>Login</button>
          </div>
        </div>
      </section>
      <footer className="landing-footer">
        <button type="button" className="brand brand-button" onClick={handleBrandClick} aria-label="Go to home page">
          <img className="brand-logo" src="/assets/logo/sona-logo.jpeg" alt="SONA logo" />
          <span className="brand-text">Sona</span>
        </button>
        <p>Sentimental oriented neural audio for face, voice, and recommendation-driven listening.</p>
      </footer>
      {showAuthModal ? renderAuthModal() : null}
    </div>
  );

  return (
    !authReady ? (
      <div className="landing-shell loading">Loading SONA...</div>
    ) : !authUser ? (
      renderLanding()
    ) : (
    <div className="sona-app">
      <header className="top-nav">
        <div className="top-nav-left">
          <div className="sidebar-hamburger" onClick={() => setSidebarCollapsed((prev) => !prev)} role="button" tabIndex={0} aria-label="Toggle sidebar">
            <span />
            <span />
            <span />
          </div>
          <button type="button" className="brand brand-button" onClick={handleBrandClick} aria-label="Go to home page">
            <img className="brand-logo" src="/assets/logo/sona-logo.jpeg" alt="SONA logo" />
            <span className="brand-text">Sona</span>
          </button>
        </div>
        <div className="top-nav-right">
          <nav className="top-links">
            <button
              type="button"
              className={`theme-toggle nav-theme-toggle ${isLightMode ? "active" : ""}`}
              onClick={() => setIsLightMode((prev) => !prev)}
              title="Toggle theme"
              aria-label="Toggle theme"
            >
              <ThemeIcon isLightMode={isLightMode} />
            </button>
            {NAV_ITEMS.map((item) => (
              <button key={item.key} className={activePage === item.key ? "active" : ""} onClick={() => navigateToPage(item.key)}>
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="top-meta">
            <button
              type="button"
              className="profile-icon-btn"
              onClick={() => setShowProfileMenu((prev) => !prev)}
              title="Settings"
              aria-label="Settings"
            >
              <ProfileIcon />
            </button>
            {showProfileMenu ? (
              <div className="profile-menu-card">
                <p className="profile-menu-name">{authUser?.username || "User"}</p>
                <p className="profile-menu-email">{authUser?.email || "user@sona.app"}</p>
                <div className="profile-menu-icon-row">
                  <button
                    type="button"
                    className="profile-menu-icon-btn"
                    title="Settings"
                    aria-label="Settings"
                    onClick={() => {
                      navigateToPage("profile");
                      setShowProfileMenu(false);
                    }}
                  >
                    <SettingsIcon />
                  </button>
                  <button
                    type="button"
                    className="profile-menu-icon-btn logout"
                    title="Logout"
                    aria-label="Logout"
                    onClick={handleLogout}
                  >
                    <LogoutIcon />
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <div className="layout-shell">
        <aside className={sidebarCollapsed ? "left-rail collapsed" : "left-rail"}>
          {!sidebarCollapsed
            ? (
                <>
                  {NAV_ITEMS.map((item) => (
                    <div
                      key={`side-${item.key}`}
                      className={activePage === item.key ? "side-text active" : "side-text"}
                      onClick={() => navigateToPage(item.key)}
                      role="button"
                      tabIndex={0}
                    >
                      {item.label}
                    </div>
                  ))}
                  <div className="side-logout-wrap">
                    <div className="side-text side-logout" onClick={handleLogout} role="button" tabIndex={0}>
                      Logout
                    </div>
                  </div>
                </>
              )
            : null}
        </aside>

        <main className="content-area">
          {activePage === "home" ? renderHome() : null}
          {activePage === "metrics" ? renderMetrics() : null}
          {activePage === "profile" ? renderProfile() : null}
          {activePage === "contacts" ? renderContacts() : null}
        </main>

        <aside className="right-rail">
          <section className="player-card">
            <div className="player-head">
              <h3>Music Player</h3>
              <span className={playbackMode === "autoplay" ? "mode-chip auto" : "mode-chip manual"}>{playbackMode === "autoplay" ? "Autoplay" : "Manual"}</span>
            </div>
            <div className="artwork">
              {getTrackArtwork(currentTrack) ? (
                <img src={getTrackArtwork(currentTrack)} alt={currentTrack?.name || "Track artwork"} />
              ) : currentTrack ? (
                <MusicNoteIcon />
              ) : (
                <MusicNoteIcon />
              )}
            </div>
            <p className="track-title">{currentTrack?.name || "No track loaded"}</p>
            <p className="track-meta">
              {currentTrack
                ? `${currentTrack.artist || "Unknown"} • ${currentTrack.language || inferTrackLanguage(currentTrack)} • ${sourceLabel}`
                : "Waiting for new emotion lock"}
            </p>
            <div className="controls">
              <button aria-label="Previous track" title="Previous" onClick={playPrev}><PrevTrackIcon /></button>
              <button aria-label={isPlaying ? "Pause" : "Play"} title={isPlaying ? "Pause" : "Play"} onClick={togglePlayPause}>
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </button>
              <button aria-label="Next track" title="Next" onClick={playNext}><NextTrackIcon /></button>
            </div>
            <input
              className="player-seek"
              type="range"
              min={0}
              max={duration || 0}
              value={seek}
              style={{ "--seek-pct": `${duration ? Math.min(100, (seek / duration) * 100) : 0}%` }}
              onChange={(e) => {
                const v = Number(e.target.value);
                setSeek(v);
                if (audioRef.current) audioRef.current.currentTime = v;
              }}
            />
            <div className="time-row"><span>{formatClockTime(seek)}</span><span>{formatClockTime(duration)}</span></div>
            <audio
              ref={audioRef}
              src={currentTrack?.preview_url || undefined}
              preload="auto"
              crossOrigin="anonymous"
              onTimeUpdate={(e) => setSeek(e.currentTarget.currentTime)}
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
              onEnded={() => { setIsPlaying(false); playNext(); }}
              onError={() => {
                if (!currentTrack?.preview_url) return;
                setIsPlaying(false);
                setUiError((prev) => prev || "Track preview failed. Trying the next song.");
                if (playlist.length > 1) {
                  setTimeout(() => playNext(), 0);
                }
              }}
            />
          </section>

          <section className="playlist-card">
            <h3>Playlist</h3>
            <div className="playlist-list">
              {playlist.length === 0 ? (
                <p className="empty">Playlist refreshes only after a new emotion lock.</p>
              ) : (
                playlist.map((track, index) => (
                  <div
                    key={track.id || `${track.name}-${index}`}
                    className={index === currentTrackIndex ? "playlist-entry active" : "playlist-entry"}
                    onClick={() => setCurrentTrackIndex(index)}
                    role="button"
                    tabIndex={0}
                  >
                    <p className="playlist-line">
                      <span className="playlist-index">{index + 1}.</span> {track.name || "Untitled"}
                    </p>
                    <small className="playlist-subline">{`${track.artist || "Unknown artist"} • ${track.language || inferTrackLanguage(track)}`}</small>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </div>

      <footer className="app-footer">Retake always stops playback and keeps player idle until a new lock event.</footer>
    </div>
    )
  );
}
