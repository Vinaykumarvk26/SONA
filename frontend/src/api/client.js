import axios from "axios";

const AUTH_TOKEN_KEY = "emotion_auth_token";
const rawExplicitApiBase = (import.meta.env.VITE_API_URL || "").trim();
const runtimeHost = typeof window !== "undefined" ? window.location.hostname : "localhost";
const runtimeDirectApiBase = `http://${runtimeHost || "localhost"}:8010/api/v1`;

function normalizeApiBase(base) {
  const value = (base || "").trim().replace(/\/+$/, "");
  if (!value) return "";
  if (value.endsWith("/api/v1")) return value;
  if (/^https?:\/\/[^/]+$/i.test(value)) return `${value}/api/v1`;
  return value;
}

const explicitApiBase = normalizeApiBase(rawExplicitApiBase);
const defaultApiBase = explicitApiBase || (import.meta.env.DEV ? "/api/v1" : runtimeDirectApiBase);

const api = axios.create({
  baseURL: defaultApiBase,
});

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY);
  }
}

export function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY) || "";
}

api.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

const fallbackApiBases = [];
if (import.meta.env.DEV) {
  fallbackApiBases.push("/api/v1", runtimeDirectApiBase, "http://127.0.0.1:8010/api/v1", "http://localhost:8010/api/v1");
}
if (rawExplicitApiBase && explicitApiBase && explicitApiBase !== rawExplicitApiBase) {
  fallbackApiBases.unshift(explicitApiBase);
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error?.config;
    if (!config) return Promise.reject(error);

    // Exponential backoff logic for connection issues (cold starts)
    if (!error?.response) {
      config.__retryCount = config.__retryCount || 0;
      if (config.__retryCount < 3) {
        config.__retryCount += 1;
        const waitTime = config.__retryCount * 2000; // 2s, 4s, 6s
        await delay(waitTime);
        return api.request(config);
      }
    }

    if (error?.response || config.__apiFallbackRetried || fallbackApiBases.length === 0 || import.meta.env.PROD) {
      return Promise.reject(error);
    }

    config.__apiFallbackRetried = true;
    for (const candidate of fallbackApiBases) {
      if (!candidate || candidate === config.baseURL || candidate === api.defaults.baseURL) continue;
      try {
        api.defaults.baseURL = candidate;
        config.baseURL = candidate;
        return await api.request(config);
      } catch (retryErr) {
        if (retryErr?.response) return Promise.reject(retryErr);
      }
    }
    return Promise.reject(error);
  }
);

export default api;
