import api, { getAuthToken, setAuthToken } from "./client";

export async function signupWithDetails(payload) {
  const normalized = {
    username: payload.username,
    email: payload.email,
    password: payload.password,
  };
  const { data } = await api.post("/auth/signup", normalized);
  setAuthToken(data.token);
  return data;
}

export async function signinWithIdentifier(identifier, password) {
  const normalized = (identifier || "").trim();
  const { data } = await api.post("/auth/signin", {
    identifier: normalized,
    email: normalized,
    password,
  });
  setAuthToken(data.token);
  return data;
}

export async function signinWithGoogleAccessToken(accessToken) {
  const { data } = await api.post("/auth/google", { access_token: accessToken });
  setAuthToken(data.token);
  return data;
}

export async function requestPasswordReset(identifier) {
  const { data } = await api.post("/auth/forgot-password", { identifier });
  return data;
}

export async function resetPassword(resetToken, newPassword) {
  const { data } = await api.post("/auth/reset-password", {
    reset_token: resetToken,
    new_password: newPassword,
  });
  return data;
}

export async function getCurrentUser() {
  const token = getAuthToken();
  if (!token) return null;
  const { data } = await api.get("/auth/me");
  return data;
}

export async function updateMyProfile(payload) {
  const { data } = await api.patch("/auth/me", payload);
  return data;
}

export async function getMyPreferences() {
  const { data } = await api.get("/preferences/me");
  return data;
}

export async function saveMyPreferences(payload) {
  const { data } = await api.put("/preferences/me", payload);
  return data;
}

export function signout() {
  setAuthToken("");
}
