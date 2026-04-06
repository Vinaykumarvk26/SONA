import api from "./client";

export async function getBackendHealth() {
  const { data } = await api.get("/health");
  return data;
}

export async function inferFrame(blob, userId = "") {
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  if (userId?.trim()) {
    form.append("user_id", userId.trim());
  }
  const { data } = await api.post("/detect-face", form);
  return data;
}

export async function inferSpeech(blob, transcript = "", userId = "") {
  const form = new FormData();
  form.append("audio", blob, "speech.wav");
  if (transcript?.trim()) {
    form.append("transcript", transcript.trim());
  }
  if (userId?.trim()) {
    form.append("user_id", userId.trim());
  }
  const { data } = await api.post("/detect-voice", form);
  return data;
}

export async function inferMultimodal({ userId, context, imageBlob, audioBlob, transcript = "" }) {
  const form = new FormData();
  form.append("user_id", userId || "default-user");
  form.append("time_of_day", String(context?.time_of_day ?? 0.5));
  form.append("skip_rate", String(context?.skip_rate ?? 0));
  form.append("device_type", context?.device_type || "desktop");
  form.append("listening_history", (context?.listening_history || []).join(","));
  if (transcript?.trim()) {
    form.append("transcript", transcript.trim());
  }

  if (imageBlob) {
    form.append("image", imageBlob, "frame.jpg");
  }
  if (audioBlob) {
    form.append("audio", audioBlob, "speech.wav");
  }

  const { data } = await api.post("/detect-multimodal", form);
  return data;
}

export async function getRecommendations(userId, emotion, confidence) {
  const { data } = await api.get("/recommendations", {
    params: { user_id: userId, emotion, confidence },
  });
  return data;
}

export async function submitFeedback(payload) {
  const { data } = await api.post("/feedback", payload);
  return data;
}

export async function getTimeline(userId) {
  const { data } = await api.get("/emotion/timeline", {
    params: { user_id: userId },
  });
  return data;
}

export async function getMetricsOverview(userId) {
  const { data } = await api.get("/metrics/overview", {
    params: { user_id: userId || "default-user" },
  });
  return data;
}

export async function getModelMetrics(inputType = "all") {
  const { data } = await api.get("/metrics", {
    params: { input_type: inputType },
  });
  return data;
}

export async function runModelEvaluation(inputType = "all") {
  const { data } = await api.post("/metrics/evaluate", { input_type: inputType });
  return data;
}

export async function logMetricsEvent(payload) {
  const { data } = await api.post("/metrics/events", payload);
  return data;
}

export async function submitEmotionFeedback(payload) {
  const { data } = await api.post("/metrics/emotion-feedback", payload);
  return data;
}
