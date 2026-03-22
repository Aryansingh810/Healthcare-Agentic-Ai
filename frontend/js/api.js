const API_BASE = "";

async function apiFetch(path, options = {}) {
  const opts = {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  };
  if (opts.body && typeof opts.body === "object" && !(opts.body instanceof FormData)) {
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(API_BASE + path, opts);
  let data;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    data = await res.json();
  } else {
    data = await res.text();
  }
  if (!res.ok) {
    const err = new Error(typeof data === "object" && data.error ? data.error : res.statusText);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

async function getSession() {
  return apiFetch("/session");
}

async function signup(body) {
  return apiFetch("/signup", { method: "POST", body });
}

async function login(body) {
  return apiFetch("/login", { method: "POST", body });
}

async function logout() {
  return apiFetch("/logout", { method: "POST" });
}

async function uploadId(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(API_BASE + "/upload-id", {
    method: "POST",
    body: fd,
    credentials: "include",
  });
  const data = await res.json();
  if (!res.ok) throw Object.assign(new Error(data.error || "Upload failed"), { data });
  return data;
}

async function verifyId(path) {
  return apiFetch("/verify-id", {
    method: "POST",
    body: path ? { path } : {},
  });
}

async function runPlanner(goal) {
  return apiFetch("/planner-agent", {
    method: "POST",
    body: { goal },
  });
}

async function symptomCheck(symptoms) {
  return apiFetch("/symptom-check", {
    method: "POST",
    body: { symptoms },
  });
}

async function getAppointments() {
  return apiFetch("/doctor/appointments");
}

async function getPatientQueries() {
  return apiFetch("/doctor/patient-queries");
}

async function getDoctorProfile() {
  return apiFetch("/doctor/profile");
}
