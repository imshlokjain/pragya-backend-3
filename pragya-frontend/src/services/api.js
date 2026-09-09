export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

let token = localStorage.getItem("pragya_token") || "";

export function setToken(newToken) {
  token = newToken;
  localStorage.setItem("pragya_token", newToken);
}

export function clearToken() {
  token = "";
  localStorage.removeItem("pragya_token");
}

async function request(endpoint, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options.headers,
      },
    });
  } catch {
    throw new Error("Unable to reach PRAGYA backend. Check that the API server is running.");
  }

  if (!response.ok) {
    let message = "Something went wrong";

    try {
      const data = await response.json();
      message = typeof data.detail === "string" ? data.detail : "The request could not be completed.";
    } catch {
      message = await response.text();
    }

    if (response.status === 401) clearToken();
    throw new Error(response.status === 401 ? "Your session has expired. Please sign in again." : message);
  }

  return response.json();
}

export async function login(username, password) {
  const response = await fetch(
    `${API_BASE_URL}/auth/login?username=${encodeURIComponent(
      username
    )}&password=${encodeURIComponent(password)}`,
    {
      method: "POST",
      headers: {
        accept: "application/json",
      },
    }
  );

  if (!response.ok) {
    throw new Error("Invalid username or password");
  }

  const data = await response.json();

  const accessToken = data.access_token || data.token;

  if (!accessToken) {
    throw new Error("Authentication token not received");
  }

  setToken(accessToken);

  return data;
}

export function getRisk(zoneId) {
  return request(`/risk/${zoneId}`);
}

export function getZones() {
  return request("/zones");
}

export function predictRisk(zoneId) {
  return request(`/risk/${zoneId}/predict`, {
    method: "POST",
  });
}

export function getRainfall(zoneId) {
  return request(`/rainfall/${zoneId}`);
}

export function getRiverForecast(zoneId) {
  return request(`/river/${zoneId}/forecast`);
}

export function getFloodHistory(zoneId) {
  return request(`/flood/${zoneId}/history?limit=20`);
}

export function runScenario(data) {
  return request("/scenario", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function sendChat(message, zoneId, districtId) {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      context: {
        zone_id: zoneId,
        ...(districtId ? { district_id: districtId } : {}),
      },
    }),
  });
}
