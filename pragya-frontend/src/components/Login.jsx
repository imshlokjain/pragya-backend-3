import { useState } from "react";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();

    setLoading(true);
    setError("");

    try {
      await onLogin(username, password);
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-background"></div>

      <div className="login-container">
        <div className="brand login-brand">
          <div className="brand-icon">P</div>
          <div>
            <h1>PRAGYA</h1>
            <span>Flood Intelligence Platform</span>
          </div>
        </div>

        <div className="login-card">
          <div className="login-header">
            <p className="eyebrow">SECURE ACCESS</p>
            <h2>Welcome back</h2>
            <p>Sign in to access the flood monitoring dashboard.</p>
          </div>

          <form onSubmit={handleSubmit}>
            <label>
              Username
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
              />
            </label>

            <label>
              Password
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
              />
            </label>

            {error && (
              <div className="error-message">
                ⚠ {error}
              </div>
            )}

            <button
              className="primary-button login-button"
              disabled={loading}
            >
              {loading ? "Authenticating..." : "Access Dashboard →"}
            </button>
          </form>

          <div className="login-footer">
            <span className="status-dot"></span>
            Backend security system online
          </div>
        </div>
      </div>
    </div>
  );
}
