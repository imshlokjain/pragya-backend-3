import { useEffect, useState } from "react";

import Login from "./components/Login";
import RiskCard from "./components/RiskCard";
import Drivers from "./components/Drivers";
import ScenarioPanel from "./components/ScenarioPanel";
import Chat from "./components/Chat";

import {
  login,
  clearToken,
  getRisk,
  predictRisk,
  getRainfall,
  runScenario,
  sendChat,
} from "./services/api";

import "./styles.css";


const ZONES = [
  {
    id: "5f041f43-3868-41f0-af0c-9fd21a4eea5f",
    name: "Assam River Basin",
    region: "Assam, India",
  },
];


export default function App() {
  const [authenticated, setAuthenticated] = useState(
    !!localStorage.getItem("pragya_token")
  );

  const [selectedZone, setSelectedZone] = useState(
    ZONES[0]
  );

  const [risk, setRisk] = useState(null);
  const [rainfall, setRainfall] = useState(null);

  const [loadingRisk, setLoadingRisk] =
    useState(false);

  const [error, setError] = useState("");

  const zoneId = selectedZone.id;


  async function loadDashboard() {
    setLoadingRisk(true);
    setError("");

    try {
      const [riskData, rainfallData] =
        await Promise.all([
          getRisk(zoneId),

          getRainfall(zoneId).catch(() => null),
        ]);

      setRisk(riskData);
      setRainfall(rainfallData);

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
          "Unable to load flood intelligence data."
      );

    } finally {
      setLoadingRisk(false);
    }
  }


  useEffect(() => {
    if (authenticated) {
      loadDashboard();
    }
  }, [authenticated, zoneId]);


  async function handleLogin(username, password) {
    await login(username, password);

    setAuthenticated(true);
  }


  async function handleRefreshPrediction() {
    setLoadingRisk(true);
    setError("");

    try {
      const prediction =
        await predictRisk(zoneId);

      setRisk(prediction);

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
          "Unable to refresh prediction."
      );

    } finally {
      setLoadingRisk(false);
    }
  }


  function handleLogout() {
    clearToken();

    setAuthenticated(false);

    setRisk(null);
    setRainfall(null);
  }


  if (!authenticated) {
    return (
      <Login
        onLogin={handleLogin}
      />
    );
  }


  return (
    <div className="app-shell">

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            ≋
          </div>

          <div>
            <h1>PRAGYA</h1>

            <span>
              Flood Intelligence
            </span>
          </div>

        </div>


        <nav>

          <button className="nav-item active">
            <span>▦</span>
            Overview
          </button>


          <button className="nav-item">
            <span>⌁</span>
            Risk Monitoring
          </button>


          <button className="nav-item">
            <span>◈</span>
            Scenario Lab
          </button>


          <button className="nav-item">
            <span>♧</span>
            AI Assistant
          </button>

        </nav>


        <div className="sidebar-bottom">

          <div className="system-status">
            <span></span>

            Systems Operational
          </div>


          <button
            className="logout-button"
            onClick={handleLogout}
          >
            Sign out →
          </button>

        </div>

      </aside>



      <main className="main-content">

        <header className="topbar">

          <div>

            <p className="eyebrow">
              AI-POWERED DECISION SUPPORT
            </p>


            <h2>
              Flood Intelligence Command Center
            </h2>


            <p className="topbar-description">
              Live observations, ML forecasting,
              and operational scenario analysis.
            </p>

          </div>



          <div className="topbar-actions">

            <div className="zone-selector">

              <span className="location-icon">
                📍
              </span>


              <div>

                <small>
                  SELECTED REGION
                </small>


                <select
                  value={selectedZone.id}
                  onChange={(e) => {

                    const zone =
                      ZONES.find(
                        (item) =>
                          item.id === e.target.value
                      );

                    if (zone) {
                      setSelectedZone(zone);
                    }

                  }}
                >

                  {ZONES.map((zone) => (

                    <option
                      key={zone.id}
                      value={zone.id}
                    >
                      {zone.name}
                    </option>

                  ))}

                </select>

              </div>

            </div>



            <div className="live-indicator">

              <span></span>

              LIVE DATA

            </div>


            <button
              className="refresh-button"
              onClick={loadDashboard}
              disabled={loadingRisk}
            >
              <span className={loadingRisk ? "spinning" : ""}>↻</span>
              {loadingRisk ? "Refreshing..." : "Refresh"}
            </button>

          </div>

        </header>



        <div className="location-banner">

          <div className="location-main">

            <span className="location-pin">
              📍
            </span>


            <div>

              <span className="location-label">
                MONITORING REGION
              </span>


              <strong>
                {selectedZone.name}
              </strong>


              <small>
                {selectedZone.region}
              </small>

            </div>

          </div>


          <div className="zone-id-display">

            <span>ZONE ID</span>

            <code>
              {zoneId}
            </code>

          </div>

        </div>



        {error && (

          <div className="global-error">

            ⚠ {error}

          </div>

        )}



        <section className="dashboard-grid">

          <div className="main-column">

            <RiskCard
              risk={risk}
              loading={loadingRisk}
              onRefresh={handleRefreshPrediction}
              zone={selectedZone}
            />


            <section className="observations-section">

              <div className="section-heading">

                <div>

                  <p className="eyebrow">
                    LIVE OBSERVATIONS
                  </p>

                  <h3>
                    River & Rainfall Intelligence
                  </h3>

                </div>


                <div className="region-mini-badge">
                  📍 {selectedZone.region}
                </div>

              </div>



              <div className="metrics-grid">


                <div className="metric-card">

                  <div className="metric-icon river">
                    ≋
                  </div>


                  <div>

                    <p>River Level</p>


                    <strong>
                      {risk?.river_level ??
                        risk?.water_level ??
                        "Unavailable"}
                    </strong>


                    <small>
                      Current monitored level
                    </small>

                  </div>

                </div>



                <div className="metric-card">

                  <div className="metric-icon forecast">
                    ⬡
                  </div>


                  <div>

                    <p>Forecast Change</p>


                    <strong>
                      {risk?.forecast_change ??
                        risk?.forecast_horizon_hours ??
                        "—"}
                    </strong>


                    <small>
                      ML forecast outlook
                    </small>

                  </div>

                </div>



                <div className="metric-card">

                  <div className="metric-icon rainfall">
                    ☔
                  </div>


                  <div>

                    <p>Current Rainfall</p>


                    <strong>
                      {rainfall?.rainfall_mm ??
                        "Unavailable"}
                      {rainfall?.rainfall_mm != null
                        ? " mm"
                        : ""}
                    </strong>


                    <small>
                      24-hour accumulation:{" "}
                      {rainfall?.rainfall_24h ??
                        "—"} mm
                    </small>

                  </div>

                </div>


              </div>

            </section>



            <Drivers
              drivers={risk?.drivers}
            />

          </div>



          <div className="side-column">

            <ScenarioPanel
              zoneId={zoneId}
              zoneName={selectedZone.name}
              onRunScenario={runScenario}
            />


            <Chat
              zoneId={zoneId}
              zoneName={selectedZone.name}
              onSend={sendChat}
            />

          </div>

        </section>

      </main>

    </div>
  );
}
