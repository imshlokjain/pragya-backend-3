import { useState } from "react";

export default function ScenarioPanel({
  zoneId,
  onRunScenario,
}) {
  const [rainfallMultiplier, setRainfallMultiplier] =
    useState(1.2);

  const [riverIncrease, setRiverIncrease] =
    useState(0.5);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleScenario() {
    setLoading(true);
    setError("");

    try {
      const data = await onRunScenario({
        zone_id: zoneId,
        rainfall_multiplier: Number(rainfallMultiplier),
        river_level_increase: Number(riverIncrease),
      });

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel scenario-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">SCENARIO ANALYSIS</p>
          <h3>Scenario Lab</h3>
        </div>

        <span className="simulation-icon">◈</span>
      </div>

      <div className="scenario-input">
        <div className="input-label">
          <span>Rainfall Multiplier</span>
          <strong>{rainfallMultiplier}×</strong>
        </div>

        <input
          type="range"
          min="1"
          max="1.5"
          step="0.1"
          value={rainfallMultiplier}
          onChange={(e) =>
            setRainfallMultiplier(e.target.value)
          }
        />

        <div className="range-labels">
          <span>Normal</span>
          <span>+50%</span>
        </div>
      </div>

      <div className="scenario-input">
        <div className="input-label">
          <span>River Level Increase</span>
          <strong>{riverIncrease} m</strong>
        </div>

        <input
          type="range"
          min="0"
          max="3"
          step="0.1"
          value={riverIncrease}
          onChange={(e) =>
            setRiverIncrease(e.target.value)
          }
        />

        <div className="range-labels">
          <span>0 m</span>
          <span>+3 m</span>
        </div>
      </div>

      <button
        className="primary-button scenario-button"
        onClick={handleScenario}
        disabled={loading}
      >
        {loading ? "Running Simulation..." : "Run Simulation →"}
      </button>

      {error && (
        <div className="error-message">
          ⚠ {error}
        </div>
      )}

      {result && (
        <div className="scenario-result">
          <div><span>Baseline · {result.baseline_category}</span><strong>{result.baseline_risk}</strong></div>

          <div className="scenario-arrow">→</div>

          <div><span>Scenario · {result.scenario_category}</span><strong>{result.scenario_risk}</strong></div>

          <div className="risk-change">
            {result.risk_change >= 0 ? "+" : ""}{result.risk_change} pts
          </div>
        </div>
      )}
    </div>
  );
}
