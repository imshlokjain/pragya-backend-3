export default function Drivers({ drivers }) {
  return (
    <div className="panel drivers-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">MODEL EXPLAINABILITY</p>
          <h3>Risk Drivers</h3>
        </div>

        <span className="panel-badge">
          {drivers?.length || 0} signals
        </span>
      </div>

      {!drivers || drivers.length === 0 ? (
        <div className="empty-state">
          No contributing factors available.
        </div>
      ) : (
        <div className="drivers-list">
          {drivers.map((driver, index) => {
            if (typeof driver === "string") {
              return (
                <div className="driver-item" key={index}>
                  <div className="driver-number">
                    {String(index + 1).padStart(2, "0")}
                  </div>

                  <div className="driver-content">
                    <strong>{driver}</strong>
                    <span>Detected risk signal</span>
                  </div>

                  <div className="driver-arrow">→</div>
                </div>
              );
            }

            return (
              <div className="driver-item" key={index}>
                <div className="driver-number">
                  {String(index + 1).padStart(2, "0")}
                </div>

                <div className="driver-content">
                  <strong>{driver.feature}</strong>
                  <span>
                    Value: {driver.value} · {driver.direction}
                  </span>
                </div>

                <div className="driver-contribution">
                  {driver.contribution}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
