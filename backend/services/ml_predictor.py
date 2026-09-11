import os
import joblib
import pandas as pd


from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
_POSSIBLE_PATHS = [
    _PKG_ROOT / "ml" / "models" / "assam_river_model.pkl",
    Path("ml/models/assam_river_model.pkl"),
    Path("pragya-backend-3/ml/models/assam_river_model.pkl"),
]
MODEL_PATH = next((p for p in _POSSIBLE_PATHS if p.exists()), _POSSIBLE_PATHS[0])


class AssamRiverPredictor:

    def __init__(self):
        self.model = None
        self.features = None
        self.load_model()

    def load_model(self):
        """
        Load the trained Assam river prediction model.
        """

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        model_data = joblib.load(MODEL_PATH)

        self.model = model_data["model"]
        self.features = model_data["features"]

        print("Assam river prediction model loaded successfully.")

    def predict(
        self,
        month,
        day_of_year,
        rainfall_mm,
        rainfall_previous,
        rainfall_3_obs,
        rainfall_6_obs,
        river_level,
        river_level_change,
        river_level_rolling_mean_3,
        river_level_rolling_max_6,
    ):

        input_data = pd.DataFrame([{
            "month": month,
            "day_of_year": day_of_year,
            "rainfall_mm": rainfall_mm,
            "rainfall_previous": rainfall_previous,
            "rainfall_3_obs": rainfall_3_obs,
            "rainfall_6_obs": rainfall_6_obs,
            "river_level": river_level,
            "river_level_change": river_level_change,
            "river_level_rolling_mean_3": river_level_rolling_mean_3,
            "river_level_rolling_max_6": river_level_rolling_max_6,
        }])

        input_data = input_data[self.features]

        prediction = self.model.predict(input_data)[0]

        return float(prediction)


predictor = AssamRiverPredictor()
