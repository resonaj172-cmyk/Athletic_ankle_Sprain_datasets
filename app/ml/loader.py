import json
import joblib
from pathlib import Path
import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


class ModelNotFoundError(Exception):
    pass


class ModelLoader:
    _instance = None
    _selected_metadata = None
    _loaded_model = None
    _model_config = None

    @classmethod
    def get_selected_metadata(cls):
        selected_json_path = config.MODELS_DIR / "selected_model.json"
        if not selected_json_path.exists():
            return None
        try:
            return json.loads(selected_json_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @classmethod
    def load_selected_model(cls, force_reload=False):
        """
        Loads only the model specified in models/selected_model.json.
        Returns (model, model_config_dict, selected_metadata).
        """
        if cls._loaded_model is not None and not force_reload:
            return cls._loaded_model, cls._model_config, cls._selected_metadata

        meta = cls.get_selected_metadata()
        if meta is None:
            raise ModelNotFoundError(
                "Selected model configuration (models/selected_model.json) is missing. "
                "Please run 'python training/run_all.py' first."
            )

        model_key = meta.get("model_key")
        model_path_str = meta.get("model_path")
        config_path_str = meta.get("config_path")

        model_path = Path(model_path_str) if model_path_str else (config.MODELS_DIR / model_key / "model.joblib")
        config_path = Path(config_path_str) if config_path_str else (config.MODELS_DIR / model_key / "model_config.json")

        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model file not found at: {model_path}. Please run training pipeline first."
            )

        if not config_path.exists():
            raise ModelNotFoundError(
                f"Model config not found at: {config_path}. Please run training pipeline first."
            )

        model_config = json.loads(config_path.read_text(encoding="utf-8"))

        if model_key == "xgboost":
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
        else:
            model = joblib.load(model_path)

        cls._loaded_model = model
        cls._model_config = model_config
        cls._selected_metadata = meta

        return cls._loaded_model, cls._model_config, cls._selected_metadata

    @classmethod
    def get_comparison_table(cls):
        comp_path = config.REPORTS_DIR / "model_comparison.csv"
        if not comp_path.exists():
            return None
        import pandas as pd
        return pd.read_csv(comp_path).to_dict(orient="records")

    @classmethod
    def get_selection_report(cls):
        rep_path = config.REPORTS_DIR / "selection_report.md"
        if not rep_path.exists():
            return None
        return rep_path.read_text(encoding="utf-8")
