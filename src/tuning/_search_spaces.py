from src.entity.model_trainer_tuning_entity import ModelTrainerTuningConfig
from collections.abc import Mapping
from dataclasses import asdict

class SearchSpaces:
    def __init__(self, cfg: ModelTrainerTuningConfig) -> None:
        self.cfg = cfg           

    def _build_space(self, trial, cfg):
        params = {}
        cfg=asdict(cfg)
        for key, spec in cfg.items():

            if isinstance(spec, Mapping) and "type" in spec:
                ptype = spec["type"]

                if ptype == "float":
                    params[key] = trial.suggest_float(
                        key,
                        spec["low"],
                        spec["high"],
                        log=spec.get("log", False)
                    )

                elif ptype == "int":
                    params[key] = trial.suggest_int(
                        key,
                        spec["low"],
                        spec["high"]
                    )

                else:
                    raise ValueError(f"Unsupported parameter type: {ptype}")

            else:
                params[key] = trial.suggest_categorical(key, spec)

        return params

