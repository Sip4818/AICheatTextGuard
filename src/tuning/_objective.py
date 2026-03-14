from random import seed
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.constants.constants import SEED


class Objective:
    def __init__(self, model_name, search_spaces, cfg, X, y):
        self.model_name = model_name
        self.search_spaces = search_spaces
        self.cfg = cfg
        self.X = X
        self.y = y

    def __call__(self, trial):
        # 1. Build hyperparams dynamically from YAML
        params = self.search_spaces._build_space(trial, self.cfg)

        # 2. Build model
        model = self._build_model(params)

        # 3. Evaluate with CV
        score = self._evaluate(model)

        return score

    def _build_model(self, params):
        if self.model_name == "lr":
            return LogisticRegression(**params)

        if self.model_name == "xgb":
            return XGBClassifier(**params)

        raise ValueError(f"Unknown model: {self.model_name}")

    def _evaluate(self, model):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        auc = cross_val_score(model, self.X, self.y, cv=cv, scoring="roc_auc").mean()

        return auc
