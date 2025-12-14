import numpy as np
class StackedModel:
    def __init__(self, preprocessor, lr_model, xgb_model, meta_model):
        self.preprocessor = preprocessor
        self.lr = lr_model
        self.xgb = xgb_model
        self.meta = meta_model

    def predict_proba(self, X):
        Xp = self.preprocessor.transform(X)

        p_lr = self.lr.predict_proba(Xp)[:, 1]
        p_xgb = self.xgb.predict_proba(Xp)[:, 1]

        meta_X = np.column_stack([p_xgb, p_lr])
        return self.meta.predict_proba(meta_X)
