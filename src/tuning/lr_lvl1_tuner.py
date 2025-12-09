def lr_lvl1_hp_tuning(self):
        def objective(trial):
            params=self.model_tuning_config.level1.lr
            C = trial.suggest_loguniform('C', 1e-4, 1e2)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            
            # Choose valid solver based on penalty
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                raise optuna.TrialPruned()  # invalid combo
            if penalty == 'elasticnet' and solver != 'saga':
                raise optuna.TrialPruned()
            
            l1_ratio = None
            if penalty == 'elasticnet':
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
            tune_params=self.mode_tuning_config.level1.lr
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    tune_params
                ))
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            return auc

        study = optuna.create_study(direction='maximize')  # maximize AUC
        study.optimize(objective, n_trials=30)
        print("Best trial:")
        print(f"  ROC-AUC Value: {study.best_trial.value:.4f}")
        print("  Best Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")