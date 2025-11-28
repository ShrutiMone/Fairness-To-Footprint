# backend/utils/mitigation.py

from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from .fairness_metrics import compute_fairness_metrics
from sklearn.base import BaseEstimator, ClassifierMixin


class PipelineSampleWeightAdapter(BaseEstimator, ClassifierMixin):
    """
    Wrapper to allow estimators which forward sample_weight to the final
    classifier step of a Pipeline. fairlearn's ExponentiatedGradient may
    pass `sample_weight` to the estimator's fit; sklearn's Pipeline does
    not accept `sample_weight` as a top-level kwarg — it must be passed
    to the final step using the `stepname__sample_weight` format. This
    small wrapper provides a fit signature that accepts sample_weight and
    forwards it correctly.
    """
    def __init__(self, pipeline, clf_step_name="clf"):
        self.pipeline = pipeline
        self.clf_step_name = clf_step_name

    def fit(self, X, y, sample_weight=None, **fit_params):
        kwargs = dict(fit_params)
        if sample_weight is not None:
            kwargs[f"{self.clf_step_name}__sample_weight"] = sample_weight
        self.pipeline.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_params(self, deep=True):
        out = {"pipeline": self.pipeline, "clf_step_name": self.clf_step_name}
        if deep and hasattr(self.pipeline, "get_params"):
            for k, v in self.pipeline.get_params(deep=True).items():
                out[f"pipeline__{k}"] = v
        return out

    def set_params(self, **params):
        if "pipeline" in params:
            self.pipeline = params.pop("pipeline")
        if "clf_step_name" in params:
            self.clf_step_name = params.pop("clf_step_name")
        pipeline_params = {k.replace("pipeline__", ""): v for k, v in params.items() if k.startswith("pipeline__")}
        if pipeline_params:
            self.pipeline.set_params(**pipeline_params)
        return self


def _prepare_features(df: pd.DataFrame, target_col: str, sensitive_col: str):
    """
    Prepares numerical + categorical preprocessing transformer.
    Sensitive attribute is excluded from X.
    """
    X = df.drop(columns=[target_col, sensitive_col], errors='ignore')

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )
    return transformer


def mitigate_with_exponentiated_gradient(df: pd.DataFrame, target_col: str, sensitive_col: str,
                                         constraint="demographic_parity"):
    """
    Runs Fairlearn mitigation using ExponentiatedGradient with chosen constraint.
    - Ensures target is binary numeric (0/1)
    - Ensures sensitive attribute is not in the feature set
    - Computes fairness metrics after mitigation
    """

    # ===============================
    # 1. Encode target → 0/1
    # ===============================
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])   # Yes/No → 1/0

    # Sensitive features remain as-is (string labels allowed)
    sensitive = df[sensitive_col].values

    # ===============================
    # 2. Prepare feature matrix X
    # ===============================
    transformer = _prepare_features(df, target_col, sensitive_col)

    X = df.drop(columns=[target_col])              # original X
    X = X.drop(columns=[sensitive_col])            # remove sensitive from X

    # ===============================
    # 3. Build model
    # ===============================
    base_clf = LogisticRegression(max_iter=2000)

    clf_pipeline = Pipeline(steps=[
        ("pre", transformer),
        ("clf", base_clf)
    ])

    wrapped_estimator = PipelineSampleWeightAdapter(clf_pipeline, clf_step_name="clf")

    # Choose constraint
    cons = DemographicParity() if constraint == "demographic_parity" else EqualizedOdds()

    mitigator = ExponentiatedGradient(
        estimator=wrapped_estimator,
        constraints=cons
    )

    # ===============================
    # 4. Fit baseline pipeline for pre-mitigation metrics
    # ===============================
    clf_pipeline.fit(X, y)
    y_pred_baseline = clf_pipeline.predict(X)

    # ===============================
    # 5. Fit mitigator
    # ===============================
    mitigator.fit(X, y, sensitive_features=sensitive)

    # Predict
    y_pred_mitigated = mitigator.predict(X)

    # ===============================
    # 6. Compute baseline (pre-mitigation) metrics
    # ===============================
    tmp_baseline = df.copy()
    tmp_baseline["y_pred_baseline"] = y_pred_baseline

    metrics_baseline = compute_fairness_metrics(
        tmp_baseline,
        target_col=target_col,
        sensitive_col=sensitive_col,
        pred_col="y_pred_baseline"
    )

    # ===============================
    # 6. Build temp DF for mitigation metric evaluation
    # ===============================
    tmp = df.copy()
    tmp["y_pred_mitigated"] = y_pred_mitigated

    # ===============================
    # 7. Compute mitigated metrics
    # ===============================
    metrics_after = compute_fairness_metrics(
        tmp,
        target_col=target_col,
        sensitive_col=sensitive_col,
        pred_col="y_pred_mitigated"
    )

    return {
        "predictions": y_pred_mitigated.tolist(),
        "metrics_baseline": metrics_baseline,
        "metrics_after_mitigation": metrics_after,
        "num_predictors": len(mitigator.predictors_),
        "weights": [float(w) for w in mitigator.weights_],
        "mitigator": mitigator,
        "transformer": transformer,
        "label_encoder": le
    }


def mitigate_user_model(df: pd.DataFrame, user_model, target_col: str, sensitive_col: str,
                       constraint="demographic_parity"):
    """
    Applies fairness mitigation to a user-provided pre-trained model.
    
    Args:
        df: DataFrame with data (must contain target_col and sensitive_col)
        user_model: Pre-trained classifier with .predict() method
        target_col: Name of target column (ground truth labels)
        sensitive_col: Name of sensitive attribute column
        constraint: "demographic_parity" or "equalized_odds"
    
    Returns:
        Dict with baseline metrics, mitigated metrics, and mitigator object
    """
    
    # ===============================
    # 1. Encode target → 0/1
    # ===============================
    le = LabelEncoder()
    y_true = le.fit_transform(df[target_col])
    
    # Sensitive features
    sensitive = df[sensitive_col].values
    
    # ===============================
    # 2. Prepare feature matrix X
    # ===============================
    X = df.drop(columns=[target_col, sensitive_col], errors='ignore')
    
    # ===============================
    # 3. Get baseline predictions from user model
    # ===============================
    y_pred_baseline = user_model.predict(X)
    
    # Ensure binary encoding
    y_pred_baseline = np.asarray(y_pred_baseline)
    if y_pred_baseline.dtype == object:
        le_baseline = LabelEncoder()
        y_pred_baseline = le_baseline.fit_transform(y_pred_baseline)
    else:
        y_pred_baseline = y_pred_baseline.astype(int)
    
    # Compute baseline metrics
    tmp_baseline = df.copy()
    tmp_baseline["y_pred_baseline"] = y_pred_baseline
    
    metrics_baseline = compute_fairness_metrics(
        tmp_baseline,
        target_col=target_col,
        sensitive_col=sensitive_col,
        pred_col="y_pred_baseline"
    )
    
    # ===============================
    # 4. Wrap user model for mitigation
    # ===============================
    class UserModelWrapper(BaseEstimator, ClassifierMixin):
        """Wraps user's pre-trained model to work with fairlearn."""
        def __init__(self, model):
            self.model = model
        
        def fit(self, X, y, **fit_params):
            # User model is already trained; just return self
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            return None
    
    wrapped_model = UserModelWrapper(user_model)
    
    # ===============================
    # 5. Choose constraint and create mitigator
    # ===============================
    cons = DemographicParity() if constraint == "demographic_parity" else EqualizedOdds()
    
    mitigator = ExponentiatedGradient(
        estimator=wrapped_model,
        constraints=cons
    )
    
    # ===============================
    # 6. Fit mitigator (learns mixture weights)
    # ===============================
    mitigator.fit(X, y_true, sensitive_features=sensitive)
    
    # ===============================
    # 7. Get mitigated predictions
    # ===============================
    y_pred_mitigated = mitigator.predict(X)
    
    # Compute mitigated metrics
    tmp_mitigated = df.copy()
    tmp_mitigated["y_pred_mitigated"] = y_pred_mitigated
    
    metrics_after = compute_fairness_metrics(
        tmp_mitigated,
        target_col=target_col,
        sensitive_col=sensitive_col,
        pred_col="y_pred_mitigated"
    )
    
    return {
        "predictions": y_pred_mitigated.tolist(),
        "metrics_baseline": metrics_baseline,
        "metrics_after_mitigation": metrics_after,
        "num_predictors": len(mitigator.predictors_),
        "weights": [float(w) for w in mitigator.weights_],
        "mitigator": mitigator,
        "user_model": user_model
    }
