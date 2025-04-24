import numpy as np
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_validate, KFold
)
from sklearn.metrics import (
    make_scorer, mean_squared_error,
    mean_absolute_error, r2_score
)

from data_preparation import collect_dataset_from_directory

# 1) Define pipelines + parameter grids
svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])
svr_param_grid = {
    "svr__kernel": ["linear", "rbf"],
    "svr__C": [0.1, 1, 10],
    "svr__gamma": ["scale", "auto", 0.1],
}

gb_pipe = Pipeline([
    ("scaler", StandardScaler()),   # helps Gradient Boosting too
    ("gb", GradientBoostingRegressor(random_state=42))
])
gb_param_grid = {
    "gb__n_estimators": [50, 100, 200],
    "gb__learning_rate": [0.01, 0.1, 0.2],
    "gb__max_depth": [3, 5, 7],
}

# 2) Setup nested CV
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# scoring dict for multiple metrics
scoring = {
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score)
}

def nested_cv_evaluate(X, y, pipe, param_grid, name="Model", search_type="grid", n_iter=20):
    """
    Perform nested cross-validation and print summary metrics.
    """
    # choose search method
    if search_type == "random":
        search = RandomizedSearchCV(
            pipe, param_grid,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42,
            verbose=1
        )
    else:
        search = GridSearchCV(
            pipe, param_grid,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
    # run outer CV
    cv_results = cross_validate(
        search, X, y,
        cv=outer_cv,
        scoring=scoring,
        return_train_score=False
    )
    # assemble summary
    summary = pd.DataFrame({
        "Fold": range(1, outer_cv.get_n_splits() + 1),
        "Test MSE": -cv_results["test_MSE"],
        "Test MAE": -cv_results["test_MAE"],
        "Test R2": cv_results["test_R2"],
    })
    print(f"\n=== Nested CV results for {name} ===")
    print(summary)
    print(f"Mean Test MSE: {summary['Test MSE'].mean():.4f}")
    print(f"Mean Test MAE: {summary['Test MAE'].mean():.4f}")
    print(f"Mean Test R2:  {summary['Test R2'].mean():.4f}\n")
    return summary


def train_final_model(X, y, pipe, param_grid, name="Model", 
                      search_type="grid", n_iter=20, output_dir="models"):  
    """
    Train a final pipeline on the full dataset using best hyperparameters,
    then save model, scaler, and feature list for deployment.
    Returns file paths: (model_path, scaler_path, features_path)
    """
    # build search
    if search_type == "random":
        search = RandomizedSearchCV(
            pipe, param_grid,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42,
            verbose=1
        )
    else:
        search = GridSearchCV(
            pipe, param_grid,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
    # fit on full data
    search.fit(X, y)
    final_model = search.best_estimator_
    best_params = search.best_params_
    print(f"Best {name} parameters: {best_params}")

    # make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # file paths
    base = os.path.join(output_dir, name.lower().replace(' ', '_'))
    model_path = base + "_model.pkl"
    scaler_path = base + "_scaler.pkl"
    features_path = base + "_features.pkl"

    # extract and save
    joblib.dump(final_model.named_steps[next(iter([k for k in final_model.named_steps if k!='scaler']))], model_path)
    joblib.dump(final_model.named_steps['scaler'], scaler_path)
    joblib.dump(list(X.columns), features_path)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved feature list to {features_path}")

    return model_path, scaler_path, features_path


def run_ml_pipeline(data_dir, tool_lifetime=3600, output_dir="models"):
    """
    Full ML pipeline: collect data, nested CV compare, then train & save final best model.
    Returns (model_path, scaler_path, features_path)
    """
    # 1) collect features & labels
    X, y = collect_dataset_from_directory(data_dir, tool_lifetime)

    # 2) nested CV for both models
    svr_summary = nested_cv_evaluate(X, y, svr_pipe, svr_param_grid, "SVR", search_type="grid")
    gb_summary  = nested_cv_evaluate(X, y, gb_pipe, gb_param_grid, "Gradient_Boosting", search_type="random", n_iter=30)

    # choose best by mean R2
    svr_r2 = svr_summary['Test R2'].mean()
    gb_r2  = gb_summary['Test R2'].mean()
    if gb_r2 > svr_r2:
        print("\nGradient Boosting wins, retraining final model...")
        return train_final_model(X, y, gb_pipe, gb_param_grid, "Gradient_Boosting", search_type="random", n_iter=30, output_dir=output_dir)
    else:
        print("\nSVR wins, retraining final model...")
        return train_final_model(X, y, svr_pipe, svr_param_grid, "SVR", search_type="grid", output_dir=output_dir)

# Legacy entrypoint
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train ML models for tool wear prediction")
    parser.add_argument("--data", required=True, help="Directory containing data files")
    parser.add_argument("--lifetime", type=float, default=3600, help="Tool lifetime in seconds")
    parser.add_argument("--output", default="models", help="Output directory for models")
    args = parser.parse_args()
    run_ml_pipeline(args.data, args.lifetime, args.output)
