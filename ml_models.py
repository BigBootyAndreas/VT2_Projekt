import numpy as np
import pandas as pd
import joblib
import os
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_validate, TimeSeriesSplit, GroupKFold
)
from sklearn.metrics import (
    make_scorer, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.impute import SimpleImputer  

# Suppress specific warnings
warnings.filterwarnings('ignore', message='n_quantiles .* is greater than the total number of samples')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._data')

# Import the enhanced data preparation
from data_preparation import collect_dataset_from_directory, advanced_data_preprocessing

# Enhanced pipelines with NaN handling - using RobustScaler by default
# Pipeline with imputation, variance threshold, feature selection, and robust scaling
base_preprocessing_steps = [
    ("imputer", SimpleImputer(strategy='median')),  # Handle NaN values first
    ("variance_threshold", VarianceThreshold(threshold=0.01)),
    ("feature_selection", SelectKBest(f_regression, k=50)),
    ("scaler", RobustScaler())  # RobustScaler doesn't have n_quantiles issues
]

# Enhanced SVR pipeline with NaN handling - using RobustScaler
svr_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),  # Handle NaN values
    ("variance_threshold", VarianceThreshold(threshold=0.01)),
    ("feature_selection", SelectKBest(f_regression, k=30)),
    ("scaler", RobustScaler()),  
    ("svr", SVR())
])

# Enhanced SVR parameter grid
svr_param_grid = {
    "feature_selection__k": [30],
    "svr__kernel": ["rbf"],
    "svr__C": [1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.1],
    "svr__epsilon": [0.1, 0.2],
    "svr__degree": [3],
}

# Enhanced Gradient Boosting pipeline with NaN handling
gb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),  # Handle NaN values
    ("variance_threshold", VarianceThreshold(threshold=0.01)),
    ("feature_selection", SelectKBest(f_regression, k=50)),
    ("scaler", RobustScaler()),  
    ("gb", GradientBoostingRegressor(random_state=42))
])

# Enhanced GB parameter grid
gb_param_grid = {
    "feature_selection__k": [20],   
    "gb__n_estimators": [50, 100],  
    "gb__learning_rate": [0.01, 0.05, 0.1],  
    "gb__max_depth": [2, 3],  
    "gb__subsample": [0.7, 0.8],  
    "gb__min_samples_split": [10, 15],  
    "gb__min_samples_leaf": [5, 8],  
    "gb__max_features": ["sqrt"],  
    "gb__loss": ["squared_error", "huber"],  
}

# Setup enhanced cross-validation with group awareness
def create_cv_strategy(info_df=None, n_splits=5, stratified=True):
    """
    Create cross-validation strategy that respects sample grouping
    """
    if info_df is not None and 'sample_number' in info_df.columns:
        # Use GroupKFold to ensure samples don't leak between folds
        groups = info_df['sample_number'].values
        return GroupKFold(n_splits=n_splits), groups
    else:
        # Fallback to TimeSeriesSplit
        return TimeSeriesSplit(n_splits=n_splits), None

# Enhanced scoring metrics
scoring = {
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score),
    "RMSE": make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
}

def check_and_clean_data(X, y, verbose=True):
    """
    Check for and handle data quality issues
    """
    if verbose:
        print("=== Data Quality Check ===")
    
    # Check for NaN values
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        if verbose:
            print(f"Found {nan_count} NaN values")
            print("NaN columns:", X.columns[X.isnull().any()].tolist())
        
        # Fill NaN with median strategy
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        if verbose:
            print("NaN values filled with median")
    else:
        X_clean = X.copy()
        if verbose:
            print("No NaN values found")
    
    # Check for infinite values
    inf_count = np.isinf(X_clean.values).sum()
    if inf_count > 0:
        if verbose:
            print(f"Found {inf_count} infinite values")
        
        # Replace infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill the newly created NaN values
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
        
        if verbose:
            print("Infinite values replaced and filled")
    else:
        if verbose:
            print("No infinite values found")
    
    # Check target variable
    if y.isnull().any():
        if verbose:
            print("Found NaN in target variable, using median fill")
        y_clean = y.fillna(y.median())
    else:
        y_clean = y.copy()
        if verbose:
            print("Target variable is clean")
    
    # Final verification
    assert not X_clean.isnull().any().any(), "Still have NaN values after cleaning!"
    assert not np.isinf(X_clean.values).any(), "Still have infinite values after cleaning!"
    assert not y_clean.isnull().any(), "Still have NaN in target after cleaning!"
    
    if verbose:
        print("Data cleaning completed successfully")
    
    return X_clean, y_clean

def nested_cv_evaluate(X, y, pipe, param_grid, name="Model", search_type="grid", 
                      n_iter=20, info_df=None):
    """
    Perform nested cross-validation with enhanced error handling and data cleaning
    """
    # Clean data first
    X_clean, y_clean = check_and_clean_data(X, y, verbose=True)
    
    # Create appropriate CV strategy
    cv_strategy, groups = create_cv_strategy(info_df)
    inner_cv_strategy, _ = create_cv_strategy(info_df, n_splits=3)
    
    # Adjust feature selection k based on available features
    if 'feature_selection__k' in param_grid:
        max_features = X_clean.shape[1]
        adjusted_k_values = [k for k in param_grid['feature_selection__k'] if k <= max_features]
        if not adjusted_k_values:
            adjusted_k_values = [min(20, max_features)]
        param_grid = param_grid.copy()  # Don't modify original
        param_grid['feature_selection__k'] = adjusted_k_values
        print(f"Adjusted feature selection k values to: {adjusted_k_values}")
    
    # Choose search method
    if search_type == "random":
        search = RandomizedSearchCV(
            pipe, param_grid,
            cv=inner_cv_strategy,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42,
            verbose=1,
            error_score='raise'
        )
    else:
        search = GridSearchCV(
            pipe, param_grid,
            cv=inner_cv_strategy,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )
    
    # Run outer CV with error handling
    try:
        cv_results = cross_validate(
            search, X_clean, y_clean,
            groups=groups,
            cv=cv_strategy,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            verbose=1
        )
    except Exception as e:
        print(f"Error during cross-validation for {name}: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy results in case of error
        n_splits = cv_strategy.get_n_splits()
        return pd.DataFrame({
            "Fold": range(1, n_splits + 1),
            "Test MSE": [999999] * n_splits,
            "Test MAE": [999999] * n_splits,
            "Test R2": [-999999] * n_splits,
            "Test RMSE": [999999] * n_splits,
        })
    
    # Assemble summary with both training and test scores
    summary = pd.DataFrame({
        "Fold": range(1, cv_strategy.get_n_splits() + 1),
        "Train MSE": -cv_results["train_MSE"],
        "Test MSE": -cv_results["test_MSE"],
        "Train MAE": -cv_results["train_MAE"],
        "Test MAE": -cv_results["test_MAE"],
        "Train R2": cv_results["train_R2"],
        "Test R2": cv_results["test_R2"],
        "Train RMSE": -cv_results["train_RMSE"],
        "Test RMSE": -cv_results["test_RMSE"],
    })
    
    print(f"\n=== Nested CV results for {name} ===")
    print(summary)
    print(f"Mean Test MSE: {summary['Test MSE'].mean():.4f} (+/- {summary['Test MSE'].std()*2:.4f})")
    print(f"Mean Test MAE: {summary['Test MAE'].mean():.4f} (+/- {summary['Test MAE'].std()*2:.4f})")
    print(f"Mean Test R2:  {summary['Test R2'].mean():.4f} (+/- {summary['Test R2'].std()*2:.4f})")
    print(f"Mean Test RMSE: {summary['Test RMSE'].mean():.4f} (+/- {summary['Test RMSE'].std()*2:.4f})")
    
    # Check for overfitting
    train_r2_mean = summary['Train R2'].mean()
    test_r2_mean = summary['Test R2'].mean()
    overfitting_threshold = 0.2
    
    if train_r2_mean - test_r2_mean > overfitting_threshold:
        print(f"⚠️  Potential overfitting detected: Train R2 ({train_r2_mean:.3f}) >> Test R2 ({test_r2_mean:.3f})")
    
    return summary

# Fix for train_final_model function in ml_models.py

def train_final_model(X, y, pipe, param_grid, name="Model", 
                      search_type="grid", n_iter=20, output_dir="models", info_df=None):  
    """
    Train a final pipeline with enhanced preprocessing and data cleaning
    FIXED: Handle 'all' value in feature_selection__k parameter
    """
    # Clean data first
    print(f"Cleaning data for {name}...")
    X_clean, y_clean = check_and_clean_data(X, y, verbose=True)
    
    # FIXED: Better feature selection adjustment to handle 'all' value
    if 'feature_selection__k' in param_grid:
        max_features = X_clean.shape[1]
        original_k_values = param_grid['feature_selection__k']
        adjusted_k_values = []
        
        for k in original_k_values:
            if k == 'all':
                # 'all' means use all available features
                adjusted_k_values.append('all')
            elif isinstance(k, (int, float)):
                # For numeric values, ensure they don't exceed max_features
                if k <= max_features:
                    adjusted_k_values.append(int(k))
                else:
                    # If k is too large, use a reasonable fraction of available features
                    reasonable_k = min(max_features, max(10, int(max_features * 0.8)))
                    adjusted_k_values.append(reasonable_k)
                    print(f"Adjusted feature selection k from {k} to {reasonable_k} (max available: {max_features})")
            else:
                # Handle any other unexpected values
                print(f"Warning: Unexpected value in feature_selection__k: {k}, skipping")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        adjusted_k_values = [k for k in adjusted_k_values if k not in seen and not seen.add(k)]
        
        # Update the parameter grid
        param_grid = param_grid.copy()  # Don't modify original
        param_grid['feature_selection__k'] = adjusted_k_values
        print(f"Feature selection k values: {adjusted_k_values}")
    
    # Create CV strategy
    inner_cv_strategy, groups = create_cv_strategy(info_df, n_splits=3)
    
    # Build search with error handling
    try:
        if search_type == "random":
            search = RandomizedSearchCV(
                pipe, param_grid,
                cv=inner_cv_strategy,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                n_iter=n_iter,
                random_state=42,
                verbose=1,
                error_score='raise'
            )
        else:
            search = GridSearchCV(
                pipe, param_grid,
                cv=inner_cv_strategy,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1,
                error_score='raise'
            )
        
        # Fit on full data
        print(f"Training {name} on full dataset...")
        search.fit(X_clean, y_clean, groups=groups)
        final_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"Best {name} parameters: {best_params}")
        print(f"Best {name} cross-validation score: {-best_score:.4f}")
        
    except Exception as e:
        print(f"Error during training {name}: {e}")
        print("Falling back to default parameters...")
        import traceback
        traceback.print_exc()
        
        # Fallback to default parameters
        final_model = pipe
        print("Fitting with default parameters...")
        final_model.fit(X_clean, y_clean)
        best_params = "Default parameters due to error"
        best_score = 0
    
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    base = os.path.join(output_dir, name.lower().replace(' ', '_'))
    model_path = base + "_model.pkl"
    scaler_path = base + "_scaler.pkl"
    features_path = base + "_features.pkl"
    
    # Extract and save components
    try:
        # Find the actual estimator (not the preprocessing steps)
        estimator_names = [step for step in final_model.named_steps.keys() 
                         if step not in ['imputer', 'variance_threshold', 'feature_selection', 'scaler']]
        
        if estimator_names:
            estimator_name = estimator_names[0]
            actual_model = final_model.named_steps[estimator_name]
        else:
            # If no specific estimator found, save the whole pipeline
            actual_model = final_model
        
        # Save components
        joblib.dump(actual_model, model_path)
        
        # Save scaler (or imputer if scaler not found)
        if 'scaler' in final_model.named_steps:
            joblib.dump(final_model.named_steps['scaler'], scaler_path)
        elif 'imputer' in final_model.named_steps:
            # If no scaler, save imputer as scaler for compatibility
            joblib.dump(final_model.named_steps['imputer'], scaler_path)
        else:
            # Create a dummy scaler
            from sklearn.preprocessing import StandardScaler
            dummy_scaler = StandardScaler()
            dummy_scaler.fit(X_clean)
            joblib.dump(dummy_scaler, scaler_path)
        
        # Get feature names after preprocessing
        final_features = list(X_clean.columns)
        
        # Apply preprocessing steps to get final feature names
        if 'variance_threshold' in final_model.named_steps:
            vt = final_model.named_steps['variance_threshold']
            if hasattr(vt, 'get_support'):
                # Apply variance threshold to feature names
                final_features = [f for i, f in enumerate(final_features) if i < len(vt.get_support()) and vt.get_support()[i]]
        
        if 'feature_selection' in final_model.named_steps:
            fs = final_model.named_steps['feature_selection']
            if hasattr(fs, 'get_support'):
                # Apply feature selection to remaining feature names
                final_features = [f for i, f in enumerate(final_features) if i < len(fs.get_support()) and fs.get_support()[i]]
        
        joblib.dump(final_features, features_path)
        
    except Exception as e:
        print(f"Warning: Error extracting pipeline components: {e}")
        # Fallback: save the entire pipeline
        joblib.dump(final_model, model_path)
        joblib.dump(final_model, scaler_path)
        joblib.dump(list(X_clean.columns), features_path)
    
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved feature list to {features_path}")
    
    # Save training metadata
    metadata = {
        'model_name': name,
        'best_params': best_params,
        'best_score': best_score,
        'n_features': len(X_clean.columns),
        'n_samples': len(X_clean),
        'feature_selection_applied': 'feature_selection' in final_model.named_steps,
        'scaler_type': type(final_model.named_steps.get('scaler', None)).__name__,
        'data_cleaned': True
    }
    
    metadata_path = base + "_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")
    
    return model_path, scaler_path, features_path

def run_ml_pipeline(data_dir, tool_lifetime=3600, output_dir="models"):
    """
    Enhanced ML pipeline with better preprocessing and error handling
    """
    print("=== Enhanced ML Pipeline with NaN Handling ===")
    print(f"Data directory: {data_dir}")
    print(f"Tool lifetime: {tool_lifetime} seconds")
    print(f"Output directory: {output_dir}")
    
    # 1) Collect features & labels with enhanced preprocessing
    print("\n1. Collecting and preprocessing data...")
    try:
        result = collect_dataset_from_directory(data_dir, tool_lifetime)
        if len(result) == 5:
            X, y, scaler, feature_selector, preprocessing_info = result
            info_df = None
        else:
            X, y = result
            scaler = None
            feature_selector = None
            preprocessing_info = {}
            info_df = None
    except Exception as e:
        print(f"Error collecting data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Clean data immediately after collection
    print("\n2. Cleaning collected data...")
    X, y = check_and_clean_data(X, y, verbose=True)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print("Preprocessing info:", preprocessing_info)
    
    # 3) Enhanced nested CV for both models
    print("\n3. Performing enhanced nested cross-validation...")
    
    # SVR evaluation with enhanced preprocessing
    print("\n--- Evaluating Enhanced SVR ---")
    svr_summary = nested_cv_evaluate(
        X, y, svr_pipe, svr_param_grid, 
        "Enhanced_SVR", search_type="random", n_iter=30, info_df=info_df
    )
    
    # GB evaluation with enhanced preprocessing
    print("\n--- Evaluating Enhanced Gradient Boosting ---")
    gb_summary = nested_cv_evaluate(
        X, y, gb_pipe, gb_param_grid, 
        "Enhanced_Gradient_Boosting", search_type="random", n_iter=30, info_df=info_df
    )
    
    # 4) Choose best model and train final version
    print("\n4. Selecting best model and training final version...")
    svr_r2 = svr_summary['Test R2'].mean()
    gb_r2 = gb_summary['Test R2'].mean()
    
    print(f"\nFinal Results:")
    print(f"Enhanced SVR - Mean Test R2: {svr_r2:.4f}")
    print(f"Enhanced GB  - Mean Test R2: {gb_r2:.4f}")
    
    # Train both models and let user see results
    print("\n5. Training final models...")
    
    # Train SVR
    print("\n--- Training final Enhanced SVR model ---")
    try:
        svr_files = train_final_model(
            X, y, svr_pipe, svr_param_grid, 
            "Enhanced_SVR", search_type="random", n_iter=20, 
            output_dir=output_dir, info_df=info_df
        )
        print("✓ Enhanced SVR model completed")
    except Exception as e:
        print(f"✗ Enhanced SVR model failed: {e}")
        svr_files = None
    
    # Train GB
    print("\n--- Training final Enhanced Gradient Boosting model ---")
    try:
        gb_files = train_final_model(
            X, y, gb_pipe, gb_param_grid, 
            "Enhanced_Gradient_Boosting", search_type="random", n_iter=20, 
            output_dir=output_dir, info_df=info_df
        )
        print("✓ Enhanced Gradient Boosting model completed")
    except Exception as e:
        print(f"✗ Enhanced Gradient Boosting model failed: {e}")
        gb_files = None
    
    # Return the better performing model
    if gb_r2 > svr_r2 and gb_files:
        print(f"\n🏆 Enhanced Gradient Boosting wins with R² = {gb_r2:.4f}")
        return gb_files
    elif svr_files:
        print(f"\n🏆 Enhanced SVR wins with R² = {svr_r2:.4f}")
        return svr_files
    else:
        print("\n⚠️  Both models failed, check errors above")
        return None

# Legacy entrypoint
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced ML models for tool wear prediction")
    parser.add_argument("--data", required=True, help="Directory containing data files")
    parser.add_argument("--lifetime", type=float, default=3600, help="Tool lifetime in seconds")
    parser.add_argument("--output", default="models", help="Output directory for models")
    args = parser.parse_args()
    
    result = run_ml_pipeline(args.data, args.lifetime, args.output)
    if result:
        print(f"\n✓ Pipeline completed successfully!")
        print(f"Model files: {result}")
    else:
        print(f"\n✗ Pipeline failed!")
