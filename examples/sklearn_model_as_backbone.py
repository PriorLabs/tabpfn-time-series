"""
Example: Using Sklearn Models as Inference Backbone

This example demonstrates how to use any standard sklearn regressor as the
inference backbone for time series forecasting. This is useful when you want to:
- Use traditional ML models (RandomForest, XGBoost, LightGBM, etc.)
- Have full control over fit and predict parameters
- Experiment with different model architectures
"""

from sklearn.ensemble import RandomForestRegressor

from tabpfn_time_series import FeatureTransformer, TimeSeriesPredictor
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

from common import create_sample_time_series


def main():
    print("\n" + "=" * 70)
    print("  🤖 Example: Sklearn Model as Inference Backbone")
    print("=" * 70)

    # Step 1: Create sample data
    print("\n📊 [Step 1/4] Creating sample time series data")
    print("-" * 70)
    train_tsdf, test_tsdf = create_sample_time_series()
    print(
        f"  ✓ Train data: {len(train_tsdf)} observations across {train_tsdf.num_items} items"
    )
    print(f"  ✓ Test data:  {len(test_tsdf)} timestamps")

    # Step 2: Create feature transformer
    print("\n🔧 [Step 2/4] Setting up feature engineering")
    print("-" * 70)
    feature_transformer = FeatureTransformer(
        [
            RunningIndexFeature(),  # Add time index
            CalendarFeature(),  # Add calendar features (day, month, weekday, etc.)
            AutoSeasonalFeature(),  # Add seasonal lag features
        ]
    )

    # Apply features to both train and test
    train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)
    print(f"  ✓ Generated {len(train_tsdf.columns)} feature columns")

    # Step 3: Create predictor with RandomForestRegressor
    print("\n🌲 [Step 3/4] Creating predictor with RandomForestRegressor")
    print("-" * 70)
    predictor = TimeSeriesPredictor.from_point_prediction_regressor(
        regressor_class=RandomForestRegressor,
        regressor_config={
            "n_estimators": 100,  # Number of trees
            "max_depth": 10,  # Maximum tree depth
            "min_samples_split": 5,  # Minimum samples to split
            "random_state": 42,  # For reproducibility
            "n_jobs": -1,  # Use all CPU cores
        },
        regressor_fit_config={
            # Add any fit-specific parameters here if needed
            # e.g., sample_weight, etc.
        },
        regressor_predict_config={
            # Add any predict-specific parameters here if needed
        },
    )
    print("  ✓ Predictor initialized")

    # Step 4: Make predictions
    print("\n🚀 [Step 4/4] Training model and making predictions")
    print("-" * 70)
    predictions = predictor.predict(train_tsdf, test_tsdf)
    print("  ✓ Predictions complete")

    # Display results
    print("\n" + "=" * 70)
    print("  📈 RESULTS")
    print("=" * 70)
    print(predictions.head(10))

    timestamps = predictions.index.get_level_values("timestamp")
    print("\n📋 Summary:")
    print(f"  • Total predictions: {len(predictions)}")
    print(f"  • Forecast range:    {timestamps.min()} to {timestamps.max()}")

    print("\n" + "=" * 70)
    print("  ✅ SUCCESS! Example completed.")
    print("=" * 70)
    print("\n💡 Tip: Replace RandomForestRegressor with any sklearn-compatible model")
    print("   (XGBoost, LightGBM, CatBoost, etc.) or use from_tabpfn_family().\n")


if __name__ == "__main__":
    main()
