"""
Example: Using TabPFN-Family Models as Inference Backbone

This example demonstrates how to use any TabPFN-family model as the
inference backbone for time series forecasting. This is useful when you want to:
- Use TabPFNRegressor from tabpfn-client (API-based)
- Customize TabPFN parameters
- Control output selection (mean, median, etc.)
"""

from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor

from tabpfn_time_series import FeatureTransformer, TimeSeriesPredictor
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

from common import create_sample_time_series


def main():
    print("\n" + "=" * 70)
    print("  🧠 Example: TabPFN-Family Model as Inference Backbone")
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

    # Step 3: Create predictor with custom TabPFN model
    print("\n⚡ [Step 3/4] Creating predictor with TabPFNRegressor")
    print("-" * 70)
    predictor = TimeSeriesPredictor.from_tabpfn_family(
        tabpfn_class=TabPFNClientRegressor,
        tabpfn_config={
            "n_estimators": 8,  # Number of ensemble members
        },
        tabpfn_output_selection="median",  # Use median of ensemble
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
    print("\n💡 Tip: Replace TabPFNRegressor with other TabPFN-family models or")
    print("   use from_point_prediction_regressor() for sklearn models.\n")


if __name__ == "__main__":
    main()
