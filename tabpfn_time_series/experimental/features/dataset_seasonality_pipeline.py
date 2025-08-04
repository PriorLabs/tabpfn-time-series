from typing import Iterator, List
from collections import Counter
from joblib import Parallel, delayed
import logging

from gluonts.itertools import batcher
from gluonts.model.forecast import Forecast

from tabpfn_time_series.experimental.pipeline import (
    PipelineConfig,
    TimeSeriesEvalPipeline,
)
from tabpfn_time_series.experimental.features import (
    FeatureTransformer,
    AutoSeasonalFeature,
    PeriodicSinCosineFeature,
)

logger = logging.getLogger(__name__)


class DatasetSeasonalityPipeline(TimeSeriesEvalPipeline):
    def __init__(
        self,
        config: PipelineConfig,
        ds_prediction_length: int,
        ds_freq: str,
        debug: bool = False,
    ):
        super().__init__(
            config=config,
            ds_prediction_length=ds_prediction_length,
            ds_freq=ds_freq,
            debug=debug,
        )

        per_time_series_auto_seasonal_feature_config = config.features[
            "AutoSeasonalFeature"
        ]["config"]
        self.per_dataset_auto_seasonal_feature_config = (
            per_time_series_auto_seasonal_feature_config.copy()
        )
        self.per_dataset_auto_seasonal_feature_config["max_top_k"] = 10

    def predict(self, test_data_input) -> Iterator[Forecast]:
        most_common_seasonalities = self._get_dataset_seasonality(test_data_input)

        # Extend the feature_transformer to also generate the features for the most common seasonalities
        self.feature_transformer = FeatureTransformer(
            self.selected_features
            + [
                PeriodicSinCosineFeature(
                    periods=most_common_seasonalities,
                    name_suffix="common",
                )
            ]
        )

        return super().predict(test_data_input)

    def _get_dataset_seasonality(
        self,
        test_data_input,
        most_common_k: int = 3,
    ) -> List[float]:
        all_detected_seasonalities = []
        for batch in batcher(test_data_input, batch_size=1024):
            tsdf = self.convert_to_timeseries_dataframe(batch)
            tsdf = self.handle_nan_values(tsdf)

            detected_seasonalities = Parallel(
                n_jobs=16,
                backend="loky",
            )(
                delayed(AutoSeasonalFeature.find_seasonal_periods)(
                    tsdf.loc[item_id].target,
                    **self.per_dataset_auto_seasonal_feature_config,
                )
                for item_id in tsdf.item_ids
            )
            all_detected_seasonalities.extend(
                [x[0] for sublist in detected_seasonalities for x in sublist]
            )

        if self.debug:
            # Save the detected seasonalities to a CSV file
            import csv

            with open("detected_seasonalities.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["seasonality"])  # Header
                for seasonality in all_detected_seasonalities:
                    writer.writerow([seasonality])

        # Choose the most common seasonalities
        most_common_seasonalities_w_counts = Counter(
            all_detected_seasonalities
        ).most_common(most_common_k)
        logger.info(f"Most common seasonalities: {most_common_seasonalities_w_counts}")
        assert len(most_common_seasonalities_w_counts) == most_common_k

        return [seasonality for seasonality, _ in most_common_seasonalities_w_counts]
