from tabpfn_time_series import (
    TimeSeriesDataFrame,
    TabPFNTimeSeriesPredictor,
    TabPFNMode,
)
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_CONFIG


class TabPFNNoisyTranformPredictor(TabPFNTimeSeriesPredictor):
    def __init__(
        self,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        noise_level: float = 1.0,
    ) -> None:
        self.config = TABPFN_TS_DEFAULT_CONFIG
        self.noise_level = noise_level

        # # disable y_preprocess in TabPFN
        # self.config["tabpfn_internal"]["inference_config"] = {
        #     "REGRESSION_Y_PREPROCESS_TRANSFORMS": [None]
        # }

        super().__init__(tabpfn_mode, self.config)

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        original_train_tsdf = train_tsdf.copy()

        # Normalize target (must be done before adding noise to be scale-invariant)
        train_tsdf, _ = self.target_normalization_transform(
            train_tsdf=original_train_tsdf,
            pred_tsdf=None,
            target_column_name="target",
            inverse=False,
        )

        # Add noise to train target
        train_tsdf = self.noise_transform(
            tsdf=train_tsdf,
            index_column_name="running_index",
            target_column_name="target",
            inverse=False,
            noise_amplitude=self.noise_level,
        )

        # Predict on noisy train target
        noisy_pred = super().predict(train_tsdf, test_tsdf)

        # Add running index to noisy_pred
        noisy_pred["running_index"] = test_tsdf["running_index"]

        # Denoise prediction
        denoised_pred = self.noise_transform(
            tsdf=noisy_pred,
            index_column_name="running_index",
            target_column_name="target",
            inverse=True,
            noise_amplitude=self.noise_level,
        )

        # Remove running index from denoised_pred
        denoised_pred = denoised_pred.drop(columns=["running_index"])

        # Denormalize target
        _, denoised_pred = self.target_normalization_transform(
            train_tsdf=original_train_tsdf,
            pred_tsdf=denoised_pred,
            target_column_name="target",
            inverse=True,
        )

        return denoised_pred

    @staticmethod
    def noise_transform(
        tsdf: TimeSeriesDataFrame,
        index_column_name: str,
        target_column_name: str,
        inverse: bool = False,
        noise_amplitude: float = 1.0,
    ):
        """
        Apply or remove systematic noise to data based on index parity:
        - For even indices: add noise_amplitude to the value (or subtract if inverse=True)
        - For odd indices: subtract noise_amplitude from the value (or add if inverse=True)

        Args:
            tsdf: TimeSeriesDataFrame
            index_column_name: str, name of the index column
            target_column_name: str, name of the target column
            inverse: bool, if True removes the noise instead of adding it

        Returns:
            numpy array with systematic noise applied or removed
        """

        assert index_column_name in tsdf.columns
        assert target_column_name in tsdf.columns

        index_values = tsdf[index_column_name].values

        # Vectorized operation based on index parity
        even_mask = index_values % 2 == 0
        odd_mask = ~even_mask

        # Apply noise based on parity and inverse flag
        if inverse:
            tsdf.loc[even_mask, target_column_name] -= noise_amplitude
            tsdf.loc[odd_mask, target_column_name] += noise_amplitude
        else:
            tsdf.loc[even_mask, target_column_name] += noise_amplitude
            tsdf.loc[odd_mask, target_column_name] -= noise_amplitude

        return tsdf

    @staticmethod
    def target_normalization_transform(
        train_tsdf: TimeSeriesDataFrame,
        pred_tsdf: TimeSeriesDataFrame = None,
        target_column_name: str = "target",
        inverse: bool = False,
    ):
        # Calculate mean and std of target column in train_tsdf
        target_mean = train_tsdf[target_column_name].groupby("item_id").mean()
        target_std = train_tsdf[target_column_name].groupby("item_id").std()

        new_train_tsdf = train_tsdf.copy()
        new_pred_tsdf = pred_tsdf.copy() if pred_tsdf is not None else None

        if not inverse:
            assert pred_tsdf is None, "We don't normalize prediction"
            # Normalize train_tsdf
            new_train_tsdf[target_column_name] = (
                new_train_tsdf[target_column_name] - target_mean
            ) / target_std

        else:
            # Denormalize train_tsdf and pred_tsdf
            new_train_tsdf[target_column_name] = (
                new_train_tsdf[target_column_name] * target_std + target_mean
            )

            # Apply to all columns of pred_tsdf
            for col in new_pred_tsdf.columns:
                new_pred_tsdf[col] = new_pred_tsdf[col] * target_std + target_mean

        return new_train_tsdf, new_pred_tsdf
