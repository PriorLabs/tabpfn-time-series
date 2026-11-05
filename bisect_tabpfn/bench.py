"""Time TabPFN fit+predict on the explainability example's matrices."""

import statistics
import time
from pathlib import Path

import pandas as pd
import tabpfn
import torch
from tabpfn import TabPFNRegressor

DATA = Path(__file__).parent / "data"
REPEATS = 3


def _load(name):
    train_X = pd.read_parquet(DATA / f"{name}_train_X.parquet")
    train_y = pd.read_parquet(DATA / f"{name}_train_y.parquet")["target"]
    test_X = pd.read_parquet(DATA / f"{name}_test_X.parquet")
    return train_X, train_y, test_X


def _fit_predict(train_X, train_y, test_X):
    model = TabPFNRegressor(device="cpu")
    model.fit(train_X, train_y)
    model.predict(test_X, output_type="main")


def main():
    print(
        f"tabpfn {tabpfn.__version__} torch {torch.__version__} threads {torch.get_num_threads()}"
    )
    matrices = {name: _load(name) for name in ["pdp", "shap"]}
    _fit_predict(*matrices["pdp"])
    for name, args in matrices.items():
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            _fit_predict(*args)
            times.append(time.perf_counter() - start)
        print(
            f"RESULT {name} rows={len(args[2])} median={statistics.median(times):.1f}s all={[round(t, 1) for t in times]}"
        )


if __name__ == "__main__":
    main()
