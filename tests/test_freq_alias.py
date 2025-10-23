"""Simple tests for alias canonicalization and prediction_length.

The expected values are derived from the ground-truth maps in gift_eval.data
after applying the same normalization used in production code, so this works
across pandas versions.
"""

from unittest.mock import MagicMock

from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import norm_freq_str
from gift_eval.data import (
    NEW_PANDAS_TS_ALIASES,
    PRED_LENGTH_MAP,
    M4_PRED_LENGTH_MAP,
    Term,
    Dataset,
)


def test_new_pandas_aliases_mapping_simple():
    # Ensure expected new/alt aliases map to legacy keys we use
    expected_mappings = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    for alias, expected in expected_mappings.items():
        assert NEW_PANDAS_TS_ALIASES[alias] == expected


def test_legacy_aliases_unchanged_simple():
    # Legacy aliases should pass through unchanged by the mapping
    for alias in ["H", "D", "W", "M", "Q", "A", "T", "S"]:
        assert NEW_PANDAS_TS_ALIASES.get(alias, alias) == alias


def test_prediction_length_minimal():
    def make_ds(freq: str, name: str = "regular", term: Term = Term.SHORT) -> Dataset:
        mock_ds = MagicMock(spec=Dataset)
        mock_ds.freq = freq
        mock_ds.name = name
        mock_ds.term = term

        # The property under test needs to be attached to the class for `cached_property` to work
        type(mock_ds).prediction_length = Dataset.prediction_length
        return mock_ds

    def expected_for(freq: str, use_m4: bool = False) -> int:
        normalized = norm_freq_str(to_offset(freq).name)
        key = NEW_PANDAS_TS_ALIASES.get(normalized, normalized)
        mapping = M4_PRED_LENGTH_MAP if use_m4 else PRED_LENGTH_MAP
        return mapping[key]

    # Regular map checks (PRED_LENGTH_MAP)
    for f in ["H", "h", "T", "min", "D", "d", "W", "M", "S", "s"]:
        assert make_ds(f).prediction_length == expected_for(f)

    # M4 map checks (M4_PRED_LENGTH_MAP)
    for f in ["H", "D", "W", "M", "Q", "A", "Y"]:
        assert make_ds(f, name="m4").prediction_length == expected_for(f, use_m4=True)
