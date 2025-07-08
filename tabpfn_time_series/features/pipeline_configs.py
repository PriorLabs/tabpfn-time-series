from dataclasses import dataclass


@dataclass
class ColumnConfig:
    """
    Base class for column configuration.
    """

    timestamp_col_name: str = None
    target_col_name: str = None
    item_id_col_name: str = None


@dataclass
class DefaultColumnConfig(ColumnConfig):
    """
    Default column configuration.
    """

    timestamp_col_name: str = "timestamp"
    target_col_name: str = "target"
    item_id_col_name: str = "item_id"
