"""utils package with various utility/helper functions."""

from .config_utils import (
    BaseConfig,
    BaseConfigProcessor,
    BaseGeneralConfig,
    BaseOutputConfig,
    LoggingConfig,
)
from .dict_utils import convert_enum_to_value, flatten_dict, remove_nulls
from .hydrofabric_utils import DistanceStore, area_weighted_average, find_gages_within_buffer
from .io_utils import read_table, save_data
from .logging_utils import setup_logging
from .plot_utils import plot_histogram, plot_spatial_map
from .string_utils import expand_with_lists, recursive_substitute, recursive_substitute_multi_lists
from .validation_utils import check_columns_dataframe, check_columns_hydrofabric, check_options

__all__ = [
    "BaseConfig",
    "BaseGeneralConfig",
    "BaseOutputConfig",
    "BaseConfigProcessor",
    "LoggingConfig",
    "remove_nulls",
    "convert_enum_to_value",
    "flatten_dict",
    "find_gages_within_buffer",
    "area_weighted_average",
    "DistanceStore",
    "read_table",
    "save_data",
    "setup_logging",
    "plot_histogram",
    "plot_spatial_map",
    "expand_with_lists",
    "recursive_substitute",
    "recursive_substitute_multi_lists",
    "check_columns_dataframe",
    "check_columns_hydrofabric",
    "check_options",
]
