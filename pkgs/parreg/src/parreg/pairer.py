"""Base model of Pairer for donor-receiver pairing."""

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Pairer(BaseModel):
    """Base Pairer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: dict
    """Configuration for the Pairer."""

    df_attr_all: pd.DataFrame
    """DataFrame containing attributes for all donors and receivers."""

    dist_store_path: str | Path
    """Path to the spatial distance store DB file."""
