from __future__ import annotations

import pandas as pd


def last_gw_baseline(df: pd.DataFrame) -> pd.Series:
    """Use the previous Gameweek's points."""
    return df["pts_lag1"]


def three_gw_baseline(df: pd.DataFrame) -> pd.Series:
    """Use the player's rolling 3-GW form."""
    return df["form_3gw"]


def five_gw_baseline(df: pd.DataFrame) -> pd.Series:
    """Use the player's rolling 5-GW form."""
    return df["form_5gw"]


def exponential_form_baseline(df: pd.DataFrame) -> pd.Series:
    """Use exponentially weighted recent form."""
    return df["exp_form"]


BASELINES = {
    "last_gw": last_gw_baseline,
    "form_3gw": three_gw_baseline,
    "form_5gw": five_gw_baseline,
    "exp_form": exponential_form_baseline,
}
