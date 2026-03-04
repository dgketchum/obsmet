"""RAWS WRCC response parser.

Parses the HTML/text response from WRCC CGI endpoint into a pandas
DataFrame with daily meteorological observations.

Ported from dads-mvp/extract/met_data/obs/raws_wrcc.py.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from obsmet.sources.raws_wrcc.download import DAILY_COLUMNS

# Missing value sentinel in WRCC data
_MISSING = -9999.0

# Number of expected columns
_N_COLS = len(DAILY_COLUMNS)


def parse_response(html: str) -> pd.DataFrame:
    """Parse WRCC HTML response into a daily DataFrame.

    Extracts data rows from <PRE> blocks, handles short rows by padding
    with missing values, and replaces -9999 sentinels with NaN.

    Returns DataFrame with DAILY_COLUMNS, indexed by date.
    """
    if not html:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    # Check for error responses
    if "Improper program call" in html or "Access to WRCC" in html:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    # Find data lines matching date pattern MM/DD/YYYY
    date_pat = re.compile(r"\d{2}/\d{2}/\d{4}")
    rows = []
    for line in html.split("\n"):
        line = line.strip()
        if date_pat.match(line):
            fields = line.split()
            # Pad short rows with missing values
            while len(fields) < _N_COLS:
                fields.append(str(int(_MISSING)))
            rows.append(fields[:_N_COLS])

    if not rows:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    df = pd.DataFrame(rows, columns=DAILY_COLUMNS)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    # Convert numeric columns
    numeric_cols = DAILY_COLUMNS[1:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace missing value sentinel
    df = df.replace(_MISSING, np.nan)

    # Drop invalid dates and duplicates
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df
