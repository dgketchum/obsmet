"""QAQC pipeline engine (plan section 11.2).

Runs observations through a chain of QC rules, collecting results.
Stores both machine-decision outputs and original native flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from obsmet.qaqc.rules.base import QCResult, QCRule


@dataclass
class QCPipeline:
    """Ordered chain of QC rules applied to observations."""

    rules: list[QCRule] = field(default_factory=list)

    def add_rule(self, rule: QCRule) -> None:
        self.rules.append(rule)

    def run(self, value: float, **context) -> list[QCResult]:
        """Apply all rules in order. Returns list of QCResult."""
        results = []
        for rule in self.rules:
            result = rule.check(value, **context)
            results.append(result)
        return results

    @staticmethod
    def aggregate_state(results: list[QCResult]) -> str:
        """Determine final qc_state from a list of rule results.

        Precedence: fail > suspect > pass.
        """
        states = {r.state for r in results}
        if "fail" in states:
            return "fail"
        if "suspect" in states:
            return "suspect"
        return "pass"

    @staticmethod
    def reason_codes(results: list[QCResult]) -> list[str]:
        """Collect non-pass reason codes from results."""
        return [r.reason for r in results if r.state != "pass"]


# --------------------------------------------------------------------------- #
# QC profiles
# --------------------------------------------------------------------------- #

QC_PROFILES: dict[str, dict] = {
    "strict": {"qcr_mask": 115, "description": "Full MADIS QC (operational)"},
    "permissive": {"qcr_mask": 2, "description": "Validity-only (training data)"},
}


# --------------------------------------------------------------------------- #
# Variable columns and MADIS reverse mapping
# --------------------------------------------------------------------------- #

_VARIABLE_COLUMNS: dict[str, list[str]] = {
    "madis": ["tmax", "tmin", "tmean", "td", "rh", "wind", "wind_dir", "prcp", "rsds"],
    "isd": ["tair", "td", "wind", "wind_dir", "slp", "prcp"],
    "ghcnh": ["tmax", "tmin", "tmean", "td", "wind", "wind_dir", "slp", "psfc", "prcp", "rh"],
    "ghcnd": ["tmax", "tmin", "tmean", "prcp", "snow", "snow_depth", "wind", "swe"],
    "gdas": ["tair", "td", "wind", "wind_dir", "psfc", "prcp"],
    "raws": ["tmean", "tmax", "tmin", "wind", "wind_dir", "rh", "prcp", "rsds"],
    "ndbc": ["tair", "td", "wind", "wind_dir", "slp"],
    "snotel": ["tmax", "tmin", "tmean", "prcp", "swe", "rh", "wind"],
}

# Maps canonical variable names back to MADIS native names for DD/QCR column lookup
_MADIS_VAR_REVERSE: dict[str, str] = {
    "tair": "temperature",
    "td": "dewpoint",
    "rh": "relHumidity",
    "wind": "windSpeed",
    "wind_dir": "windDir",
}

_GDAS_QM_COLUMNS: dict[str, str] = {
    "tair": "tair_qm",
    "td": "td_qm",
    "wind": "wind_qm",
    "wind_dir": "wind_qm",
    "psfc": "psfc_qm",
    "q": "q_qm",
    "u": "u_qm",
    "v": "v_qm",
}


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #


def build_default_pipeline(source: str, **kwargs) -> QCPipeline:
    """Build a default QC pipeline for a given source."""
    from obsmet.qaqc.rules.bounds import DewpointConsistencyRule, PhysicalBoundsRule

    pipeline = QCPipeline()

    if source == "madis":
        from obsmet.qaqc.rules.madis import MadisDDRule, MadisQCRRule

        pipeline.add_rule(MadisDDRule())
        qcr_mask = kwargs.get("qcr_mask")
        if qcr_mask is not None:
            pipeline.add_rule(MadisQCRRule(reject_mask=qcr_mask))
        else:
            pipeline.add_rule(MadisQCRRule())

    if source == "gdas":
        from obsmet.qaqc.rules.gdas import GdasQualityMarkerRule

        pipeline.add_rule(GdasQualityMarkerRule())

    pipeline.add_rule(PhysicalBoundsRule())
    pipeline.add_rule(DewpointConsistencyRule())
    return pipeline


# --------------------------------------------------------------------------- #
# DataFrame-level pipeline application
# --------------------------------------------------------------------------- #

_STATE_PRECEDENCE = {"fail": 2, "suspect": 1, "pass": 0}


def apply_pipeline_to_df(
    df: pd.DataFrame,
    pipeline: QCPipeline,
    variable_columns: list[str],
    *,
    source: str = "",
) -> pd.DataFrame:
    """Apply QC pipeline to a wide-form DataFrame, writing qc_state/qc_reason_codes columns.

    Iterates rows; for each variable column, runs pipeline.run() with appropriate context.
    Aggregates across all variables per row: worst state wins.
    """
    is_madis = source == "madis"
    is_gdas = source == "gdas"

    qc_states = []
    qc_reasons = []

    for row in df.itertuples(index=False):
        row_results: list[QCResult] = []

        for col in variable_columns:
            val = getattr(row, col, None)
            if val is None or pd.isna(val):
                continue

            ctx: dict = {"variable": col}

            # MADIS-specific: supply DD and QCR context
            if is_madis:
                native = _MADIS_VAR_REVERSE.get(col)
                if native:
                    dd_col = f"{native}DD"
                    qcr_col = f"{native}QCR"
                    dd_val = getattr(row, dd_col, None)
                    if dd_val is not None and not pd.isna(dd_val):
                        ctx["dd_flag"] = str(dd_val)
                    qcr_val = getattr(row, qcr_col, None)
                    if qcr_val is not None and not pd.isna(qcr_val):
                        ctx["qcr_value"] = int(qcr_val)

            if is_gdas:
                qm_col = _GDAS_QM_COLUMNS.get(col)
                if qm_col:
                    qm_val = getattr(row, qm_col, None)
                    if qm_val is not None and not pd.isna(qm_val):
                        ctx["qm"] = int(qm_val)

            # Dewpoint check needs tair context
            if col == "td":
                tair_val = getattr(row, "tair", None)
                if tair_val is not None and not pd.isna(tair_val):
                    ctx["tair"] = tair_val

            results = pipeline.run(float(val), **ctx)
            row_results.extend(results)

        if row_results:
            state = QCPipeline.aggregate_state(row_results)
            reasons = QCPipeline.reason_codes(row_results)
        else:
            state = "pass"
            reasons = []

        qc_states.append(state)
        qc_reasons.append(",".join(reasons) if reasons else "")

    df = df.copy()
    df["qc_state"] = qc_states
    df["qc_reason_codes"] = qc_reasons

    # Replace old qc_passed bool if present
    if "qc_passed" in df.columns:
        df = df.drop(columns=["qc_passed"])

    return df
