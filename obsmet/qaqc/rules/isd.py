"""Tier 0: ISD source-native QC rules (per-variable quality codes).

ISD quality codes are single-character flags attached to each measurement.
Good codes are "1" (passed all checks) and "5" (estimated/high-confidence).
Codes "2" (suspect), "9" (missing), and others are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier
from obsmet.sources.isd.extract import GOOD_QC

# Human-readable labels for common QC codes
_QC_CODE_LABELS = {
    "0": "not_checked",
    "1": "passed_all",
    "2": "suspect",
    "3": "erroneous",
    "4": "corrected",
    "5": "estimated",
    "6": "exception",
    "9": "missing",
}


@dataclass
class IsdQualityCodeRule(QCRule):
    """Tier 0: Interpret ISD per-variable quality codes.

    ISD attaches a single-character quality code to each measured variable.
    This rule translates those codes into obsmet QCResult objects.

    Good codes (pass): "1" (passed all QC checks), "5" (estimated value).
    Reject codes: "2" (suspect), "9" (missing/not available), others.
    """

    tier: QCTier = QCTier.SOURCE_NATIVE
    name: str = "isd_quality_code"
    version: str = "1"

    def check(self, value: float, **context) -> QCResult:
        """Check ISD quality code from context['qc_code'].

        Parameters
        ----------
        value : The observation value.
        **context : Must include 'qc_code' (single-character string).

        Returns
        -------
        QCResult with state pass/fail/suspect.
        """
        qc_code = context.get("qc_code", "")

        if not qc_code or qc_code == "":
            return QCResult(
                state="suspect",
                reason="qc_missing",
                tier=self.tier,
                rule_name=self.name,
                detail="No quality code present",
            )

        if qc_code in GOOD_QC:
            label = _QC_CODE_LABELS.get(qc_code, qc_code)
            return QCResult(
                state="pass",
                reason=f"qc_{label}",
                tier=self.tier,
                rule_name=self.name,
            )

        label = _QC_CODE_LABELS.get(qc_code, f"code_{qc_code}")
        return QCResult(
            state="fail",
            reason=f"qc_rejected_{label}",
            tier=self.tier,
            rule_name=self.name,
            detail=f"QC code {qc_code!r} not in good set {GOOD_QC}",
        )
