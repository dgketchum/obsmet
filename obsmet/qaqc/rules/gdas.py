"""Tier 0: GDAS PrepBUFR source-native QC rules (Quality Marker interpretation).

GDAS PrepBUFR quality markers (QM) encode the outcome of NCEP's QC pipeline:
- QM 0-1: Good (passed all checks)
- QM 2: Neutral / passive
- QM 3: Suspect (passed complex QC, inflated error)
- QM 4-14: Rejected by various QC steps
- QM 15: Purged
"""

from __future__ import annotations

from dataclasses import dataclass

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier

# QM threshold: values <= this are usable
GOOD_QM_MAX = 3

_QM_LABELS = {
    0: "not_checked",
    1: "good",
    2: "neutral",
    3: "suspect",
    4: "reject_level1",
    5: "reject_level2",
    6: "reject_level3",
    7: "reject_level4",
    8: "reject_level5",
    9: "purged",
    10: "reject_level6",
    11: "reject_level7",
    12: "purged",
    13: "reject_level8",
    14: "reject_level9",
    15: "purged",
}


@dataclass
class GdasQualityMarkerRule(QCRule):
    """Tier 0: Interpret GDAS PrepBUFR quality markers.

    QM <= 3 is usable (pass for 0-2, suspect for 3).
    QM >= 4 is rejected.
    Missing QM passes by default.
    """

    tier: QCTier = QCTier.SOURCE_NATIVE
    name: str = "gdas_qm"
    version: str = "1"

    def check(self, value: float, **context) -> QCResult:
        """Check GDAS quality marker from context['qm'].

        Parameters
        ----------
        value : The observation value.
        **context : Must include 'qm' (integer quality marker).
        """
        qm = context.get("qm")

        if qm is None:
            return QCResult(
                state="pass",
                reason="qm_missing",
                tier=self.tier,
                rule_name=self.name,
                detail="No quality marker present; passing by default",
            )

        qm = int(qm)
        label = _QM_LABELS.get(qm, f"qm_{qm}")

        if qm <= 2:
            return QCResult(
                state="pass",
                reason=f"qm_{label}",
                tier=self.tier,
                rule_name=self.name,
            )

        if qm == 3:
            return QCResult(
                state="suspect",
                reason="qm_suspect",
                tier=self.tier,
                rule_name=self.name,
                detail="QM=3: passed complex QC but error inflated",
            )

        return QCResult(
            state="fail",
            reason=f"qm_rejected_{label}",
            tier=self.tier,
            rule_name=self.name,
            detail=f"QM={qm}: rejected by NCEP QC pipeline",
        )
