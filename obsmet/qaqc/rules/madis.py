"""Tier 0: MADIS source-native QC rules (DD flags and QCR bitmask).

These rules interpret MADIS's own quality control outputs within the
obsmet QAQC framework.  They do not apply independent checks — they
translate the source's decisions into obsmet QCResult objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier
from obsmet.sources.madis.extract import ACCEPTABLE_DD, QCR_REJECT_BITS


@dataclass
class MadisDDRule(QCRule):
    """Tier 0: Interpret MADIS Data Descriptor (DD) flag.

    MADIS DD flags encode summary QC status:
    - V = Verified (passed all 3 MADIS levels)
    - S = Screened (passed levels 1-2)
    - C = Coarse pass (passed level 1 only)
    - G = Subjectively good
    - Q = Questioned (failed L2/L3) → REJECT
    - X = Excluded (failed L1) → REJECT
    - B = Subjectively bad → REJECT
    - Z = No QC applied → REJECT
    """

    tier: QCTier = QCTier.SOURCE_NATIVE
    name: str = "madis_dd"
    version: str = "1"

    def check(self, value: float, **context) -> QCResult:
        """Check DD flag from context['dd_flag']."""
        dd_flag = context.get("dd_flag", "")

        if not dd_flag or dd_flag == "":
            return QCResult(
                state="suspect",
                reason="dd_missing",
                tier=self.tier,
                rule_name=self.name,
                detail="No DD flag present",
            )

        if dd_flag in ACCEPTABLE_DD:
            return QCResult(
                state="pass",
                reason=f"dd_{dd_flag.lower()}",
                tier=self.tier,
                rule_name=self.name,
            )

        return QCResult(
            state="fail",
            reason=f"dd_rejected_{dd_flag.lower()}",
            tier=self.tier,
            rule_name=self.name,
            detail=f"DD flag {dd_flag!r} not in acceptable set {ACCEPTABLE_DD}",
        )


@dataclass
class MadisQCRRule(QCRule):
    """Tier 0: Interpret MADIS QCR bitmask.

    QCR bitmask structure:
    - Bit 1 (1):  Master check (at least one failed)
    - Bit 2 (2):  Validity check (Level 1 bounds)
    - Bit 3 (4):  [reserved]
    - Bit 4 (8):  Internal consistency (e.g., Td > T)
    - Bit 5 (16): Temporal consistency
    - Bit 6 (32): Statistical spatial consistency
    - Bit 7 (64): Spatial buddy check (OI-based)
    """

    tier: QCTier = QCTier.SOURCE_NATIVE
    name: str = "madis_qcr"
    version: str = "1"
    reject_mask: int = QCR_REJECT_BITS

    def check(self, value: float, **context) -> QCResult:
        """Check QCR bitmask from context['qcr_value']."""
        qcr = context.get("qcr_value")

        if qcr is None:
            return QCResult(
                state="pass",
                reason="qcr_missing",
                tier=self.tier,
                rule_name=self.name,
                detail="No QCR value present; passing by default",
            )

        qcr = int(qcr)
        failed_bits = qcr & self.reject_mask

        if failed_bits == 0:
            return QCResult(
                state="pass",
                reason="qcr_clear",
                tier=self.tier,
                rule_name=self.name,
            )

        # Decode which checks failed
        bit_names = {
            1: "master",
            2: "validity",
            4: "reserved",
            8: "internal_consistency",
            16: "temporal",
            32: "stat_spatial",
            64: "buddy",
        }
        failures = [bit_names.get(bit, f"bit_{bit}") for bit in bit_names if failed_bits & bit]

        return QCResult(
            state="fail",
            reason="qcr_reject",
            tier=self.tier,
            rule_name=self.name,
            detail=f"QCR={qcr} (0b{qcr:08b}), failed: {', '.join(failures)}",
        )
