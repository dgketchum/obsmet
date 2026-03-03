"""QAQC pipeline engine (plan section 11.2).

Runs observations through a chain of QC rules, collecting results.
Stores both machine-decision outputs and original native flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
