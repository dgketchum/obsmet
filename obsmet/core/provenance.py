"""Provenance tracking for obsmet pipeline runs.

Every curated dataset is stamped with schema_version, qaqc_rules_version,
crosswalk_version, and transform_version (plan section 23).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    return uuid.uuid4().hex[:12]


def file_hash(path: Path | str, algorithm: str = "sha256") -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunProvenance:
    """Provenance metadata attached to every pipeline run output."""

    run_id: str = field(default_factory=generate_run_id)
    started_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "0.1.0"
    qaqc_rules_version: str = "0.1.0"
    crosswalk_version: str = "0.1.0"
    transform_version: str = "0.1.0"
    source: str = ""
    command: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_utc": self.started_utc.isoformat(),
            "schema_version": self.schema_version,
            "qaqc_rules_version": self.qaqc_rules_version,
            "crosswalk_version": self.crosswalk_version,
            "transform_version": self.transform_version,
            "source": self.source,
            "command": self.command,
        }
