"""Same-boot Linux CLOCK_MONOTONIC job lifetime bound."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


def monotonic_clock_domain(*, peer_pid: int | None = None) -> str:
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        offsets = Path(f"/proc/{peer_pid}/timens_offsets" if peer_pid is not None
                       else "/proc/self/timens_offsets")
        offset = "monotonic 0 0"
        if offsets.exists():
            rows = [line.split() for line in offsets.read_text().splitlines()]
            monotonic = [row for row in rows if row and row[0] == "monotonic"]
            if len(monotonic) != 1 or len(monotonic[0]) != 3:
                raise ValueError("monotonic time namespace offset is unavailable")
            offset = " ".join(monotonic[0])
        else:
            raise ValueError("deadline route requires readable time namespace offsets")
    except OSError as exc:
        raise ValueError("host monotonic deadline requires a Linux boot clock identity") from exc
    if not boot:
        raise ValueError("host monotonic clock identity is unavailable")
    return hashlib.sha256(("linux:CLOCK_MONOTONIC:" + boot + ":" + offset).encode()).hexdigest()


@dataclass(frozen=True)
class HostDeadline:
    monotonic_s: float
    clock_domain: str

    def validate(self) -> None:
        if (isinstance(self.monotonic_s, bool)
                or not isinstance(self.monotonic_s, (int, float))
                or not math.isfinite(self.monotonic_s) or self.monotonic_s < 0):
            raise ValueError("deadline monotonic_s must be finite and nonnegative")
        if self.clock_domain != monotonic_clock_domain():
            raise ValueError("deadline clock domain differs from broker host boot")

    def value(self) -> dict:
        return {"monotonic_s": self.monotonic_s, "clock_domain": self.clock_domain}
