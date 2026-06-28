"""LSL marker helper with a no-LSL fallback."""

from __future__ import annotations

import logging
import time

LOGGER = logging.getLogger(__name__)


class MarkerOutlet:
    def __init__(self, stream_name: str = "MetaSleepGuardMarkers") -> None:
        self.stream_name = stream_name
        self.outlet = None
        self._lsl_clock = time.time
        self.last_wall_time = 0.0
        try:
            from pylsl import StreamInfo, StreamOutlet, local_clock

            info = StreamInfo(stream_name, "Markers", 1, 0, "string", "metasleep-guard")
            self.outlet = StreamOutlet(info)
            self._lsl_clock = local_clock
        except Exception:
            LOGGER.warning("pylsl is unavailable; marker events will be logged only")

    def push(self, marker: str) -> float:
        self.last_wall_time = time.time()
        timestamp = float(self._lsl_clock())
        if self.outlet is not None:
            self.outlet.push_sample([marker], timestamp)
        LOGGER.info("Marker %s at LSL %.3f / Unix %.3f", marker, timestamp, self.last_wall_time)
        return timestamp
