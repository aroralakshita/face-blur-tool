"""
Benchmark recording and reporting utility.

Collects FPS measurements during a benchmark run and computes
summary statistics. Separated from main.py so it can be tested
independently and reused in other contexts.

Why statistics matter here:
- Average FPS alone is misleading. A tool that runs at 28fps average
  but drops to 5fps occasionally feels broken.
- Min FPS reveals worst-case performance (what the user actually experiences).
- Max FPS shows the ceiling — useful for comparing configurations.
"""

import statistics
from dataclasses import dataclass, field


@dataclass
class BenchmarkRecorder:
    """Collects FPS samples and computes summary statistics.

    Usage:
        recorder = BenchmarkRecorder()
        for frame in frames:
            process(frame)
            recorder.record(fps_counter.get_fps())
        recorder.print_results("1280x720")
    """

    _samples: list[float] = field(default_factory=list)

    def record(self, fps: float) -> None:
        """Record a single FPS measurement.

        Args:
            fps: Current FPS reading. Values <= 0 are ignored
                 (occur during warmup before enough frames are counted).
        """
        if fps > 0:
            self._samples.append(fps)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def avg_fps(self) -> float:
        """Mean FPS across all samples."""
        if not self._samples:
            return 0.0
        return statistics.mean(self._samples)

    def min_fps(self) -> float:
        """Worst-case FPS (lowest recorded value)."""
        if not self._samples:
            return 0.0
        return min(self._samples)

    def max_fps(self) -> float:
        """Best-case FPS (highest recorded value)."""
        if not self._samples:
            return 0.0
        return max(self._samples)

    def p5_fps(self) -> float:
        """5th percentile FPS — a more realistic 'minimum' than absolute min.

        The absolute minimum can be a single outlier spike (e.g. OS interrupting
        the process). P5 gives a better picture of sustained low performance.
        """
        if len(self._samples) < 20:
            return self.min_fps()
        sorted_samples = sorted(self._samples)
        idx = max(0, int(len(sorted_samples) * 0.05) - 1)
        return sorted_samples[idx]

    def print_results(self, resolution: str) -> None:
        """Print benchmark results in a formatted, copy-paste friendly layout.

        The output format is designed to be directly usable in README tables.

        Args:
            resolution: Display string for resolution, e.g. "1280x720"
        """
        if not self._samples:
            print("No benchmark data collected.")
            return

        print()
        print("=" * 40)
        print("  Benchmark Results")
        print("=" * 40)
        print(f"  Resolution:  {resolution}")
        print(f"  Samples:     {self.sample_count} frames")
        print(f"  Avg FPS:     {self.avg_fps():.1f}")
        print(f"  Min FPS:     {self.min_fps():.1f}")
        print(f"  Max FPS:     {self.max_fps():.1f}")
        print(f"  P5 FPS:      {self.p5_fps():.1f}  (5th percentile)")
        print("=" * 40)
        print()
        print("  README table row:")
        print(f"  | {resolution} | {self.avg_fps():.0f} | {self.min_fps():.0f} | {self.max_fps():.0f} |")
        print()