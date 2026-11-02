"""Recorded sample clocks: distinguish packet delivery jitter from missing data."""
import numpy as np
from scipy.ndimage import median_filter


def estimated_rate(timestamps):
    """Estimate over long sample spans, not the tiny deltas inside USB bursts."""
    t = np.asarray(timestamps, dtype=float)
    valid = np.flatnonzero(np.isfinite(t))
    if valid.size < 32:
        return None
    lag = min(1000, max(125, valid.size // 10), valid.size - 1)
    # Limit work on large recordings without biasing towards the first segment.
    starts = np.linspace(0, valid.size - lag - 1, min(4096, valid.size - lag), dtype=int)
    left, right = valid[starts], valid[starts + lag]
    elapsed = t[right] - t[left]
    keep = elapsed > 0
    if not keep.any():
        return None
    return float(np.median((right[keep] - left[keep]) / elapsed[keep]))


def _clock_shift(t, fs, padded=None, seconds=2.0):
    """row -> samples the host clock says are missing just before it, or None.

    `padded` counts samples already inserted before each row. Medians over a
    couple of seconds on each side ride out USB bursts.
    """
    residual = t - (np.arange(t.size) + (0 if padded is None else padded)) / fs
    width = max(4, int(round(seconds * fs)))

    def at(row):
        before = residual[max(0, row - width):row]
        after = residual[row:row + width]
        before, after = before[np.isfinite(before)], after[np.isfinite(after)]
        if not before.size or not after.size:
            return None
        return float(np.median(after) - np.median(before)) * fs
    return at


def sample_positions(timestamps, fs, counters=None, counter_step=None):
    """Original rows -> padded sample indices. No signal is interpolated.

    Cyton packet counters detect short losses; clock residuals detect long
    pauses beyond a counter wrap. USB delivery bursts alone must not add data.
    Missing rows already present in the file count towards the sample clock.
    """
    t = np.asarray(timestamps, dtype=float)
    n = t.size
    missing = np.zeros(n, dtype=np.int64)
    if counters is not None and counter_step is not None:
        c = np.asarray(counters, dtype=float)
        valid = np.flatnonzero(np.isfinite(c) & (c >= 0) & (c < 256) & (c == np.floor(c)))
        left, right = valid[:-1], valid[1:]
        delta = np.mod(c[right] - c[left], 256)
        expected = (right - left) * counter_step
        # Uninitialised/placeholder counter columns carry no timing evidence.
        if delta.size and np.mean(delta == np.mod(expected, 256)) > .8:
            # RF glitches repeat or reorder packets (172 -> 168 reads as a
            # 252 step): a step back is disorder, not 126 lost samples. Count
            # the net counter progress, and only losses nothing later undoes.
            signed = np.mod(delta + 128, 256) - 128
            # A step back reads the same as a real loss of more than half a
            # counter cycle (Daisy: 64-127 samples). The wall clock around the
            # step picks the reading: nothing missing, or the loss plus any
            # whole cycles the counter wrapped through.
            # Steps are decided in order against the clock as padded so far,
            # so a cluster of garbled packets around one loss counts it once.
            cycle = 256 // counter_step
            extra = signed / counter_step - (right - left)     # samples each step adds
            for j in np.flatnonzero(signed < 0):
                added = np.zeros(n)
                added[right] = extra
                missing_now = _clock_shift(t, fs, np.cumsum(added))(right[j])
                if missing_now is None:
                    continue
                if t[right[j]] - t[left[j]] - (right[j] - left[j]) / fs > .75:
                    # A paused stream restarts its counter at 0: the counter
                    # says nothing, the clock-pause check below measures it.
                    extra[j] = 0
                    continue
                forward = delta[j] / counter_step - (right[j] - left[j])
                target = extra[j] + missing_now
                readings = [extra[j]] + [forward + k * cycle for k in range(4) if forward + k * cycle > 0]
                extra[j] = min(readings, key=lambda count: abs(count - target))
            ahead = np.concatenate([[0], np.cumsum(extra)])
            # A few garbled packets bend the count either way and then recover.
            ahead = median_filter(ahead, size=15, mode="nearest")
            settled = np.minimum.accumulate(ahead[::-1])[::-1]
            settled = np.floor(settled - settled[0])
            missing[right] = np.diff(settled).astype(np.int64)
            # Counters are evidence only while they agree with the wall clock.
            clock = np.flatnonzero(np.isfinite(t))
            if clock.size > 1:
                elapsed = t[clock] - t[clock[0]]
                total = np.cumsum(missing)[clock]
                padded = (clock - clock[0] + total - total[0]) / fs
                if np.any(padded - elapsed > 5 + .005 * elapsed):
                    missing[:] = 0

    valid = np.flatnonzero(np.isfinite(t))
    if valid.size > 1:
        left, right = valid[:-1], valid[1:]
        elapsed = t[right] - t[left]
        if np.any(elapsed < 0):
            raise ValueError("Recording clock moves backwards; split the recording at the clock reset")
        span = right - left
        uniform = np.abs(elapsed * fs - span) < .1
        if uniform.size >= 3 and uniform.mean() > .9:
            # Uniform device clocks also resolve short losses without counters.
            # Each step is rounded on its own, so a device a little off its
            # nominal rate never accumulates into invented samples. A jump
            # only counts once later samples stay behind it: a block delivered
            # late and then squeezed together catches up, and the suffix
            # minimum ignores it (the median, a stray early stamp).
            offset = np.concatenate([[0], np.cumsum(np.rint(elapsed * fs) - span)])
            offset = median_filter(offset, size=15, mode="nearest")
            settled = np.minimum.accumulate(offset[::-1])[::-1]
            settled -= settled[0]
            if settled[-1] > max(n, int(fs * 60)):
                raise ValueError("Recording has too much missing time to pad safely; split it into separate sessions")
            lost = np.diff(settled).astype(np.int64)
            missing[right] = np.maximum(missing[right], lost)
        residual = t - (np.arange(n) + np.cumsum(missing)) / fs
        # Typical Cyton USB bursts in our recordings are up to about 0.5 s.
        # Confirm a persistent shift on both sides before filling a long pause.
        candidates = right[elapsed - (right - left) / fs > .75]
        width = max(4, int(round(fs)))
        for slot, i in enumerate(candidates):
            previous = candidates[slot - 1] if slot else 0
            following = candidates[slot + 1] if slot + 1 < len(candidates) else n
            before = residual[max(previous, i - width):i]
            after = residual[i:min(following, i + width)]
            before, after = before[np.isfinite(before)], after[np.isfinite(after)]
            if not before.size or not after.size:
                continue
            shift = float(np.median(after) - np.median(before))
            if shift > .5:
                missing[i] += int(round(shift * fs))
    if int(missing.sum()) > max(n, int(fs * 60)):
        raise ValueError("Recording has too much missing time to pad safely; split it into separate sessions")
    return np.arange(n, dtype=np.int64) + np.cumsum(missing)


def event_seconds(timestamps, fs, seconds, start_time):
    """Map a Unix marker onto the actual recorded sample clock, including gaps.

    Host timestamps are arrival times: a USB/RF packet of ~0.5 s gets one
    stamp, so interpolating them directly moves markers by up to that much.
    Within each unbroken run the device clock is linear; its rate is fitted
    over the whole file and its offset follows the earliest arrivals (the
    least delayed samples). Missing stretches are bridged linearly.
    """
    t = np.asarray(timestamps, dtype=float)
    valid = np.flatnonzero(np.isfinite(t))
    if valid.size < 2 or t[valid[-1]] <= t[valid[0]]:
        return float(seconds - start_time)
    # Relative to the first stamp: Unix seconds leave little double precision.
    seconds = seconds - t[valid[0]]
    t = t - t[valid[0]]
    runs = np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1)
    centred = [(r - r.mean(), t[r] - t[r].mean()) for r in runs if r.size > 1]
    spread = sum(float(np.dot(i, i)) for i, _ in centred)
    step = sum(float(np.dot(i, s)) for i, s in centred) / spread if spread else 1 / fs
    if not abs(step * fs - 1) < .05:
        step = 1 / fs            # not a usable clock fit; trust the device rate
    knots_t, knots_i = [], []
    for r in runs:
        offset = float(np.percentile(t[r] - step * r, 1))
        knots_t += [offset + step * r[0], offset + step * r[-1]]
        knots_i += [r[0], r[-1]]
    knots_t = np.maximum.accumulate(knots_t)
    knots_i = np.asarray(knots_i, dtype=float)
    # Preserve out-of-range markers as out-of-range, instead of clamping them.
    if seconds <= knots_t[0]:
        return float((knots_i[0] + (seconds - knots_t[0]) / step) / fs)
    if seconds >= knots_t[-1]:
        return float((knots_i[-1] + (seconds - knots_t[-1]) / step) / fs)
    return float(np.interp(seconds, knots_t, knots_i) / fs)
