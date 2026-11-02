"""One rule for every live y axis: grow at once, shrink only when it lasts.

Why it is shared: an axis that follows each frame makes the curve jump, so the
signal cannot be read; an axis that only grows leaves one artefact (a headset
being adjusted) squashing a few-microvolt signal onto the baseline until
something is pressed. The spectra, the band powers and the live features all
need the same compromise, and having it in one place keeps them consistent.
"""
import time

#: how long the values must stay small before the axis follows them down
SETTLE_SECONDS = 3.0
#: room left above a new peak, so small changes do not resize the axis again
HEADROOM = 1.2
#: linear axes shrink when the values fit in this fraction of the axis...
SHRINK_RATIO = 4.0
#: ...and log axes when they fit this many decades below it
SHRINK_DECADES = 1.0


class HeldScale:
    """The top of an axis, held steady.

    `log` axes work in decades (the value passed in is already log10), linear
    ones in the unit itself.
    """

    def __init__(self, log=False, settle_seconds=SETTLE_SECONDS, headroom=HEADROOM):
        self.log = bool(log)
        self.settle_seconds = float(settle_seconds)
        self.headroom = float(headroom)
        self.top = None
        self._small_since = None

    def reset(self):
        """Forget the held value: a new source, channel or unit starts afresh."""
        self.top = None
        self._small_since = None

    def update(self, fit, now=None):
        """Give the axis top to draw with, for values that exactly fit `fit`."""
        now = time.perf_counter() if now is None else float(now)
        fit = float(fit)
        if self.top is None:
            self.top = fit
            self._small_since = None
        elif fit > self.top:
            self.top = fit if self.log else fit * self.headroom
            self._small_since = None
        elif self._much_smaller(fit):
            # A clock that jumped backwards (a seek) starts the wait again
            # rather than shrinking at once or never.
            if self._small_since is None or now < self._small_since:
                self._small_since = now
            elif now - self._small_since >= self.settle_seconds:
                self.top, self._small_since = fit, None
        else:
            self._small_since = None
        return self.top

    def _much_smaller(self, fit):
        return fit <= self.top - SHRINK_DECADES if self.log else fit <= self.top / SHRINK_RATIO
