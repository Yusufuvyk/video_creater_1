"""Background work with revision-based delivery; workers never touch widgets.

`activity()` counts the work the person is waiting for (an analysis, a pipeline run,
a report), so the window can show a moving trace instead of looking frozen.
"""
from contextlib import contextmanager

from pyqtgraph.Qt import QtCore, QtWidgets


class Activity(QtCore.QObject):
    """How much work is going on that someone waits for; `changed(True)` when the first starts,
    `changed(False)` when the last ends. Lives on the window's thread."""

    changed = QtCore.Signal(bool)

    def __init__(self):
        super().__init__()
        self.count = 0

    def begin(self):
        self.count += 1
        if self.count == 1:
            self.changed.emit(True)

    def end(self):
        if self.count == 0:
            return
        self.count -= 1
        if self.count == 0:
            self.changed.emit(False)

    @QtCore.Slot(int, object, object)
    def job_done(self, *_args):
        self.end()

    @property
    def busy(self):
        return self.count > 0


_activity = None


def activity():
    global _activity
    if _activity is None:
        _activity = Activity()
    return _activity


@contextmanager
def busy():
    """Work done on the window's thread (a report): the moving trace and the wait cursor while it lasts."""
    activity().begin()
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    QtWidgets.QApplication.processEvents()
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
        activity().end()


class Signals(QtCore.QObject):
    done = QtCore.Signal(int, object, object)


class Job(QtCore.QRunnable):
    def __init__(self, revision, function):
        super().__init__()
        self.revision, self.function = revision, function
        self.signals = Signals()

    def run(self):
        try:
            result, error = self.function(), None
        except Exception as exc:
            result, error = None, str(exc)
        self.signals.done.emit(self.revision, result, error)


class Jobs(QtCore.QObject):
    finished = QtCore.Signal(object, object)

    def __init__(self, parent=None, busy=False):
        super().__init__(parent)
        self.revision = 0
        self.active = {}
        #: work someone waits for: counted in `activity()` until it ends
        self.busy = busy

    def invalidate(self):
        self.revision += 1

    def start(self, function):
        self.invalidate()
        job = Job(self.revision, function)
        self.active[self.revision] = job
        job.signals.done.connect(self._deliver)
        if self.busy:
            activity().begin()
            job.signals.done.connect(activity().job_done)      # every job ends once, stale or not
        QtCore.QThreadPool.globalInstance().start(job)

    @QtCore.Slot(int, object, object)
    def _deliver(self, revision, result, error):
        self.active.pop(revision, None)
        if revision == self.revision:
            self.finished.emit(result, error)
