import time
from time import time

class TimeAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._cnt = 0
        self._total_time = 0
        self._total_samples = 0

    def record(self, usetime, num_samples=None):
        self._cnt += 1
        self._total_time += usetime
        if num_samples:
            self._total_samples += num_samples

    def get_average(self):
        if self._cnt == 0:
            return 0
        return self._total_time / float(self._cnt)

    def get_ips_average(self):
        if not self._total_samples or self._cnt == 0:
            return 0
        return float(self._total_samples) / self._total_time


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step * speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60 ** i))
        remaining_time %= 60 ** i
    return result.format(*arr)


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur


_g_timers = {}  # global timers


def check_time(timer_id):
    """Add check points in a single line.

    This method is suitable for running a task on a list of items. A timer will
    be registered when the method is called for the first time.

    :Example:

    >>> import time
    >>> import mmcv
    >>> for i in range(1, 6):
    >>>     # simulate a code block
    >>>     time.sleep(i)
    >>>     mmcv.check_time('task1')
    2.000
    3.000
    4.000
    5.000

    Args:
        timer_id (str): Timer identifier.
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()
