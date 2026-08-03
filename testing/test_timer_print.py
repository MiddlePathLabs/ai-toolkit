"""Regression tests for toolkit.timer.Timer.print().

The OOM-skip path in BaseSDTrainProcess cancels a running timer bucket (via the
context manager's __exit__) but leaves the empty deque that start() registered
in self.timers. Timer.print() then divided by len(timings) == 0, turning a
recovered OOM into a ZeroDivisionError. These tests pin the fix: empty buckets
are skipped while populated ones are still reported and hooks still fire.
"""
import pytest

from toolkit.timer import Timer


def _seed(t: Timer, name: str, count: int) -> None:
    for _ in range(count):
        t.start(name)
        t.stop(name)


def test_print_skips_empty_bucket_after_cancel():
    t = Timer('oom')
    t.reset()
    # simulate: bucket started then cancelled (as __exit__ does on exception)
    t.start('calculate_loss')
    t.cancel('calculate_loss')
    # a populated bucket coexists
    _seed(t, 'train_loop', 2)
    assert len(t.timers['calculate_loss']) == 0

    captured = {}
    t.add_after_print_hook(lambda d: captured.update(d))

    # must not raise ZeroDivisionError
    t.print()

    # empty bucket skipped, populated bucket reported
    assert 'calculate_loss' not in captured
    assert 'train_loop' in captured
    assert captured['train_loop'] >= 0.0


def test_print_after_context_manager_exception_does_not_raise():
    """Mirrors the real OOM path: an exception inside `with timer(name):`
    cancels the bucket, leaving it registered-but-empty."""
    t = Timer('oom')
    t.reset()
    _seed(t, 'train_loop', 1)

    with pytest.raises(RuntimeError):
        with t('calculate_loss'):
            raise RuntimeError('simulated OOM')

    # bucket is registered but empty
    assert 'calculate_loss' in t.timers
    assert len(t.timers['calculate_loss']) == 0

    captured = {}
    t.add_after_print_hook(lambda d: captured.update(d))

    # print must not turn the recovered OOM into a ZeroDivisionError
    t.print()

    assert 'calculate_loss' not in captured
    assert 'train_loop' in captured


def test_print_all_buckets_empty_after_reset_does_not_raise():
    t = Timer('empty')
    t.reset()
    t.start('a')
    t.cancel('a')
    t.start('b')
    t.cancel('b')

    fired = {}
    t.add_after_print_hook(lambda d: fired.update(d))

    # every bucket empty -> no crash, hook still fires with empty dict
    t.print()
    assert fired == {}
