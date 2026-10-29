from __future__ import annotations

import numpy as np
import pytest

from framelab.physics.models import polyfit_params, polyval
from framelab.physics.uncertainty import (
    block_bootstrap,
    combine_sigma,
    fmt,
    fmt_interval,
    percentile_ci,
    sigma_from_ci,
)


@pytest.mark.parametrize(
    ("value", "sigma", "unit", "lang", "expected"),
    [
        (9.7432, 0.1234, "m/s^2", "en", "9.74 ± 0.12 m/s²"),
        (9.7432, 0.1234, "m/s^2", "id", "9,74 ± 0,12 m/s²"),
        (9.8236, 0.0318, "m/s^2", "en", "9.82 ± 0.03 m/s²"),
        (1.4271, 1.96, "s", "en", "1.4 ± 2.0 s"),
        (0.56789, 0.0123, "m", "en", "0.568 ± 0.012 m"),
        (9.81, 0.0996, "m/s^2", "en", "9.81 ± 0.10 m/s²"),
        (123.4, 25.0, "s", "en", "123 ± 25 s"),
        (9.81, None, "m/s^2", "en", "9.81 m/s²"),
        (12.0, 0.5, "deg", "en", "12.0 ± 0.5°"),
    ],
)
def test_fmt(value, sigma, unit, lang, expected) -> None:
    assert fmt(value, sigma, unit, lang) == expected


def test_fmt_interval() -> None:
    assert fmt_interval(9.6234, 9.8611, "m/s^2", 0.12, "en") == "[9.62, 9.86] m/s²"
    assert fmt_interval(9.6234, 9.8611, "m/s^2", 0.12, "id") == "[9,62; 9,86] m/s²"


def test_sigma_helpers() -> None:
    assert sigma_from_ci(-1.96, 1.96) == pytest.approx(1.0)
    assert combine_sigma(3.0, 4.0) == pytest.approx(5.0)
    assert percentile_ci(np.arange(1001)) == pytest.approx((25.0, 975.0))


def test_block_bootstrap_reproducible_and_shaped() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 50)
    y = 3 + 2 * t + 5 * t**2 + rng.normal(0, 0.1, t.size)

    def fit(tt, yy):
        return polyfit_params(tt, yy, 2)

    def pred(tt, p):
        return polyval(tt, p)

    a = block_bootstrap(t, y, fit, pred, n=200, seed=7)
    b = block_bootstrap(t, y, fit, pred, n=200, seed=7)
    assert a.shape == (200, 3) and np.array_equal(a, b)
    c2 = polyfit_params(t, y, 2)[2]
    lo, hi = percentile_ci(a[:, 2])
    assert lo < c2 < hi and 0.1 < hi - lo < 2.0
