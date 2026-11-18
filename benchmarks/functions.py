import numpy as np

FORRESTER_BOUNDS = [(0.0, 1.0)]
BRANIN_BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]


def forrester(x):
    """1D тестовая функция Форрестера: f(x) = (6x - 2)^2 * sin(12x - 4), x in [0, 1]."""
    x = np.asarray(x).ravel()
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def branin(x):
    """2D тестовая функция Бранина: x1 in [-5, 10], x2 in [0, 15]."""
    x = np.asarray(x)
    x1, x2 = x[..., 0], x[..., 1]
    a, b = 1.0, 5.1 / (4.0 * np.pi**2)
    c, r = 5.0 / np.pi, 6.0
    s, t = 10.0, 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s
