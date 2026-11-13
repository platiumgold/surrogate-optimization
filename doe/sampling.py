import numpy as np
def random_sampling(bounds, n_samples, seed=None):
    """план выбора n_samples точек на области bounds=[(l1, h1), (l2, h2) ...] случайным образом"""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, len(bounds)))


def latin_hypercube_sampling(bounds, n_samples, seed=None):
    """план выбора n_samples точек на области bounds=[(l1, h1), (l2, h2) ...] методом латинского гиперкуба"""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds)
    dims = len(bounds)
    low = bounds[:, 0]
    high = bounds[:, 1]
    perms = np.asarray([rng.permutation(n_samples) for _ in range(dims)]).T
    #случайный сдвиг внутри ячейки
    strat = rng.random((n_samples, dims))
    x_normed = (perms + strat) / n_samples
    x = low + x_normed * (high - low)
    return x

if __name__ == "__main__":
    bounds = [(10,20), (30,40)]
    n_samples = 4
    x = latin_hypercube_sampling(bounds, n_samples, seed=42)
    print(x)