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
    import matplotlib.pyplot as plt

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    n_samples = 10
    seed = 42

    x_rnd = random_sampling(bounds, n_samples, seed=seed)
    x_lhs = latin_hypercube_sampling(bounds, n_samples, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    ticks = np.linspace(0.0, 1.0, n_samples + 1)

    ax1.scatter(x_rnd[:, 0], x_rnd[:, 1], color="k", zorder=3)
    ax1.set_title("Случайный выбор")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.scatter(x_lhs[:, 0], x_lhs[:, 1], color="b", zorder=3)
    ax2.set_title("Выборка по латинскому гиперкубу")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    import os
    os.makedirs("report/images", exist_ok=True)
    plt.savefig("report/images/fig1_doe_comparison.png", dpi=300)
    plt.show()