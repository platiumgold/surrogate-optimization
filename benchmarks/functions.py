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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x_1d = np.linspace(FORRESTER_BOUNDS[0][0], FORRESTER_BOUNDS[0][1], 500)
    y_1d = forrester(x_1d)

    ax1.plot(x_1d, y_1d, "b-", lw=2, label="f(x) Форрестера")
    ax1.scatter([0.7572], [-6.0207], color="red", marker="*", s=180, zorder=5, label="Глобальный мин. (0.76, -6.02)")
    ax1.scatter([0.1426], [-0.9863], color="orange", marker="o", s=100, zorder=5, label="Локальный мин. (0.14, -0.99)")

    ax1.set_title("1D тестовая функция Форрестера", fontsize=12)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("f(x)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    x1_grid = np.linspace(BRANIN_BOUNDS[0][0], BRANIN_BOUNDS[0][1], 300)
    x2_grid = np.linspace(BRANIN_BOUNDS[1][0], BRANIN_BOUNDS[1][1], 300)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    pts_2d = np.column_stack([X1.ravel(), X2.ravel()])
    Z = branin(pts_2d).reshape(X1.shape)

    levels = np.logspace(np.log10(0.39), np.log10(315), 35)
    cs = ax2.contourf(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.9, extend="both")
    ax2.contour(X1, X2, Z, levels=levels[::3], colors="k", linewidths=0.5, alpha=0.4)

    branin_minima = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
    ax2.scatter(
        branin_minima[:, 0],
        branin_minima[:, 1],
        color="red",
        marker="*",
        s=180,
        zorder=5,
        label="Глобальные минимумы (f ≈ 0.398)",
        edgecolors="white",
        linewidths=0.8,
    )

    cbar = fig.colorbar(cs, ax=ax2)
    cbar.set_label("f(x₁, x₂)", fontsize=11)

    ax2.set_title("2D тестовая функция Бранина", fontsize=12)
    ax2.set_xlabel("x₁", fontsize=11)
    ax2.set_ylabel("x₂", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("report/images/fig4_benchmark_functions.png", dpi=300)
    plt.show()

