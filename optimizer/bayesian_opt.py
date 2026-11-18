from acquisition.criteria import propose_candidate
from doe.sampling import latin_hypercube_sampling
import numpy as np

class BayesianOptimizer:
    """Контур байесовской оптимизации (DoE -> surrogate fit -> acquisition -> step)."""

    def __init__(self, objective, bounds, surrogate, acquisition_func, n_candidates=1000):
        self.objective = objective
        self.bounds = bounds
        self.surrogate = surrogate
        self.acquisition_func = acquisition_func
        self.n_candidates = n_candidates

        self.X_history = None
        self.y_history = None

    def initialize(self, n_init=4, seed=None):
        """Холодный старт: генерация n_init точек через LHS и первичное обучение ГП."""
        start_sample = latin_hypercube_sampling(self.bounds, n_init, seed)
        y_start = np.array([self.objective(x) for x in start_sample]).ravel()
        self.X_history = start_sample
        self.y_history = y_start
        self.surrogate.fit(self.X_history, self.y_history)

    def step(self):
        """Один шаг: выбор x_next по критерию, замер функции и дообучение ГП."""
        x_next = propose_candidate(self.acquisition_func, self.surrogate, self.bounds, self.n_candidates)
        y_next = self.objective(x_next)

        self.X_history = np.vstack((self.X_history, x_next))
        self.y_history = np.append(self.y_history, y_next)

        self.surrogate.fit(self.X_history, self.y_history)

        return x_next, y_next

    def optimize(self, n_steps=15):
        """Запуск цикла на n_steps шагов. Возвращает (best_x, best_y)."""
        if self.X_history is None:
            self.initialize()
            
        for _ in range(n_steps):
            self.step()
        best_idx = np.argmin(self.y_history)
        return self.X_history[best_idx], self.y_history[best_idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from benchmarks.functions import forrester, FORRESTER_BOUNDS
    from models.gaussian_process import GaussianProcess, SqExpKernel
    from acquisition.criteria import expected_improvement

    gp = GaussianProcess(SqExpKernel(l=0.15), noise_var=1e-6)
    opt = BayesianOptimizer(forrester, FORRESTER_BOUNDS, gp, expected_improvement)
    opt.initialize(n_init=4)
    best_x, best_y = opt.optimize(n_steps=12)

    X_plot = np.linspace(0.0, 1.0, 200)[:, None]
    mu, std = opt.surrogate.predict(X_plot)
    y_true = forrester(X_plot)
    running_best = np.minimum.accumulate(opt.y_history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(X_plot, y_true, "k--", label="f(x) Форрестера")
    ax1.plot(X_plot, mu, "b-", label="Суррогат μ(x)")
    ax1.fill_between(X_plot.ravel(), mu - 1.96 * std, mu + 1.96 * std, color="b", alpha=0.15)
    ax1.scatter(opt.X_history[:4], opt.y_history[:4], color="gray", label="Старт (LHS)", zorder=3)
    ax1.scatter(opt.X_history[4:], opt.y_history[4:], color="blue", label="Шаги BO", zorder=4)
    ax1.scatter(best_x, best_y, color="red", marker="*", s=150, label=f"Минимум: {best_y:.2f}", zorder=5)
    ax1.legend()
    ax1.set_title("Финальная аппроксимация функции")

    ax2.plot(range(1, len(running_best) + 1), running_best, "r-o")
    ax2.axhline(-6.0207, color="k", linestyle="--", label="Истинный оптимум (-6.02)")
    ax2.set_xlabel("Номер замера")
    ax2.set_ylabel("Лучшее найденное y")
    ax2.set_title("Кривая сходимости")
    ax2.legend()

    plt.tight_layout()
    plt.show()