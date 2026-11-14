import doe.sampling as sampling
import numpy as np
from scipy.stats import norm
def pure_exploitation(X, gp):
    """исследование, основанное на предсказании
    возвращает полезность точек (m,) для X (m, n)"""
    
    exp_vals, _ = gp.predict(X)
    return -exp_vals

def pure_exploration(X, gp):
    """исследование на основе ошибок
    возвращает полезность точек (m,) для X (m, n)"""

    _ , stds = gp.predict(X)
    return stds

def lower_confidence_bound(X, gp, alpha=2.0):
    """исследование на основе нижней доверительной границы
    возвращает полезность точек (m,) для X (m, n)"""

    exp_vals, stds = gp.predict(X)
    return - (exp_vals - alpha * stds)

def probability_of_improvement(X, gp):
    """исследование на основе вероятности улучшения
    возвращает полезность точек (m,) для X (m, n)"""
    y_min = np.min(gp.y)
    exp_vals, stds = gp.predict(X)

    # Считаем z только там, где есть неопределенность, в остальных точках вероятность 0.0
    mask = stds > 1e-9
    probs = np.zeros_like(stds)
    z = (y_min - exp_vals[mask]) / stds[mask]
    probs[mask] = norm.cdf(z)
    return probs



def propose_candidate(acq_func, gp, bounds, n_candidates=1000, seed=None):
    """Поиск наилучшего кандидата в пространствах малых размерностей 
    из случайно сгенерированных n_candidates точек по acq_func критерию
    при помощи gp процесса в пространстве bounds=[(l1, h1), (l2, h2) ...]"""

    candidates = sampling.latin_hypercube_sampling(bounds, n_candidates, seed)
    utilities = acq_func(candidates, gp)
    best_idx = np.argmax(utilities)
    return candidates[best_idx]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from models.gaussian_process import GaussianProcess, SqExpKernel

    gp = GaussianProcess(SqExpKernel(l=1.0), noise_var=1e-6)
    X_train = np.array([[0.5], [1.5], [4.5]])
    y_train = np.cos(X_train[:, 0])
    gp.fit(X_train, y_train)

    bounds = [(0.0, 5.0)]
    x_exploit = propose_candidate(pure_exploitation, gp, bounds, seed=42)
    x_explore = propose_candidate(pure_exploration, gp, bounds, seed=42)
    x_lcb = propose_candidate(lower_confidence_bound, gp, bounds, seed=42)
    x_pi = propose_candidate(probability_of_improvement, gp, bounds, seed=42)

    X_plot = np.linspace(0.0, 5.0, 200)[:, None]
    mu, std = gp.predict(X_plot)
    u_exploit = pure_exploitation(X_plot, gp)
    u_explore = pure_exploration(X_plot, gp)
    u_lcb = lower_confidence_bound(X_plot, gp)
    u_pi = probability_of_improvement(X_plot, gp)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax1.plot(X_plot, np.cos(X_plot), "k--", label="f(x)")
    ax1.plot(X_plot, mu, "b-", label="μ(x)")
    ax1.fill_between(X_plot.ravel(), mu - 1.96 * std, mu + 1.96 * std, color="b", alpha=0.15)
    ax1.scatter(X_train, y_train, color="k", zorder=3, label="Замеры")
    ax1.legend()

    ax2.plot(X_plot, u_exploit, "g-", label="Exploit (-μ)")
    ax2.plot(X_plot, u_explore, "m-", label="Explore (σ)")
    ax2.plot(X_plot, u_lcb, "c-", label="LCB (-μ + 2σ)")
    ax2.axvline(x_exploit[0], color="g", linestyle="--")
    ax2.axvline(x_explore[0], color="m", linestyle="--")
    ax2.axvline(x_lcb[0], color="c", linestyle="--")
    ax2.legend()

    ax3.plot(X_plot, u_pi, "r-", label="Prob of Improvement (PI)")
    ax3.axvline(x_pi[0], color="r", linestyle="--", label=f"Next PI: {x_pi[0]:.2f}")
    ax3.legend()

    plt.tight_layout()
    plt.show()
    