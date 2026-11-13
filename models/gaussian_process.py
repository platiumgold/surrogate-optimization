import math as m
import numpy as np
def expected_value(X, m): return [m(x) for x in X]
def cov_matrix(X, X1, k): return np.array([[k(x, x1) for x1 in X1] for x in X])

class GaussianProcess:
    def __init__(self, k, m=None, noise_var=1e-6):
        self.m = m if m is not None else (lambda x: 0.0)
        self.k = k
        self.X = None
        self.y = None
        self.noise_var = noise_var

    def fit(self, X, y):
        """Обучает ГП на точках X формы (N, d) и значениях y формы (N,)."""
        self.X = X
        self.y = y
        self.K_xx = cov_matrix(self.X, self.X, self.k) + (self.noise_var) * np.eye(len(X))
        self.K_xx_inv = np.linalg.inv(self.K_xx)
        m_X = np.array(expected_value(X, self.m))
        #коэфт для вычисления прогноза мат ожидания зависит только от заданного X
        self.theta = self.K_xx_inv @ (self.y - m_X)

    def predict(self, X_star):
        """Возвращает прогноз mu и std для тестовых точек X_star формы (M, d)."""
        predictions = []
        stds = []
        for xs in X_star:
            cov_X_xs = cov_matrix(self.X, [xs], self.k)
            pred = self.m(xs) + self.theta.T @ cov_X_xs
            var = cov_matrix([xs], [xs], self.k) - cov_X_xs.T @ self.K_xx_inv @ cov_X_xs

            predictions.append(pred)
            #защита от nan
            var = np.maximum(var, 0.0)
            stds.append(np.sqrt(var))

        return np.array(predictions).ravel(), np.array(stds).ravel()


class SqExpKernel:
    def __init__(self, l=1.0):
        self.l = l
    def __call__(self, x, x1):
        return np.exp(-np.sum((x-x1)**2) / (2*self.l**2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gp = GaussianProcess(SqExpKernel(l=1.0), noise_var=1e-9)

    X_train = np.array([[0.0], [1.5], [3.0], [4.5]])
    y_train = np.sin(X_train[:, 0])
    gp.fit(X_train, y_train)

    X_plot = np.linspace(-0.5, 5.0, 200)[:, None]
    mu, std = gp.predict(X_plot)

    plt.figure(figsize=(8, 4))
    plt.plot(X_plot, np.sin(X_plot), "k--", label="Истинная f(x)")
    plt.plot(X_plot, mu, "b-", label="Предсказание μ(x)")
    plt.fill_between(X_plot.ravel(), mu - 1.96 * std, mu + 1.96 * std, color="b", alpha=0.15, label="Доверительный интервал 95%")
    plt.scatter(X_train, y_train, color="k", zorder=3, label="Обучающие точки")
    plt.legend()
    plt.tight_layout()
    plt.show()
