import math
import random
from collections.abc import Callable
from itertools import accumulate
from typing import Any, TypedDict

GAMMAS: list[float] = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]


class Result(TypedDict):
    alphas: list[float]
    err: list[int]
    weights: list[list[float]] | None
    gamma_bar: list[float] | None
    etas: list[float] | None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def pinball_loss(beta: float, theta: float, alpha: float) -> float:
    return alpha * (beta - theta) - min(0, beta - theta)


def _result(
        alphas: list[float],
        err: list[int],
        weights: list[list[float]] | None = None,
        gammas: list[float] | None = None,
        etas: list[float] | None = None,
        ) -> Result:
    gamma_bar = None
    if weights is not None and gammas is not None:
        gamma_bar = [sum(p[i] * gammas[i] for i in range(len(gammas))) for p in weights]
    return {
            "alphas": alphas,
            "err": err,
            "weights": weights,
            "gamma_bar": gamma_bar,
            "etas": etas,
            }

def _starting_points(alpha_starts: list[float] | None, alpha: float, k: int,
                     ) -> list[float]:
    if alpha_starts is None:
        return [alpha] * k;
    if len(alpha_starts) != k:
        raise ValueError(f"Must be 1 starting point for each expert")
    if not all(0 <= x <= 1 for x in alpha_starts):
        raise ValueError("Starting points must be in [0, 1]")
    return list(alpha_starts)

# -----------------------------------------------------------------------------
# Constant baseline
# -----------------------------------------------------------------------------


def constant(betas: list[float], alpha: float) -> Result:
    alphas = [alpha] * len(betas)
    err = [int(b < alpha) for b in betas]
    return _result(alphas, err)

# -----------------------------------------------------------------------------
# ACI
# -----------------------------------------------------------------------------


def aci(
        betas: list[float],
        alpha: float,
        gamma: float,
        alpha_start: float | None = None
        ) -> Result:
    a = alpha_start if alpha_start is not None else alpha
    alphas = []
    err = []
    for b in betas:
        alphas.append(a)
        e = int(b < a)
        err.append(e)
        a += gamma * (alpha - e)
    return _result(alphas, err)

# -----------------------------------------------------------------------------
# DtACI
# -----------------------------------------------------------------------------


def dtaci_default_eta(alpha: float, k: int, I: int = 500) -> float:
    return math.sqrt(3 / I) * math.sqrt((math.log(2 * k * I) + 1) / ((1 - alpha) ** 2 * alpha ** 2))


def dtaci(
        betas: list[float],
        alpha: float,
        gammas: list[float] = GAMMAS,
        alpha_start: list[float] | None = None,
        sigma: float | None = None,
        eta: float | None = None,
        I:int = 500,
        variable_eta: bool = False,
        ) -> Result:
    T = len(betas)
    k = len(gammas)
    if sigma is None:
        sigma = 1 / (2 * I)
    if eta is None:
        eta = dtaci_default_eta(alpha, k, I)

    w = [1.0] * k
    a = _starting_points(alpha_start, alpha, k)
    alphas: list[float] = []
    err: list[int] = []
    weights: list[list[float]] = []
    etas: list[float] = []
    sq_losses: list[float] = []
    window = I + 2
    window_sum = 1.0

    for t in range(T):
        b = betas[t]

        W = sum(w)
        p = [w[i] / W for i in range(k)]

        alpha_bar = sum(p[i] * a[i] for i in range(k))
        alphas.append(alpha_bar)
        weights.append(p)

        losses = [pinball_loss(b, a[i], alpha) for i in range(k)]

        sq_losses.append(sum(p[i] * losses[i] ** 2 for i in range(k)))
        window_sum += sq_losses[-1]
        if len(sq_losses) > window:
            window_sum -= sq_losses[-(window + 1)]
        eta_t = eta
        if variable_eta and len(sq_losses) >= window:
            eta_t = math.sqrt((math.log(2 * k * I) + 1) / window_sum)
        etas.append(eta_t)

        w_bar = [w[i] * math.exp(-eta_t * losses[i]) for i in range(k)]
        W_bar = sum(w_bar)
        w = [(1 - sigma) * w_bar[i] + W_bar * sigma / k for i in range(k)]

        W = sum(w)
        w = [w[i] / W for i in range(k)]

        err_i = [int(b < a[i]) for i in range(k)]
        err.append(int(b < alpha_bar))
        a = [a[i] + gammas[i] * (alpha - err_i[i]) for i in range(k)]

    return _result(alphas, err, weights, gammas, etas)

# -----------------------------------------------------------------------------
# AgACI
# -----------------------------------------------------------------------------


def agaci(
        betas: list[float],
        alpha: float,
        gammas: list[float] = GAMMAS,
        alpha_start: list[float] | None = None,
        ) -> Result:
    T = len(betas)
    k = len(gammas)

    L = [0.0] * k
    eta = [0.0] * k
    sum_sq = [0.0] * k
    max_abs = [0.0] * k
    a = _starting_points(alpha_start, alpha, k)
    alphas: list[float] = []
    err: list[int] = []
    weights: list[list[float]] = []
    a_prev: list[float] = a[:]
    alpha_prev = 0.0
    err_prev = 0

    for t in range(T):
        b = betas[t]

        if t > 0:
            eta_prev = eta[:]
            for i in range(k):
                # using a_t-1, typo
                l = (err_prev - alpha) * (a_prev[i] - alpha_prev)
                sum_sq[i] += l ** 2
                max_abs[i] = max(max_abs[i], abs(l))
                if max_abs[i] == 0:
                    continue
                c = math.ceil(math.log2(max_abs[i]))
                L[i] += 0.5 * (l * (1 + eta_prev[i] * l) + 2 ** (c + 1) * (eta_prev[i] * l > 0.5))
                eta[i] = min(2 ** (-c - 1), math.sqrt(math.log(k) / sum_sq[i]))

        logs = [math.log(eta[i]) - eta[i] * L[i] if eta[i] > 0 else -math.inf for i in range(k)]
        top = max(logs)
        if top == -math.inf:
            p = [1 / k] * k
        else:
            raw = [math.exp(x - top) for x in logs]
            p = [x / sum(raw) for x in raw]

        alpha_t = sum(p[i] * a[i] for i in range(k))
        alphas.append(alpha_t)
        weights.append(p)
        e = int(b < alpha_t)
        err.append(e)

        a_prev, alpha_prev, err_prev = a[:], alpha_t, e
        a = [a[i] + gammas[i] * (alpha - int(b < a[i])) for i in range(k)]

    return _result(alphas, err, weights, gammas)

# -----------------------------------------------------------------------------
# MVP
# -----------------------------------------------------------------------------


def mvp(
        betas: list[float],
        alpha: float,
        m: int = 40,
        eta: float | None = None,
        r: int = 800000,
        seed: int = 0,
        ) -> Result:
    if eta is None:
        eta = math.sqrt(math.log(m) / (4.2 * m))
    rng = random.Random(f"mvp-{seed}")

    def f(n: int) -> float:
        return math.sqrt((n + 1) * math.log(n + 2) ** 2)

    def c_val(v: float, cnt: int) -> float:
        fn = f(cnt)
        return (math.exp(eta * v / fn) - math.exp(-eta * v / fn)) / fn

    n = [0] * m
    V = [0.0] * m
    C = [c_val(V[i], n[i]) for i in range(m)]
    alphas: list[float] = []
    err: list[int] = []

    for b in betas:
        if all(c > 0 for c in C):
            q = 0.0
            bin_idx = 0
        elif all(c < 0 for c in C):
            q = 1.0
            bin_idx = m - 1
        else:
            i_star = min(i for i in range(1, m) if C[i - 1] * C[i] <= 0)
            num = abs(C[i_star])
            den = abs(C[i_star]) + abs(C[i_star - 1])
            p = 1.0 if den == 0 else num / den

            if rng.random() < p:
                q = i_star / m - 1 / (r * m)
                bin_idx = i_star - 1
            else:
                q = i_star / m
                bin_idx = i_star

        alpha_t = 1 - q
        e = int(b < alpha_t)
        alphas.append(alpha_t)
        err.append(e)

        n[bin_idx] += 1
        V[bin_idx] += alpha - e
        C[bin_idx] = c_val(V[bin_idx], n[bin_idx])

    return _result(alphas, err)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


METHODS: dict[str, Callable[..., Result]] = {
        "constant": constant,
        "aci": aci,
        "dtaci": dtaci,
        "agaci": agaci,
        "mvp": mvp,
        }


def run_method(name: str, betas: list[float], alpha: float = 0.1, **kwargs: Any) -> Result:
    return METHODS[name](betas, alpha, **kwargs)

# -----------------------------------------------------------------------------
# Coverage metrics
# -----------------------------------------------------------------------------


def global_coverage(err: list[int]) -> float:
    return 1 - sum(err) / len(err)


def local_coverage(err: list[int], w: int) -> list[float]:
    T = len(err)
    half = w // 2
    out = [math.nan] * T
    c = [0, *accumulate(err)]
    for t in range(half - 1, T - half):
        out[t] = 1 - (c[t + half + 1] - c[t - half + 1]) / w
    return out
