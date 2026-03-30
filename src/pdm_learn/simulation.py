from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_int
from sympy import diff, sqrt, utilities


def eps(size: int, std: float = 1) -> np.ndarray:
    return np.array([np.random.normal(scale=std) for _ in range(size)])


def partition(
    u,
    eq,
    bounds: tuple[float, float] = (-1, 1),
    num: int = -1,
    spacing: float = 0.1,
    iterations: int = 10,
    max_err: float = 0,
    endpoint: bool = True,
) -> np.ndarray:
    curr_x = bounds[0]
    out = np.array([curr_x])
    breakout = False

    if num != -1:
        t_func = utilities.lambdify(u, sqrt(1 + (diff(eq, u)) ** 2))
        spacing = scipy_int.quad(t_func, bounds[0], bounds[1])[0] / (num - 1)
        delta = spacing
        for _ in range(num - 1):
            curr_y = eq.subs(u, curr_x).evalf()
            for _ in range(iterations):
                next_y = eq.subs(u, curr_x + delta).evalf()
                if not next_y.is_real:
                    breakout = True
                    break
                dist = sqrt(delta**2 + (curr_y - next_y) ** 2)
                if abs(dist - spacing) < max_err:
                    break
                delta *= spacing / dist
                if curr_x + delta > bounds[1]:
                    breakout = True
                    break

            if breakout:
                break
            curr_x += delta
            out = np.append(out, curr_x)

        if len(out) < num:
            curr_x = bounds[1]
            temp = np.array([curr_x])
            delta *= -1
            while len(out) + len(temp) < num:
                curr_y = eq.subs(u, curr_x).evalf()
                for _ in range(iterations):
                    next_y = eq.subs(u, curr_x + delta).evalf()
                    dist = sqrt(delta**2 + (curr_y - next_y) ** 2)
                    if abs(dist - spacing) < max_err:
                        break
                    delta *= spacing / dist
                curr_x += delta
                temp = np.append(curr_x, temp)
            out = np.append(out, temp)

    else:
        delta = spacing
        while curr_x + delta <= bounds[1]:
            curr_y = eq.subs(u, curr_x).evalf()
            for _ in range(iterations):
                next_y = eq.subs(u, curr_x + delta).evalf()
                dist = sqrt(delta**2 + (curr_y - next_y) ** 2)
                if abs(dist - spacing) < max_err:
                    break
                delta *= spacing / dist
                if curr_x + delta > bounds[1]:
                    breakout = True
                    break
            if breakout:
                if endpoint:
                    out = np.append(out, bounds[1])
                break
            curr_x += delta
            out = np.append(out, curr_x)

    return out
