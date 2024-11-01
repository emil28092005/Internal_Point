import numpy as np
from typing import Optional
from enum import Enum


class State(Enum):
    SOLVED = 0
    UNSOLVED = 1
    UNAPPLICABLE = 2


class Result:
    solved: State
    objective_function_value: Optional[np.float64]
    solution: Optional[np.array]

    def __init__(self,
                 solved: State,
                 objective_function_value: Optional[np.array] = None,
                 solution: np.float64 = None):
        self.solved = solved
        self.objective_function_value = objective_function_value
        self.solution = solution


def interior_point(
        C: np.array,
        A: np.array,
        x_0: np.array,
        b: np.array,
        eps: np.float64 = 0.01,
        alpha: np.float64 = 0.5,
        maximizing: bool = True) -> Result:
    if (not np.all(np.dot(A, x_0) >= b) or np.any(x_0 == 0)):
        return Result(State.UNAPPLICABLE)
    if (not maximizing):
        C = -C
        
    m = len(A)
    n = len(A[0])

    x = np.ones(n)
    s = np.ones(m)

    iteration = 0

    while(True):

        for i in range(m):
            slack = b[i]
            for j in range(n):
                slack -= A[i][j] * x[j]
            s[i] = slack
        
        for i in range(min(m, n)):
            x[i] = s[i]

        D = np.diag(s)

        x_star = np.dot(np.linalg.inv(D), x)
        A_star = np.dot(A, D)
        C_star = np.dot(D, C)

        I = np.eye(n)
        A_star_transpose = np.transpose(A_star)
        P = I - np.dot(A_star_transpose, np.linalg.inv(np.dot(A_star, A_star_transpose)))
        P = np.dot(P, A_star)

        C_p = np.dot(P, C_star)
        Mu = np.max(np.absolute(C_p))

        if Mu < eps:
            result = np.dot(C, x)
            return Result(State.SOLVED, objective_function_value=result, solution=x)

        iteration += 1

        if iteration >= 1000:
            return Result(State.UNSOLVED)

        x_star += (alpha / Mu) * C_p
        x = np.dot(D, x_star)

# TODO 5 tests (from assignment 1) and comparison with simplex and alpha = 0.9
def TEST_CASE_GENERAL():
    pass
