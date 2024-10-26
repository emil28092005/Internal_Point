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
    solution: Optional[np.array[np.float64]]

    def __init__(self,
                 solved: State,
                 objective_function_value: Optional[np.array[np.float64]] = None,
                 solution: np.float64 = None):
        self.solved = solved
        self.objective_function_value = objective_function_value
        self.solution = solution


def interior_point(
        C: np.array[np.float64],
        A: np.array[np.float64],
        x_0: np.array[np.float64],
        b: np.array[np.float64],
        eps: np.float64 = 0.01,
        alpha: np.float64 = 0.5,
        maximizing: bool = True) -> Result:
    if (not np.all(np.dot(A, x_0) >= b) or np.any(x_0 == 0)):
        return Result(State.UNAPPLICABLE)
    if (not maximizing):
        C = -C
    x = x_0
    solved = False
    while not solved:
        # TODO Algorithm steps (refer to numpy.linalg for matrix stuff)
        pass

    # return value (include check for minimization)
    return Result(State.SOLVED, ...)


# TODO 5 tests (from assignment 1) and comparison with simplex and alpha = 0.9
def TEST_CASE_GENERAL():
    pass
