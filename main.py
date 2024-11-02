import numpy as np
from typing import Optional
from enum import Enum


class State(Enum):
    SOLVED = 0
    UNSOLVED = 1
    INAPPLICABLE = 2


class Result:
    state: State
    objective_function_value: Optional[np.float64]
    solution: Optional[np.array]
    maximize: bool

    def __init__(self,
                 state: State,
                 objective_function_value: Optional[np.array] = None,
                 solution: np.float64 = None, maximize: bool = True):
        self.state = state
        self.objective_function_value = objective_function_value
        self.solution = solution
        self.maximize = maximize


def print_initial_inputs(
        C: np.array,  # Vector of objective function coefficients
        A: np.array,  # Matrix of constraint coefficients
        x_0: np.array,  # Initial point (vector)
        b: np.array,  # Vector of right-hand side values of constraints
        eps: np.float64 = 0.01,  # Solution accuracy
        alpha: np.float64 = 0.5,  # Step coefficient
        maximize: bool = True):

    print("Running for the following inputs:\n")
    print(f"epsilon: {eps} ")
    print(f"alpha: {alpha}")
    print(f"x_0: {x_0} \n")
    if (maximize):

        print("Maximize")

    else:

        print("Minimize")

    z_str = "z = "
    previousIsZero = True
    lastNonZero = False
    for i in range(len(C)):

        isNegative = False

        for k in range(len(C)):

            if (C[k] == 0):

                lastNonZero = True
            else:
                lastNonZero = False
                break

        if (not previousIsZero and not lastNonZero):
            z_str += " + "

        if (C[i] != 0):
            if (C[i] != 1):
                if (C[i] < 0):
                    isNegative = True
                    z_str += "("

                z_str += str(C[i]) + " * "

            z_str += "x" + str(i + 1)
            if (isNegative):
                z_str += ")"
            previousIsZero = False
        else:
            previousIsZero = True

    print(z_str)
    print("\nsubject to the constrains:\n")
    for i in range(len(b)):
        c_str = ""
        previousIsZero = True
        lastNonZero = False
        for j in range(len(A[i])):
            isNegative = False
            for k in range(j, len(A[i])):

                if (A[i][k] == 0):

                    lastNonZero = True
                else:
                    lastNonZero = False
                    break

            if (not previousIsZero and not lastNonZero):
                c_str += " + "

            if (A[i][j] != 0):
                if (A[i][j] != 1):
                    if (A[i][j] < 0):
                        isNegative = True
                        c_str += "("

                    c_str += str(A[i][j]) + " * "

                c_str += "x" + str(j + 1)
                if (isNegative):
                    c_str += ")"

                previousIsZero = False
            else:
                previousIsZero = True

        c_str += " <= " + str(b[i])
        print(c_str)


def print_result(result: Result):

    if (result.state == State.INAPPLICABLE):
        print("The method is not applicable!")
    elif (result.state == State.UNSOLVED):
        print("Unsolved problem!")
    else:
        print("SOLVED!")
        decVar_str = ""
        decVar_str += "Decision variables: ["
        for i in range(len(result.solution)):
            decVar_str += str(result.solution[i])
            if (i != len(result.solution) - 1):

                decVar_str += ", "

        decVar_str += "]"
        print(decVar_str)
        res_str = ""
        if (result.maximize):
            res_str += "Maximum "
        else:
            res_str += "Minimum "

        res_str += f"objective function value: {result.objective_function_value}"
        print(res_str)
    return 0


def interior_point(
        C: np.array,  # Vector of objective function coefficients
        A: np.array,  # Matrix of constraint coefficients
        b: np.array,  # Vector of right-hand side values of constraints
        x_0: np.array,  # Initial point (vector)
        eps: np.float64 = 0.01,  # Solution accuracy
        alpha: np.float64 = 0.5,  # Step coefficient
        maximizing: bool = True) -> Result:  # Flag for maximization or minimization
    # Check if the method is applicable: the initial point must satisfy the constraints
    print(A)
    if (not np.all(np.dot(A, x_0) <= b) or np.any(x_0 == 0)):
        print(np.dot(A, x_0), b)
        return Result(State.INAPPLICABLE, maximize=maximizing)
    # If the problem is a minimization, invert the coefficients of the objective function
    if (not maximizing):
        C = -C

    m = A.shape[0]  # Number of constraints
    n = A.shape[1]  # Number of variables

    # Initialize variables for the initial iteration
    x_0 = np.concatenate((x_0, b-np.dot(A, x_0)))
    x = x_0  # Solution vector

    A = np.concatenate((A, np.eye(m)), axis=1)
    C = np.concatenate((C, np.zeros(m)))

    iteration = 0  # Iteration counter

    while (True):
        # Create a diagonal matrix from the slack variables vector
        D = np.diag(x)

        # Solve the system of equations to find x*
        x_star = np.dot(np.linalg.inv(D), x)
        A_star = np.dot(A, D)
        C_star = np.dot(D, C)

        # Form the projection matrix
        A_star_transpose = np.transpose(A_star)
        P = np.eye(n+m) - np.dot(np.dot(A_star_transpose, np.linalg.inv(np.dot(A_star, A_star_transpose))), A_star)

        # Calculate the gradient of the objective function
        C_p = np.dot(P, C_star)
        Mu = np.min(C_p)
        if (Mu > 0):
            return Result(State.INAPPLICABLE)

        # Update the value of x* considering the step size and gradient
        x_star = np.ones(n+m) + alpha / np.abs(Mu) * C_p
        x_new = np.dot(D, x_star)
        # Check the stopping criterion based on accuracy
        if np.linalg.norm(x_new - x) <= eps:
            result = np.dot(C, x) if (maximizing) else -np.dot(C, x)
            return Result(State.SOLVED, objective_function_value=result, solution=x, maximize=maximizing)

        iteration += 1

        # Check the iteration limit
        if iteration >= 1000:
            return Result(State.UNSOLVED, maximize=maximizing)

        x = x_new


# TODO 5 tests (from assignment 1) and comparison with simplex and alpha = 0.9
def TEST_CASE_GENERAL():

    print("----------------------------RUNNING_TEST_GENERAL_CASE----------------------------")
    C = np.array([5, 4])
    A = np.array([
        [6, 4],
        [1, 2],
        [-1, 1],
        [0, 1]])
    b = np.array([24, 6, 1, 2])
    x_0 = np.array([1, 1])
    eps = 0.01
    alpha = 0.5
    maximize = True

    print_initial_inputs(C, A, b, x_0, eps, alpha, maximize)
    result = interior_point(C, A, b, x_0, eps, alpha, maximize)

    expected_state = State.SOLVED
    if result.state == expected_state:
        print_result(result)
        return 1
    else:
        state_name = ""
        if result.state == State.UNSOLVED:
            state_name = "UNSOLVED"
        elif result.state == State.INAPPLICABLE:
            state_name = "INAPPLICABLE"
        elif result.state == State.SOLVED:
            state_name = "SOLVED"
        print(f"incorrect state type. expected SOLVED, got {state_name}.")


def TEST_MINIMIZE_CASE():
    print("----------------------------RUNNING_TEST_MINIMIZE_CASE----------------------------")
    C = np.array([-2, 2, -6])
    A = np.array([
        [2, 1, -2],
        [1, 2, 4],
        [1, -1, 2]])
    b = np.array([24, 23, 10])
    x_0 = np.array([1, 1, 1])
    eps = 0.01
    alpha = 0.5
    maximize = False

    print_initial_inputs(C, A, b, x_0, eps, alpha, maximize)
    result = interior_point(C, A, b, x_0, eps, alpha, maximize)

    expected_state = State.SOLVED
    if result.state == expected_state:
        print_result(result)
        return 1
    else:
        state_name = ""
        if result.state == State.UNSOLVED:
            state_name = "UNSOLVED"
        elif result.state == State.INAPPLICABLE:
            state_name = "INAPPLICABLE"
        elif result.state == State.SOLVED:
            state_name = "SOLVED"
        print(f"incorrect state type. expected SOLVED, got {state_name}.")
        return 0


def TEST_WITH_SLACK_CASE():
    print("----------------------------RUNNING_TEST_WITH_SLACK_CASE----------------------------")

    C = np.array([2, -1, 0, -1])
    A = np.array([
        [1, -2, 1, 0],
        [-2, -1, 0, -2],
        [3, 2, 0, 1]])
    b = np.array([10, 18, 36])
    x_0 = np.array([1, 1, 1, 1])
    eps = 0.01
    alpha = 0.5
    maximize = True

    print_initial_inputs(C, A, b, x_0, eps, alpha, maximize)
    result = interior_point(C, A, b, x_0, eps, alpha, maximize)

    expected_state = State.SOLVED
    if result.state == expected_state:
        print_result(result)
        return 1
    else:
        state_name = ""
        if result.state == State.UNSOLVED:
            state_name = "UNSOLVED"
        elif result.state == State.INAPPLICABLE:
            state_name = "INAPPLICABLE"
        elif result.state == State.SOLVED:
            state_name = "SOLVED"
        print(f"incorrect state type. expected SOLVED, got {state_name}.")


def TEST_UNBOUNDED_CASE():
    print("----------------------------RUNNING_TEST_UNBOUNDED_CASE----------------------------")

    C = np.array([2, 1])
    A = np.array([
        [1, -1],
        [2, 0]])
    b = np.array([10, 40])
    x_0 = np.array([1, 1])
    eps = 0.01
    alpha = 0.5
    maximize = True

    print_initial_inputs(C, A, b, x_0, eps, alpha, maximize)
    result = interior_point(C, A, b, x_0, eps, alpha, maximize)

    expected_state = State.SOLVED
    if result.state == expected_state:
        print_result(result)
        return 1
    else:
        state_name = ""
        if result.state == State.UNSOLVED:
            state_name = "UNSOLVED"
        elif result.state == State.INAPPLICABLE:
            state_name = "INAPPLICABLE"
        elif result.state == State.SOLVED:
            state_name = "SOLVED"
        print(f"incorrect state type. expected SOLVED, got {state_name}.")
        return 0


def TEST_UNSOLVABLE_CASE():
    print("----------------------------RUNNING_TEST_UNSOLVABLE_CASE----------------------------")

    C = np.array([5, 4, 0, -5, 13])
    A = np.array([
        [6, 4, 1, 3, 4],
        [1, 2, 0, 0, 2],
        [-1, 0, 0, 10, 0],
        [0, 1, 1, -5, 1]])
    b = np.array([-24, 6, 1, 2])
    x_0 = np.array([-2, -3, -1, -1, 1])
    eps = 0.01
    alpha = 0.5
    maximize = True

    print_initial_inputs(C, A, b, x_0, eps, alpha, maximize)
    result = interior_point(C, A, b, x_0, eps, alpha, maximize)

    expected_state = State.SOLVED
    if result.state == expected_state:
        print_result(result)
        return 1
    else:
        state_name = ""
        if result.state == State.UNSOLVED:
            state_name = "UNSOLVED"
        elif result.state == State.INAPPLICABLE:
            state_name = "INAPPLICABLE"
        elif result.state == State.SOLVED:
            state_name = "SOLVED"
        print(f"incorrect state type. expected SOLVED, got {state_name}.")

        return 0


tests = [
    TEST_CASE_GENERAL(),
    TEST_MINIMIZE_CASE(),
    TEST_WITH_SLACK_CASE(),
    TEST_UNBOUNDED_CASE(),
    TEST_UNSOLVABLE_CASE()
]
tests_passed = 0
for test in tests:
    tests_passed += test


print("----------------------------RESULTS----------------------------")
print("Total number of tests: ", tests.size())
print("Total number of passed tests: ", tests_passed)
